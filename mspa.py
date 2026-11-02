"""
MSPA: Morpho-Spectral Projection Analysis
=========================================
Supervised dimensionality reduction operator for contiguous 1D reflectance,
transmittance, or absorbance spectra, compatible with scikit-learn pipelines.

Author: Lucas Prado Osco et al. (2026)
License: MIT
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

__all__ = ["MSPA", "kneedle_breaking_point", "generate_synthetic_vegetation_spectra"]
__version__ = "1.0.0"


def kneedle_breaking_point(y_curve: np.ndarray) -> int:
    """
    Computes the 1-based index of maximum curvature (Kneedle algorithm)
    on a discrete monotonic curve.
    """
    n_pts = len(y_curve)
    if n_pts <= 2:
        return n_pts
    x_norm = np.linspace(0.0, 1.0, n_pts)
    y_min, y_max = float(np.min(y_curve)), float(np.max(y_curve))
    if np.isclose(y_max, y_min):
        return 1
    y_norm = (y_curve - y_min) / (y_max - y_min)
    diff = (y_norm - x_norm) if (y_curve[-1] >= y_curve[0]) else ((1.0 - y_norm) - x_norm)
    return int(np.argmax(diff)) + 1


def _svd_orthogonal_basis(X_centered: np.ndarray, n_components: int):
    """
    Extracts the top `n_components` right singular vectors (Karhunen-Loeve orthogonal
    functional basis) and explained energy ratios via exact Singular Value Decomposition
    (SVD) with deterministic sign alignment.
    """
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    max_abs_cols = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_abs_cols, range(U.shape[1])])
    signs[signs == 0] = 1.0
    Vt *= signs[:, np.newaxis]

    n_k = min(int(n_components), Vt.shape[0])
    basis_vectors = Vt[:n_k]
    total_energy = float(np.sum(S ** 2))
    energy_ratio = (S[:n_k] ** 2) / max(total_energy, 1e-12)
    return basis_vectors, energy_ratio


class MSPA(BaseEstimator, TransformerMixin):
    """
    Morpho-Spectral Projection Analysis (MSPA).

    Supervised dimensionality reduction operator designed for 1D contiguous spectral
    signals (reflectance, transmittance, or absorbance). Preserves the sequential
    physical ordering of wavelengths through three complementary blocks:
      - Module 1 (Biophysical & Inflection Geometry): Continuum-removed maximum band
        depth and integrated absorption area across major absorption valleys, coupled
        with 1st and 2nd derivative inflection kinematics and plateau contrasts.
      - Module 2 (Local Multi-Order Sub-Window Eigenmodes): Dominant 1D singular
        basis projections of Order 0 (amplitude), Order 1 (slope), and Order 2
        (concavity) extracted independently within contiguous spectral sub-windows.
      - Module 3 (Global Fisher-Weighted Canonical Subspace): Multi-order state space
        [X^(0), X^(1), X^(2)] weighted by inter/intra-class Fisher discriminability
        and projected via Ledoit-Wolf shrinkage-regularized discriminant analysis
        (with rank-safe singular-value-scaled orthogonal completion when C - 1 < K_can).

    Parameters
    ----------
    n_components : int or 'auto', default=64
        Target number of morpho-spectral variables to retain.
    wavelengths : array-like of shape (n_features,), optional (default=None)
        Wavelengths (in nm) or wavenumbers corresponding to the columns of X.
        If None and X is a pandas DataFrame, MSPA automatically parses numerical
        wavelengths from column names (e.g., 'WL_401', '401.0').
    sg_window : int, default=11
        Window length for piecewise Savitzky-Golay derivative calculation.
    sg_poly : int, default=2
        Polynomial order for Savitzky-Golay derivative calculation.
    shrinkage : float, default=0.35
        Analytical covariance shrinkage parameter in (0, 1) for regularized canonical
        discriminant extraction in Module 3.
    n_fallback_windows : int, default=8
        Number of contiguous sub-windows used when the input spectrum lies outside
        the 380-2500 nm optical vegetation domain.
    random_state : int, default=42
        Random seed for deterministic orientation.
    """

    def __init__(
        self,
        n_components: int | str = 64,
        wavelengths=None,
        sg_window: int = 11,
        sg_poly: int = 2,
        shrinkage: float = 0.35,
        n_fallback_windows: int = 8,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.wavelengths = wavelengths
        self.sg_window = sg_window
        self.sg_poly = sg_poly
        self.shrinkage = shrinkage
        self.n_fallback_windows = n_fallback_windows
        self.random_state = random_state

    def _resolve_wavelengths_and_segments(self, X, n_features: int):
        parsed_wl = None
        if self.wavelengths is not None:
            parsed_wl = np.asarray(self.wavelengths, dtype=float)
            if len(parsed_wl) != n_features:
                raise ValueError(
                    f"Length of `wavelengths` ({len(parsed_wl)}) does not match "
                    f"the number of columns in X ({n_features})."
                )
        elif hasattr(X, "columns"):
            extracted = []
            for col in X.columns:
                matches = re.findall(r"[-+]?\d*\.?\d+", str(col))
                if len(matches) > 0:
                    extracted.append(float(matches[-1]))
            if len(extracted) == n_features and len(np.unique(extracted)) == n_features:
                parsed_wl = np.asarray(extracted, dtype=float)

        if parsed_wl is None:
            self.wavelengths_ = np.arange(n_features, dtype=float)
            self.is_optical_vegetation_domain_ = False
        else:
            self.wavelengths_ = parsed_wl
            overlap = np.sum((self.wavelengths_ >= 380.0) & (self.wavelengths_ <= 2500.0))
            self.is_optical_vegetation_domain_ = bool(overlap >= min(30, n_features // 2))

        diffs = np.abs(np.diff(self.wavelengths_))
        med_step = float(np.median(diffs)) if len(diffs) > 0 else 1.0
        gap_threshold = max(5.0 * med_step, 10.0)
        split_pts = np.where(diffs > gap_threshold)[0] + 1
        segments = np.split(np.arange(n_features), split_pts)
        self.contiguous_segments_ = [seg for seg in segments if len(seg) > 0]

    def _compute_piecewise_derivatives(self, X_arr: np.ndarray):
        X_d1 = np.zeros_like(X_arr, dtype=np.float64)
        X_d2 = np.zeros_like(X_arr, dtype=np.float64)
        for seg in self.contiguous_segments_:
            n_seg = len(seg)
            w_len = min(self.sg_window, n_seg if n_seg % 2 == 1 else n_seg - 1)
            if w_len > self.sg_poly and w_len >= 5:
                X_d1[:, seg] = savgol_filter(
                    X_arr[:, seg], window_length=w_len, polyorder=self.sg_poly, deriv=1, axis=1
                )
                X_d2[:, seg] = savgol_filter(
                    X_arr[:, seg], window_length=w_len, polyorder=self.sg_poly, deriv=2, axis=1
                )
            elif n_seg >= 3:
                X_d1[:, seg] = np.gradient(X_arr[:, seg], axis=1)
                X_d2[:, seg] = np.gradient(X_d1[:, seg], axis=1)
        return X_d1, X_d2

    @staticmethod
    def _continuum_removed_depth_and_area(X_win: np.ndarray):
        n_w = X_win.shape[1]
        t = np.linspace(0.0, 1.0, n_w)[None, :]
        continuum = X_win[:, :1] * (1.0 - t) + X_win[:, -1:] * t
        depth_profile = continuum - X_win
        max_depth = np.max(depth_profile, axis=1, keepdims=True)
        area_abs = np.mean(depth_profile, axis=1, keepdims=True)
        return max_depth, area_abs

    def _configure_domains(self, n_features: int):
        wl = self.wavelengths_
        min_bands = 6

        if self.is_optical_vegetation_domain_:
            candidate_windows = [
                ("Blue_Carot", (wl >= 400) & (wl <= 510)),
                ("Green_Anth", (wl >= 511) & (wl <= 620)),
                ("Red_Chl", (wl >= 621) & (wl <= 685)),
                ("Red_Edge", (wl >= 686) & (wl <= 765)),
                ("NIR_Mesophyll", (wl >= 766) & (wl <= 910)),
                ("NIR_Water", (wl >= 911) & (wl <= 1350)),
                ("SWIR1_Lignin", (wl >= 1450) & (wl <= 1790)),
                ("SWIR2_Cellulose", (wl >= 1980) & (wl <= 2450)),
            ]
            candidate_valleys = [
                ("Chl", (wl >= 560) & (wl <= 740)),
                ("NIR_W1", (wl >= 910) & (wl <= 1060)),
                ("NIR_W2", (wl >= 1130) & (wl <= 1280)),
                ("SWIR1", (wl >= 1520) & (wl <= 1760)),
                ("SWIR2", (wl >= 2050) & (wl <= 2360)),
            ]
            windows = [(nm, np.where(mask)[0]) for nm, mask in candidate_windows if int(np.sum(mask)) >= min_bands]
            valleys = [(nm, np.where(mask)[0]) for nm, mask in candidate_valleys if int(np.sum(mask)) >= min_bands]
        else:
            windows, valleys = [], []

        if len(windows) < 2:
            n_w = min(self.n_fallback_windows, max(2, n_features // 10))
            splits = np.array_split(np.arange(n_features), n_w)
            windows = [(f"Win{i+1:02d}", s) for i, s in enumerate(splits) if len(s) >= 3]

        if len(valleys) == 0:
            valleys = [(f"Val{i+1:02d}", idx) for i, (_, idx) in enumerate(windows[:min(5, len(windows))])]

        self.active_windows_ = windows
        self.active_valleys_ = valleys

        re_mask = (wl >= 680) & (wl <= 760) if self.is_optical_vegetation_domain_ else np.zeros(n_features, dtype=bool)
        if int(np.sum(re_mask)) >= min_bands:
            self.inflection_idx_ = np.where(re_mask)[0]
        else:
            self.inflection_idx_ = windows[min(1, len(windows) - 1)][1]

        m_green = (wl >= 530) & (wl <= 570) if self.is_optical_vegetation_domain_ else np.zeros(n_features, dtype=bool)
        m_red = (wl >= 650) & (wl <= 685) if self.is_optical_vegetation_domain_ else np.zeros(n_features, dtype=bool)
        m_nir = (wl >= 780) & (wl <= 900) if self.is_optical_vegetation_domain_ else np.zeros(n_features, dtype=bool)

        if int(np.sum(m_green)) >= 3 and int(np.sum(m_red)) >= 3 and int(np.sum(m_nir)) >= 3:
            self.contrast_triplets_ = (np.where(m_green)[0], np.where(m_red)[0], np.where(m_nir)[0])
        elif len(windows) >= 3:
            self.contrast_triplets_ = (
                windows[0][1],
                windows[1][1],
                windows[min(2, len(windows) - 1)][1],
            )
        else:
            self.contrast_triplets_ = None

    def _extract_module1_biophysical(self, X_arr: np.ndarray, X_d1: np.ndarray, X_d2: np.ndarray):
        feats = []
        names = []
        for v_name, v_idx in self.active_valleys_:
            d_max, a_abs = self._continuum_removed_depth_and_area(X_arr[:, v_idx])
            feats.extend([d_max, a_abs])
            names.extend([f"MSPA_BIO_Dmax_{v_name}", f"MSPA_BIO_Area_{v_name}"])

        inf_idx = self.inflection_idx_
        wl_inf = self.wavelengths_[inf_idx]
        re_d1 = X_d1[:, inf_idx]
        re_d2 = X_d2[:, inf_idx]

        re_max_slope = np.max(re_d1, axis=1, keepdims=True)
        re_mean_slope = np.mean(re_d1, axis=1, keepdims=True)
        re_weights = np.maximum(re_d1, 0.0) + 1e-8
        re_lambda_pos = np.sum(re_weights * wl_inf[None, :], axis=1, keepdims=True) / np.sum(
            re_weights, axis=1, keepdims=True
        )
        re_concav_amp = np.max(re_d2, axis=1, keepdims=True) - np.min(re_d2, axis=1, keepdims=True)
        feats.extend([re_max_slope, re_mean_slope, re_lambda_pos, re_concav_amp])
        names.extend([
            "MSPA_BIO_Inflect_MaxSlope",
            "MSPA_BIO_Inflect_MeanSlope",
            "MSPA_BIO_Inflect_LambdaCentroid",
            "MSPA_BIO_Inflect_ConcavityAmp",
        ])

        if self.contrast_triplets_ is not None:
            idx_g, idx_r, idx_n = self.contrast_triplets_
            mean_g = np.mean(X_arr[:, idx_g], axis=1, keepdims=True)
            mean_r = np.mean(X_arr[:, idx_r], axis=1, keepdims=True)
            mean_n = np.mean(X_arr[:, idx_n], axis=1, keepdims=True)
            feats.extend([mean_n - mean_r, mean_g - mean_r])
            names.extend(["MSPA_BIO_Contrast_NIR_Red", "MSPA_BIO_Contrast_Green_Red"])

        return np.hstack(feats), names

    def _fit_safe_shrinkage_lda(self, X_in: np.ndarray, y: np.ndarray, n_comp: int, shrinkage_val: float):
        n_cls = len(np.unique(y))
        max_k = max(1, min(80, X_in.shape[1], X_in.shape[0] - n_cls - 2))
        mu_in = X_in.mean(axis=0, keepdims=True)
        V_basis, _ = _svd_orthogonal_basis(X_in - mu_in, max_k)
        X_p = (X_in - mu_in) @ V_basis.T
        n_c_eff = min(n_comp, X_p.shape[1], n_cls - 1)
        try:
            lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=shrinkage_val, n_components=n_c_eff)
            Z = lda.fit_transform(X_p, y)
        except Exception:
            lda = LinearDiscriminantAnalysis(solver="svd", n_components=n_c_eff)
            Z = lda.fit_transform(X_p, y)
        sc = StandardScaler().fit(Z)
        return (mu_in, V_basis, lda, sc), sc.transform(Z)

    @staticmethod
    def _compute_fisher_scores(Z_mat: np.ndarray, y: np.ndarray) -> np.ndarray:
        classes = np.unique(y)
        centroids = np.vstack([Z_mat[y == c].mean(axis=0) for c in classes])
        within_std = np.vstack([Z_mat[y == c].std(axis=0) for c in classes]).mean(axis=0) + 1e-6
        return np.std(centroids, axis=0) / within_std

    def fit(self, X, y):
        """Fits the MSPA projection model on training spectra X and class labels y."""
        if y is None:
            raise ValueError("MSPA is a supervised projection operator and requires class labels `y` in `fit(X, y)`.")

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        if X_arr.shape[0] != len(y_arr):
            raise ValueError(f"X ({X_arr.shape[0]} samples) and y ({len(y_arr)} samples) have inconsistent lengths.")

        if y_arr.dtype.kind not in "iu":
            _, y_arr = np.unique(y_arr, return_inverse=True)
        else:
            y_arr = y_arr.astype(int)

        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        self._resolve_wavelengths_and_segments(X, n_features)
        self._configure_domains(n_features)
        X_d1, X_d2 = self._compute_piecewise_derivatives(X_arr)

        # 1. Module 1: Biophysical Continuum-Removal & Inflection Descriptors
        F_bio_tr, bio_names = self._extract_module1_biophysical(X_arr, X_d1, X_d2)
        self.scaler_bio_ = StandardScaler().fit(F_bio_tr)
        Z_bio_tr = self.scaler_bio_.transform(F_bio_tr)
        n_mod1 = Z_bio_tr.shape[1]

        # 2. Module 2: Local Multi-Order Sub-Window Eigenmodes (Rank-1 SVD per Order & Window)
        self.window_projectors_ = []
        F_win_list = []
        win_names = []
        for w_name, w_idx in self.active_windows_:
            sc_0 = StandardScaler().fit(X_arr[:, w_idx])
            sc_1 = StandardScaler().fit(X_d1[:, w_idx])
            sc_2 = StandardScaler().fit(X_d2[:, w_idx])

            X0_w = sc_0.transform(X_arr[:, w_idx])
            X1_w = sc_1.transform(X_d1[:, w_idx])
            X2_w = sc_2.transform(X_d2[:, w_idx])

            v0, _ = _svd_orthogonal_basis(X0_w, 1)
            v1, _ = _svd_orthogonal_basis(X1_w, 1)
            v2, _ = _svd_orthogonal_basis(X2_w, 1)

            self.window_projectors_.append((w_idx, sc_0, v0, sc_1, v1, sc_2, v2))
            F_win_list.append(np.hstack([X0_w @ v0.T, X1_w @ v1.T, X2_w @ v2.T]))
            win_names.extend([
                f"MSPA_WIN_{w_name}_Ord0",
                f"MSPA_WIN_{w_name}_Ord1",
                f"MSPA_WIN_{w_name}_Ord2",
            ])

        F_win_tr = np.hstack(F_win_list)
        self.scaler_win_ = StandardScaler().fit(F_win_tr)
        Z_win_tr = self.scaler_win_.transform(F_win_tr)
        n_mod2 = Z_win_tr.shape[1]

        # 3. Module 3: Global Fisher-Weighted Multi-Order Canonical Subspace
        X_multi_all = np.hstack([X_arr, X_d1, X_d2])
        self.scaler_multi_ = StandardScaler().fit(X_multi_all)
        X_multi_sc = self.scaler_multi_.transform(X_multi_all)

        classes = np.unique(y_arr)
        n_cls = len(classes)
        centroids = np.vstack([X_multi_sc[y_arr == c].mean(axis=0) for c in classes])
        within_std = np.vstack([X_multi_sc[y_arr == c].std(axis=0) for c in classes]).mean(axis=0) + 1e-4
        fisher_w = np.sqrt(np.std(centroids, axis=0) / within_std)
        self.fisher_weights_ = fisher_w / np.mean(fisher_w)
        X_weighted = X_multi_sc * self.fisher_weights_[None, :]
        self.mu_weighted_ = X_weighted.mean(axis=0, keepdims=True)
        X_w_cent = X_weighted - self.mu_weighted_

        if self.n_components == "auto":
            max_probe = min(60, n_samples - 2, X_w_cent.shape[1])
            _, ev_probe = _svd_orthogonal_basis(X_w_cent, max_probe)
            cum_var = np.cumsum(ev_probe)
            n_can = int(np.clip(kneedle_breaking_point(cum_var), 12, 36))
            target_total = n_mod1 + n_mod2 + n_can
        elif isinstance(self.n_components, (int, np.integer)) and self.n_components >= 1:
            target_total = int(self.n_components)
            n_can = max(4, target_total - (n_mod1 + n_mod2))
        else:
            raise ValueError("`n_components` must be a positive integer or 'auto'.")

        self.n_can_ = n_can
        self.is_many_class_ = bool((n_cls - 1) >= n_can)

        if self.is_many_class_:
            n_pre = max(n_can, min(140, n_samples - n_cls - 5, X_w_cent.shape[1]))
            self.V_pre_, _ = _svd_orthogonal_basis(X_w_cent, n_pre)
            X_pre_tr = X_w_cent @ self.V_pre_.T
            try:
                self.global_lda_ = LinearDiscriminantAnalysis(
                    solver="eigen", shrinkage=self.shrinkage, n_components=n_can
                )
                Z_can_raw = self.global_lda_.fit_transform(X_pre_tr, y_arr)
            except Exception:
                self.global_lda_ = LinearDiscriminantAnalysis(solver="svd", n_components=n_can)
                Z_can_raw = self.global_lda_.fit_transform(X_pre_tr, y_arr)
            self.scaler_can_ = StandardScaler().fit(Z_can_raw)
            Z_can_tr = self.scaler_can_.transform(Z_can_raw)
        else:
            n_per_order = min(n_cls - 1, max(1, min(4, n_can // 6)))
            self.sc_ord0_ = StandardScaler().fit(X_arr)
            self.sc_ord1_ = StandardScaler().fit(X_d1)
            self.sc_ord2_ = StandardScaler().fit(X_d2)

            self.lda_ord0_, z_lda0 = self._fit_safe_shrinkage_lda(
                self.sc_ord0_.transform(X_arr), y_arr, n_comp=n_per_order, shrinkage_val=0.30
            )
            self.lda_ord1_, z_lda1 = self._fit_safe_shrinkage_lda(
                self.sc_ord1_.transform(X_d1), y_arr, n_comp=n_per_order, shrinkage_val=0.40
            )
            self.lda_ord2_, z_lda2 = self._fit_safe_shrinkage_lda(
                self.sc_ord2_.transform(X_d2), y_arr, n_comp=n_per_order, shrinkage_val=0.50
            )
            self.lda_comb_, z_ldac = self._fit_safe_shrinkage_lda(
                X_weighted, y_arr, n_comp=n_per_order, shrinkage_val=self.shrinkage
            )

            z_sup_all = np.hstack([z_lda0, z_lda1, z_lda2, z_ldac])
            n_rem = max(1, n_can - z_sup_all.shape[1])

            self.V_fisher_orth_, ev_ratio = _svd_orthogonal_basis(X_w_cent, n_rem)
            z_forth = X_w_cent @ self.V_fisher_orth_.T
            std_forth = np.std(z_forth, axis=0, keepdims=True) + 1e-6
            decay_weights = np.sqrt(ev_ratio / ev_ratio[0])[None, :]
            self.forth_scale_ = std_forth / np.maximum(decay_weights, 0.25)
            Z_can_tr = np.hstack([z_sup_all, z_forth / self.forth_scale_])

        can_names = [f"MSPA_CAN_{i+1:02d}" for i in range(Z_can_tr.shape[1])]
        Z_full_tr = np.hstack([Z_can_tr, Z_win_tr, Z_bio_tr])
        all_names = np.asarray(can_names + win_names + bio_names, dtype=object)

        if Z_full_tr.shape[1] > target_total:
            f_scores = self._compute_fisher_scores(Z_full_tr, y_arr)
            top_idx = np.argsort(f_scores)[::-1][:target_total]
            self.selected_indices_ = np.sort(top_idx)
        else:
            self.selected_indices_ = None

        self.feature_names_out_ = (
            all_names[self.selected_indices_] if self.selected_indices_ is not None else all_names
        )
        self.n_components_ = len(self.feature_names_out_)
        return self

    def transform(self, X) -> np.ndarray:
        """Projects spectra X into the fitted MSPA morpho-spectral subspace."""
        if not hasattr(self, "n_components_"):
            raise ValueError("This MSPA instance is not fitted yet. Call 'fit' before using 'transform'.")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but MSPA expects {self.n_features_in_} features."
            )

        X_d1, X_d2 = self._compute_piecewise_derivatives(X_arr)

        # Module 1
        F_bio, _ = self._extract_module1_biophysical(X_arr, X_d1, X_d2)
        Z_bio = self.scaler_bio_.transform(F_bio)

        # Module 2
        F_win_list = []
        for w_idx, sc_0, v0, sc_1, v1, sc_2, v2 in self.window_projectors_:
            F_win_list.append(np.hstack([
                sc_0.transform(X_arr[:, w_idx]) @ v0.T,
                sc_1.transform(X_d1[:, w_idx]) @ v1.T,
                sc_2.transform(X_d2[:, w_idx]) @ v2.T,
            ]))
        Z_win = self.scaler_win_.transform(np.hstack(F_win_list))

        # Module 3
        X_multi_all = np.hstack([X_arr, X_d1, X_d2])
        X_weighted = self.scaler_multi_.transform(X_multi_all) * self.fisher_weights_[None, :]
        X_w_cent = X_weighted - self.mu_weighted_

        if self.is_many_class_:
            Z_can = self.scaler_can_.transform(
                self.global_lda_.transform(X_w_cent @ self.V_pre_.T)
            )
        else:
            def _apply_lda_pipe(pipe, X_in):
                mu_p, V_p, lda_p, sc_p = pipe
                return sc_p.transform(lda_p.transform((X_in - mu_p) @ V_p.T))

            z0 = _apply_lda_pipe(self.lda_ord0_, self.sc_ord0_.transform(X_arr))
            z1 = _apply_lda_pipe(self.lda_ord1_, self.sc_ord1_.transform(X_d1))
            z2 = _apply_lda_pipe(self.lda_ord2_, self.sc_ord2_.transform(X_d2))
            zc = _apply_lda_pipe(self.lda_comb_, X_weighted)
            z_forth = (X_w_cent @ self.V_fisher_orth_.T) / self.forth_scale_
            Z_can = np.hstack([z0, z1, z2, zc, z_forth])

        Z_full = np.hstack([Z_can, Z_win, Z_bio]).astype(np.float32)
        if self.selected_indices_ is not None:
            return Z_full[:, self.selected_indices_]
        return Z_full

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Returns output feature names for transformation (scikit-learn standard API)."""
        if not hasattr(self, "feature_names_out_"):
            raise ValueError("This MSPA instance is not fitted yet.")
        return self.feature_names_out_


def generate_synthetic_vegetation_spectra(
    n_samples: int = 450,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generates physically realistic synthetic foliar reflectance spectra (SNV-stabilized)
    across 1,580 bands (401-1330, 1491-1770, 2031-2400 nm) for 3 physiological classes:
      - Class 0: Healthy Control
      - Class 1: Early Chlorosis & Red-Edge Blue-Shift (~2.8 nm inflection shift)
      - Class 2: Vascular & Cell-Wall Alteration (SWIR concavity & water-band shift)

    Designed for standalone tutorials and Google Colab demonstrations without requiring
    proprietary field data.
    """
    rng = np.random.default_rng(random_state)
    wls = np.array(
        list(range(401, 1331)) + list(range(1491, 1771)) + list(range(2031, 2401)),
        dtype=float,
    )
    n_bands = len(wls)
    y = np.repeat([0, 1, 2], n_samples // 3)
    if len(y) < n_samples:
        y = np.concatenate([y, np.zeros(n_samples - len(y), dtype=int)])
    y = rng.permutation(y)

    X_raw = np.zeros((n_samples, n_bands), dtype=np.float64)
    for i in range(n_samples):
        cls = y[i]
        nir_albedo = rng.normal(0.52, 0.06)
        swir1_level = nir_albedo * rng.normal(0.58, 0.05)
        swir2_level = nir_albedo * rng.normal(0.30, 0.04)
        global_tilt = rng.normal(0.0, 0.015) * ((wls - 1200.0) / 1000.0)

        re_center = 716.0 - (2.8 if cls == 1 else (0.8 if cls == 2 else 0.0)) + rng.normal(0, 1.1)
        re_width = 14.5 + (2.2 if cls == 1 else (-1.2 if cls == 2 else 0.0)) + rng.normal(0, 0.7)
        chl_depth = 0.095 - (0.012 if cls == 1 else 0.0) + rng.normal(0, 0.008)
        swir_narrow_dip = 0.038 + (0.016 if cls == 2 else 0.0) + rng.normal(0, 0.005)
        swir_dip_width = 22.0 if cls != 2 else 14.0

        green_peak = chl_depth * np.exp(-0.5 * ((wls - 552.0) / 26.0) ** 2)
        red_edge_step = nir_albedo / (1.0 + np.exp(-(wls - re_center) / re_width))
        w1_abs = 0.035 * np.exp(-0.5 * ((wls - 975.0) / 28.0) ** 2)
        w2_abs = 0.055 * np.exp(-0.5 * ((wls - 1195.0) / 34.0) ** 2)
        swir1_env = np.where(
            wls >= 1490, (swir1_level - nir_albedo) + 0.08 * np.sin((wls - 1491) / 280.0 * np.pi), 0.0
        )
        swir1_lignin = swir_narrow_dip * np.exp(-0.5 * ((wls - 1685.0) / swir_dip_width) ** 2)
        swir2_env = np.where(
            wls >= 2030, (swir2_level - swir1_level) + 0.06 * np.sin((wls - 2031) / 370.0 * np.pi), 0.0
        )
        swir2_cell = (0.025 + (0.010 if cls == 1 else -0.008 if cls == 2 else 0.0)) * np.exp(
            -0.5 * ((wls - 2180.0) / 19.0) ** 2
        )

        spec = (
            0.04 + green_peak + red_edge_step - w1_abs - w2_abs
            + swir1_env - swir1_lignin + swir2_env - swir2_cell + global_tilt
            + rng.normal(0, 0.0035, size=n_bands)
        )
        X_raw[i] = spec

    b1, b2, b3 = (wls <= 1330), (wls >= 1491) & (wls <= 1770), (wls >= 2031)
    X_sg = np.zeros_like(X_raw)
    for mask in [b1, b2, b3]:
        X_sg[:, mask] = savgol_filter(X_raw[:, mask], window_length=10, polyorder=2, deriv=0, axis=1)
    X_snv = (X_sg - X_sg.mean(axis=1, keepdims=True)) / (X_sg.std(axis=1, keepdims=True) + 1e-12)

    cols = [f"WL_{int(w)}" for w in wls]
    return pd.DataFrame(X_snv, columns=cols), y, wls