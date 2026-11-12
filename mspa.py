"""
MSPA: Morpho-Spectral Projection Analysis
=========================================
Supervised dimensionality reduction operator for contiguous 1D plant reflectance
spectra across the VNIR-SWIR domain (350-2500 nm), compatible with scikit-learn.

Reference:
Osco, L. P., Silva, M. S., Silva, F. V., Zhao, L., Ramos, A. P. M.,
Ferreira, E. J., Bendahan, A. B., & Jorge, L. A. C. (2026).
Morpho-Spectral Projection Analysis (MSPA) for Plant Spectroscopy.
Zenodo. https://doi.org/10.5281/zenodo.23890722

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
__version__ = "1.0.1"


def kneedle_breaking_point(y_curve: np.ndarray) -> int:
    """
    Computes the 1-based index of maximum perpendicular deflection distance
    (Kneedle algorithm) on a discrete cumulative energy curve.
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
    Extracts the top `n_components` right singular vectors and explained energy
    ratios via Singular Value Decomposition (SVD) with deterministic sign alignment.
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
    Morpho-Spectral Projection Analysis (MSPA) for Plant Spectroscopy.

    Supervised dimensionality reduction operator designed for contiguous 1D plant
    reflectance spectra across the VNIR-SWIR domain (350-2500 nm). Breaks column-
    permutation invariance through three complementary mathematical modules:
      - Module 1 (Biophysical Hull & Inflection Kinematics, 16 vars): Additive
        continuum-removed maximum band depth (D_max) and integrated absorption
        area (A_abs) across 5 major foliar valleys (V1-V5), coupled with 1st and
        2nd derivative red-edge inflection kinematics (lambda_RE) and plateau contrasts.
      - Module 2 (Local Multi-Order Sub-Window Eigenmodes, 24 vars): Dominant rank-1
        SVD projections of Order 0 (amplitude), Order 1 (local slope), and Order 2
        (local concavity) extracted independently within 8 radiative sub-windows (W1-W8).
      - Module 3 (Global Fisher-Weighted Canonical Subspace, K_can vars): Coupled
        differential state space [X^(0), X^(1), X^(2)] weighted by inter/intra-class
        Fisher discriminability (omega_q) and projected via analytical Ledoit-Wolf
        shrinkage discriminant analysis with rank-safe orthogonal completion.
    """

    def __init__(
        self,
        n_components: int | str = 64,
        wavelengths=None,
        sg_window: int = 11,
        sg_poly: int = 2,
        shrinkage: float = 0.35,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.wavelengths = wavelengths
        self.sg_window = sg_window
        self.sg_poly = sg_poly
        self.shrinkage = shrinkage
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
            self.wavelengths_ = np.linspace(400.0, 2500.0, n_features)
            self.is_optical_vegetation_domain_ = True
        else:
            self.wavelengths_ = parsed_wl
            overlap = np.sum((self.wavelengths_ >= 350.0) & (self.wavelengths_ <= 2500.0))
            self.is_optical_vegetation_domain_ = bool(overlap >= min(20, n_features // 2))

        diffs = np.abs(np.diff(self.wavelengths_))
        self.median_step_ = float(np.median(diffs)) if len(diffs) > 0 else 1.0
        gap_threshold = max(5.0 * self.median_step_, 10.0)
        split_pts = np.where(diffs > gap_threshold)[0] + 1
        segments = np.split(np.arange(n_features), split_pts)
        self.contiguous_segments_ = [seg for seg in segments if len(seg) > 0]

    def _compute_piecewise_derivatives(self, X_arr: np.ndarray):
        X_d1 = np.zeros_like(X_arr, dtype=np.float64)
        X_d2 = np.zeros_like(X_arr, dtype=np.float64)
        delta = max(self.median_step_, 1e-6)
        for seg in self.contiguous_segments_:
            n_seg = len(seg)
            w_len = min(self.sg_window, n_seg if n_seg % 2 == 1 else n_seg - 1)
            if w_len > self.sg_poly and w_len >= 5:
                X_d1[:, seg] = savgol_filter(
                    X_arr[:, seg], window_length=w_len, polyorder=self.sg_poly,
                    deriv=1, delta=delta, axis=1
                )
                X_d2[:, seg] = savgol_filter(
                    X_arr[:, seg], window_length=w_len, polyorder=self.sg_poly,
                    deriv=2, delta=delta, axis=1
                )
            elif n_seg >= 3:
                X_d1[:, seg] = np.gradient(X_arr[:, seg], delta, axis=1)
                X_d2[:, seg] = np.gradient(X_d1[:, seg], delta, axis=1)
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

        candidate_windows = [
            ("Blue_Carot",      (wl >= 400) & (wl <= 510),  "400-510 nm"),
            ("Green_Anth",      (wl >= 511) & (wl <= 620),  "511-620 nm"),
            ("Red_Chl",         (wl >= 621) & (wl <= 685),  "621-685 nm"),
            ("Red_Edge",        (wl >= 686) & (wl <= 765),  "686-765 nm"),
            ("NIR_Mesophyll",   (wl >= 766) & (wl <= 910),  "766-910 nm"),
            ("NIR_Water",       (wl >= 911) & (wl <= 1350), "911-1350 nm"),
            ("SWIR1_Lignin",    (wl >= 1450) & (wl <= 1790),"1450-1790 nm"),
            ("SWIR2_Cellulose", (wl >= 1980) & (wl <= 2450),"1980-2450 nm"),
        ]
        candidate_valleys = [
            ("Chl",    (wl >= 560) & (wl <= 740),   "560-740 nm"),
            ("NIR_W1", (wl >= 910) & (wl <= 1060),  "910-1060 nm"),
            ("NIR_W2", (wl >= 1130) & (wl <= 1280), "1130-1280 nm"),
            ("SWIR1",  (wl >= 1520) & (wl <= 1760), "1520-1760 nm"),
            ("SWIR2",  (wl >= 2050) & (wl <= 2360), "2050-2360 nm"),
        ]

        windows = [
            (nm, np.where(mask)[0], rng_str)
            for nm, mask, rng_str in candidate_windows
            if int(np.sum(mask)) >= min_bands
        ]
        valleys = [
            (nm, np.where(mask)[0], rng_str)
            for nm, mask, rng_str in candidate_valleys
            if int(np.sum(mask)) >= min_bands
        ]

        if len(windows) < 2:
            splits = np.array_split(np.arange(n_features), min(8, max(2, n_features // 10)))
            windows = [
                (f"Win{i+1:02d}", s, f"{wl[s[0]]:.0f}-{wl[s[-1]]:.0f} nm")
                for i, s in enumerate(splits) if len(s) >= 3
            ]
        if len(valleys) == 0:
            valleys = [
                (f"Val{i+1:02d}", idx, rng_str)
                for i, (_, idx, rng_str) in enumerate(windows[:min(5, len(windows))])
            ]

        self.active_windows_ = windows
        self.active_valleys_ = valleys

        re_mask = (wl >= 680) & (wl <= 760)
        if int(np.sum(re_mask)) >= min_bands:
            self.inflection_idx_ = np.where(re_mask)[0]
        else:
            self.inflection_idx_ = windows[min(1, len(windows) - 1)][1]

        m_green = (wl >= 530) & (wl <= 570)
        m_red   = (wl >= 650) & (wl <= 685)
        m_nir   = (wl >= 780) & (wl <= 900)

        if int(np.sum(m_green)) >= 3 and int(np.sum(m_red)) >= 3 and int(np.sum(m_nir)) >= 3:
            self.contrast_triplets_ = (np.where(m_green)[0], np.where(m_red)[0], np.where(m_nir)[0])
        elif len(windows) >= 3:
            self.contrast_triplets_ = (windows[0][1], windows[1][1], windows[min(2, len(windows) - 1)][1])
        else:
            self.contrast_triplets_ = None

    def _extract_module1_biophysical(self, X_arr: np.ndarray, X_d1: np.ndarray, X_d2: np.ndarray):
        feats, names, meta = [], [], []
        for v_name, v_idx, v_rng in self.active_valleys_:
            d_max, a_abs = self._continuum_removed_depth_and_area(X_arr[:, v_idx])
            feats.extend([d_max, a_abs])
            names.extend([f"MSPA_BIO_Dmax_{v_name}", f"MSPA_BIO_Area_{v_name}"])
            meta.extend([
                ("Module 1 (Biophysical Hull)", "Order 0 (Continuum Depth)", v_rng),
                ("Module 1 (Biophysical Hull)", "Order 0 (Continuum Area)",  v_rng),
            ])

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
        re_rng = f"{wl_inf[0]:.0f}-{wl_inf[-1]:.0f} nm"
        meta.extend([
            ("Module 1 (Biophysical Hull)", "Order 1 (Max Slope S_max)",          re_rng),
            ("Module 1 (Biophysical Hull)", "Order 1 (Mean Slope S_bar)",         re_rng),
            ("Module 1 (Biophysical Hull)", "Order 1 (Inflection Centroid λ_RE)", re_rng),
            ("Module 1 (Biophysical Hull)", "Order 2 (Concavity Wave C_amp)",     re_rng),
        ])

        if self.contrast_triplets_ is not None:
            idx_g, idx_r, idx_n = self.contrast_triplets_
            mean_g = np.mean(X_arr[:, idx_g], axis=1, keepdims=True)
            mean_r = np.mean(X_arr[:, idx_r], axis=1, keepdims=True)
            mean_n = np.mean(X_arr[:, idx_n], axis=1, keepdims=True)
            feats.extend([mean_n - mean_r, mean_g - mean_r])
            names.extend(["MSPA_BIO_Contrast_NIR_Red", "MSPA_BIO_Contrast_Green_Red"])
            meta.extend([
                ("Module 1 (Biophysical Hull)", "Order 0 (Plateau Contrast)", "780-900 vs 650-685 nm"),
                ("Module 1 (Biophysical Hull)", "Order 0 (Peak Contrast)",    "530-570 vs 650-685 nm"),
            ])

        return np.hstack(feats), names, meta

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
        """Fits the MSPA projection operator on training spectra X and discrete class labels y."""
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

        # 1. Module 1: Biophysical Continuum-Removed & Inflection Descriptors (Variables 1..16)
        F_bio_tr, bio_names, bio_meta = self._extract_module1_biophysical(X_arr, X_d1, X_d2)
        self.scaler_bio_ = StandardScaler().fit(F_bio_tr)
        Z_bio_tr = self.scaler_bio_.transform(F_bio_tr)
        n_mod1 = Z_bio_tr.shape[1]

        # 2. Module 2: Local Multi-Order Sub-Window Eigenmodes (Variables 17..40)
        self.window_projectors_ = []
        F_win_list, win_names, win_meta = [], [], []
        for w_name, w_idx, w_rng in self.active_windows_:
            sc_0 = StandardScaler().fit(X_arr[:, w_idx])
            sc_1 = StandardScaler().fit(X_d1[:, w_idx])
            sc_2 = StandardScaler().fit(X_d2[:, w_idx])

            X0_w = sc_0.transform(X_arr[:, w_idx])
            X1_w = sc_1.transform(X_d1[:, w_idx])
            X2_w = sc_2.transform(X_d2[:, w_idx])

            v0, ev0 = _svd_orthogonal_basis(X0_w, 1)
            v1, ev1 = _svd_orthogonal_basis(X1_w, 1)
            v2, ev2 = _svd_orthogonal_basis(X2_w, 1)

            self.window_projectors_.append((w_name, w_idx, sc_0, v0, float(ev0[0]), sc_1, v1, float(ev1[0]), sc_2, v2, float(ev2[0])))
            F_win_list.append(np.hstack([X0_w @ v0.T, X1_w @ v1.T, X2_w @ v2.T]))
            win_names.extend([
                f"MSPA_WIN_{w_name}_Ord0",
                f"MSPA_WIN_{w_name}_Ord1",
                f"MSPA_WIN_{w_name}_Ord2",
            ])
            win_meta.extend([
                ("Module 2 (Local Window SVD)", "Order 0 (Amplitude Mode)", w_rng),
                ("Module 2 (Local Window SVD)", "Order 1 (Slope Mode)",     w_rng),
                ("Module 2 (Local Window SVD)", "Order 2 (Concavity Mode)", w_rng),
            ])

        F_win_tr = np.hstack(F_win_list)
        self.scaler_win_ = StandardScaler().fit(F_win_tr)
        Z_win_tr = self.scaler_win_.transform(F_win_tr)
        n_mod2 = Z_win_tr.shape[1]

        # 3. Module 3: Global Fisher-Weighted Multi-Order Canonical Subspace (Variables 41..64)
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

        max_probe = min(60, n_samples - 2, X_w_cent.shape[1])
        _, ev_probe = _svd_orthogonal_basis(X_w_cent, max_probe)
        cum_var = np.cumsum(ev_probe)
        y_norm = (cum_var - cum_var[0]) / max(cum_var[-1] - cum_var[0], 1e-12)
        x_norm = np.linspace(0.0, 1.0, len(y_norm))
        self.kneedle_energy_curve_ = y_norm
        self.kneedle_diff_curve_ = y_norm - x_norm
        self.kneedle_k_can_ = int(np.clip(kneedle_breaking_point(cum_var), 12, 36))

        if self.n_components == "auto":
            n_can = self.kneedle_k_can_
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
        full_rng_str = f"{self.wavelengths_[0]:.0f}-{self.wavelengths_[-1]:.0f} nm"
        can_meta = [
            ("Module 3 (Global Canonical)", "Multi-Order (0, 1, 2)", full_rng_str)
            for _ in range(Z_can_tr.shape[1])
        ]

        Z_full_tr = np.hstack([Z_bio_tr, Z_win_tr, Z_can_tr])
        all_names = np.asarray(bio_names + win_names + can_names, dtype=object)
        all_meta = bio_meta + win_meta + can_meta
        all_f_scores = self._compute_fisher_scores(Z_full_tr, y_arr)

        if Z_full_tr.shape[1] > target_total:
            top_idx = np.argsort(all_f_scores)[::-1][:target_total]
            self.selected_indices_ = np.sort(top_idx)
        else:
            self.selected_indices_ = None

        if self.selected_indices_ is not None:
            self.feature_names_out_ = all_names[self.selected_indices_]
            self.feature_fisher_scores_ = all_f_scores[self.selected_indices_]
            self.feature_metadata_ = [all_meta[i] for i in self.selected_indices_]
        else:
            self.feature_names_out_ = all_names
            self.feature_fisher_scores_ = all_f_scores
            self.feature_metadata_ = all_meta

        self.n_components_ = len(self.feature_names_out_)
        return self

    def transform(self, X) -> np.ndarray:
        """Projects unseen spectra X into the fitted MSPA morpho-spectral subspace."""
        if not hasattr(self, "n_components_"):
            raise ValueError("This MSPA instance is not fitted yet. Call 'fit' before using 'transform'.")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but MSPA expects {self.n_features_in_} features."
            )

        X_d1, X_d2 = self._compute_piecewise_derivatives(X_arr)

        # Module 1 (Variables 1..16)
        F_bio, _, _ = self._extract_module1_biophysical(X_arr, X_d1, X_d2)
        Z_bio = self.scaler_bio_.transform(F_bio)

        # Module 2 (Variables 17..40)
        F_win_list = []
        for _, w_idx, sc_0, v0, _, sc_1, v1, _, sc_2, v2, _ in self.window_projectors_:
            F_win_list.append(np.hstack([
                sc_0.transform(X_arr[:, w_idx]) @ v0.T,
                sc_1.transform(X_d1[:, w_idx]) @ v1.T,
                sc_2.transform(X_d2[:, w_idx]) @ v2.T,
            ]))
        Z_win = self.scaler_win_.transform(np.hstack(F_win_list))

        # Module 3 (Variables 41..64)
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

        Z_full = np.hstack([Z_bio, Z_win, Z_can]).astype(np.float32)
        if self.selected_indices_ is not None:
            return Z_full[:, self.selected_indices_]
        return Z_full

    def compute_structure_loadings(self, X) -> dict[str, np.ndarray]:
        """
        Computes the Multi-Order Structure Loading matrices Lambda^(m) in R^(D x K*)
        for m in {0, 1, 2} and the composite loading footprint |Lambda^comp|.
        """
        if not hasattr(self, "n_components_"):
            raise ValueError("Fit MSPA before calling `compute_structure_loadings(X)`.")

        X_arr = np.asarray(X, dtype=np.float64)
        X_d1, X_d2 = self._compute_piecewise_derivatives(X_arr)
        Z = np.asarray(self.transform(X_arr), dtype=np.float64)

        Z_std = StandardScaler().fit_transform(Z)
        n_s = max(1, X_arr.shape[0] - 1)

        L0 = (StandardScaler().fit_transform(X_arr).T @ Z_std) / n_s
        L1 = (StandardScaler().fit_transform(X_d1).T @ Z_std) / n_s
        L2 = (StandardScaler().fit_transform(X_d2).T @ Z_std) / n_s
        L_comp = np.sqrt((L0 ** 2 + L1 ** 2 + L2 ** 2) / 3.0)

        return {
            "Order_0": L0,
            "Order_1": L1,
            "Order_2": L2,
            "Composite": L_comp,
        }

    def get_variable_catalog(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame cataloging all retained MSPA variables,
        their internal module, differential order, spectral support, and Fisher score.
        """
        if not hasattr(self, "n_components_"):
            raise ValueError("Fit MSPA before calling `get_variable_catalog()`.")

        rows = []
        for idx, (var_name, (mod_name, ord_name, w_rng), f_sc) in enumerate(
            zip(self.feature_names_out_, self.feature_metadata_, self.feature_fisher_scores_)
        ):
            rows.append({
                "Index": idx + 1,
                "Variable": var_name,
                "Module": mod_name,
                "Differential_Order": ord_name,
                "Spectral_Support": w_rng,
                "Training_Fisher_Score": round(float(f_sc), 4),
            })
        return pd.DataFrame(rows)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Returns output feature names for transformation (scikit-learn standard API)."""
        if not hasattr(self, "feature_names_out_"):
            raise ValueError("This MSPA instance is not fitted yet.")
        return self.feature_names_out_


def generate_synthetic_vegetation_spectra(
    n_samples: int = 450,
    excise_water_vapor: bool = False,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generates physically realistic synthetic foliar reflectance spectra (rho in [0, 1],
    Savitzky-Golay smoothed) across the VNIR-SWIR vegetation domain (400-2500 nm)
    spanning 3 physiological states:
      - Class 0: Healthy Control
      - Class 1: Chlorotic Stress (Pigment loss & Red-Edge blue-shift of ~2.8 nm)
      - Class 2: Structural/Hydric Stress (SWIR lignin/cellulose concavity shift)

    Parameters
    ----------
    n_samples : int, default=450
        Total number of synthetic foliar spectra to generate.
    excise_water_vapor : bool, default=False
        If False (default), returns continuous laboratory/leaf-clip spectra across
        400-2500 nm (D = 2,101 bands, including the 1450 nm and 1940 nm water bands).
        If True, excises the deep atmospheric water-vapor windows (1331-1490 nm and
        1771-2030 nm) and detector edges to simulate solar field spectra (D = 1,580 bands).
    random_state : int, default=42
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(random_state)
    wls_full = np.arange(400.0, 2501.0, 1.0)
    n_full = len(wls_full)

    y = np.repeat([0, 1, 2], n_samples // 3)
    if len(y) < n_samples:
        y = np.concatenate([y, np.zeros(n_samples - len(y), dtype=int)])
    y = rng.permutation(y)

    X_raw = np.zeros((n_samples, n_full), dtype=np.float64)
    for i in range(n_samples):
        cls = y[i]
        nir_albedo = rng.normal(0.50, 0.04)
        global_tilt = rng.normal(0.0, 0.008) * ((wls_full - 1200.0) / 1000.0)

        re_center = 716.0 - (2.8 if cls == 1 else (0.8 if cls == 2 else 0.0)) + rng.normal(0, 0.9)
        re_width = 14.5 + (2.2 if cls == 1 else (-1.2 if cls == 2 else 0.0)) + rng.normal(0, 0.5)
        chl_depth = 0.115 - (0.018 if cls == 1 else 0.0) + rng.normal(0, 0.006)
        swir1_lignin_dip = 0.042 + (0.020 if cls == 2 else 0.0) + rng.normal(0, 0.004)
        swir1_dip_width = 22.0 if cls != 2 else 14.0
        swir2_cell_dip = 0.028 + (0.012 if cls == 1 else (-0.010 if cls == 2 else 0.0)) + rng.normal(0, 0.003)

        green_peak = chl_depth * np.exp(-0.5 * ((wls_full - 552.0) / 24.0) ** 2)
        chl_well = -0.035 * np.exp(-0.5 * ((wls_full - 672.0) / 18.0) ** 2)
        red_edge_ramp = nir_albedo / (1.0 + np.exp(-(wls_full - re_center) / re_width))

        swir_decay = 1.0 - 0.36 * (1.0 / (1.0 + np.exp(-(wls_full - 1380.0) / 85.0))) \
                         - 0.38 * (1.0 / (1.0 + np.exp(-(wls_full - 1880.0) / 75.0)))

        w1_abs = 0.032 * np.exp(-0.5 * ((wls_full - 975.0) / 28.0) ** 2)
        w2_abs = 0.052 * np.exp(-0.5 * ((wls_full - 1195.0) / 34.0) ** 2)
        h2o_1450 = 0.165 * np.exp(-0.5 * ((wls_full - 1450.0) / 42.0) ** 2)
        lignin_1685 = swir1_lignin_dip * np.exp(-0.5 * ((wls_full - 1685.0) / swir1_dip_width) ** 2)
        h2o_1940 = 0.145 * np.exp(-0.5 * ((wls_full - 1940.0) / 45.0) ** 2)
        cell_2180 = swir2_cell_dip * np.exp(-0.5 * ((wls_full - 2180.0) / 20.0) ** 2)

        spec = (
            0.045 + green_peak + chl_well + red_edge_ramp * swir_decay
            - w1_abs - w2_abs - h2o_1450 - lignin_1685 - h2o_1940 - cell_2180
            + global_tilt + rng.normal(0, 0.0028, size=n_full)
        )
        X_raw[i] = spec

    if excise_water_vapor:
        keep_mask = (
            ((wls_full >= 401) & (wls_full <= 1330))
            | ((wls_full >= 1491) & (wls_full <= 1770))
            | ((wls_full >= 2031) & (wls_full <= 2400))
        )
        wls = wls_full[keep_mask]
        X_sub = X_raw[:, keep_mask]
        b1, b2, b3 = (wls <= 1330), (wls >= 1491) & (wls <= 1770), (wls >= 2031)
        X_sg = np.zeros_like(X_sub)
        for bm in [b1, b2, b3]:
            X_sg[:, bm] = savgol_filter(X_sub[:, bm], window_length=11, polyorder=2, deriv=0, axis=1)
    else:
        wls = wls_full
        X_sg = savgol_filter(X_raw, window_length=11, polyorder=2, deriv=0, axis=1)

    X_refl = np.clip(X_sg, 0.005, 0.995)
    cols = [f"WL_{int(w)}" for w in wls]
    return pd.DataFrame(X_refl, columns=cols), y, wls
