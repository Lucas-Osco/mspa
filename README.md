# MSPA: Morpho-Spectral Projection Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn compatible](https://img.shields.io/badge/scikit--learn-compatible-orange.svg)](https://scikit-learn.org/)

**Morpho-Spectral Projection Analysis (`MSPA`)** is a supervised dimensionality reduction operator designed specifically for contiguous one-dimensional reflectance, transmittance, and absorbance spectra ($\mathcal{T}: \mathbb{R}^D \to \mathbb{R}^K$).

Standard dimensionality reduction methods, such as Principal Component Analysis (`PCA`), Partial Least Squares (`PLS-DA`), `UMAP`, and Self-Organizing Maps (`SOM`), are **column-permutation invariant**: they treat adjacent wavelengths as unordered tabular columns, discarding the sequential differential topology of the spectral curve. `MSPA` overcomes this limitation while adhering strictly to the `scikit-learn` transformer API (`fit`, `transform`, `fit_transform`, `Pipeline`, and `set_output(transform="pandas")`).

---

## Architecture of the MSPA Operator

Given an input spectral matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$ ordered along wavelengths $\lambda_1 < \lambda_2 < \dots < \lambda_D$, `MSPA` projects the spectrum into $K$ morpho-spectral variables (default $K^* = 64$) structured into three complementary blocks:

1. **Module 1 — Beer-Lambert Continuum Removal & Inflection Kinematics ($16$ variables):** Extracts the maximum band depth ($D_{\max}$) and normalized integrated absorption area ($A_{\text{abs}}$) below the linear continuum hull across major biochemical absorption valleys, combined with first- and second-derivative inflection kinematics ($\lambda_{\text{RE}}$ centroid, maximum slope, and concavity amplitude) and inter-plateau radiative contrasts.
2. **Module 2 — Local Multi-Order Sub-Window Eigenmodes ($24$ variables):** Partitions the spectrum into contiguous radiative sub-windows and extracts the dominant orthogonal singular mode separately for **Order 0** (normalized amplitude), **Order 1** ($\partial\rho/\partial\lambda$, local slope), and **Order 2** ($\partial^2\rho/\partial\lambda^2$, local concavity) within each window. This prevents high-variance plateaus (such as the near-infrared mesophyll plateau) from masking narrow diagnostic inflections in the visible, red-edge, or shortwave infrared domains.
3. **Module 3 — Global Fisher-Weighted Shrinkage Canonical Subspace ($24$ variables):** Concatenates the full differential state space $[\mathbf{X}^{(0)}, \mathbf{X}^{(1)}, \mathbf{X}^{(2)}]$, weights each channel by its inter/intra-class Fisher discriminability ratio, and extracts regularized canonical discriminant directions via analytical Ledoit-Wolf covariance shrinkage (with rank-safe singular-value-scaled orthogonal completion for binary or low-cardinality classification tasks).

---

## Installation

You can install `MSPA` directly from GitHub via `pip`:

```bash
pip install git+[https://github.com/Lucas-Osco/mspa/.git](https://github.com/Lucas-Osco/mspa/.git)
```

Alternatively, copy the standalone [`mspa.py`](mspa.py) file directly into your working directory (requires only `numpy`, `scipy`, `scikit-learn`, and `pandas`).

---

## Quick Start

```python
from mspa import MSPA, generate_synthetic_vegetation_spectra
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# 1. Generate synthetic foliar reflectance spectra (N=450, D=1580 bands, C=3 classes)
X_df, y, wavelengths = generate_synthetic_vegetation_spectra(n_samples=450, random_state=42)

# 2. Standalone usage (automatically parses wavelengths from DataFrame columns)
reducer = MSPA(n_components=64)  # or n_components="auto" for Kneedle breaking-point
X_mspa = reducer.fit_transform(X_df, y)
print("Projected shape:", X_mspa.shape)  # (450, 64)

# 3. Seamless integration inside a scikit-learn Pipeline (zero data leakage)
pipe = Pipeline([
    ("mspa", MSPA(n_components=64)),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
])
cv_f1 = cross_val_score(pipe, X_df, y, cv=5, scoring="f1_macro")
print(f"5-Fold Cross-Validated Macro-F1: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")

# 4. Pandas DataFrame output with semantic feature names
df_features = MSPA(n_components=64).set_output(transform="pandas").fit_transform(X_df, y)
print(df_features.columns[:5].tolist())
```

---

## Adaptability to Arbitrary Spectral Domains

While `MSPA` automatically activates eight biophysical leaf optical windows when input wavelengths fall within the $380\text{--}2500\text{ nm}$ VNIR–SWIR domain, the operator works seamlessly across any contiguous spectral range:
* **VNIR-only sensors ($400\text{--}1000\text{ nm}$, e.g., UAV hyperspectral cameras):** Automatically activates only the visible, red-edge, and near-infrared windows covered by the sensor.
* **Mid-Infrared / FTIR ($2500\text{--}25000\text{ nm}$) or Raman spectroscopy:** Automatically partitions contiguous segments into uniform sub-windows (`n_fallback_windows=8`) and applies the same continuum-removal, multi-order derivative, and Fisher-weighted canonical projections.
* **Automatic Excision Gap Detection:** Automatically detects atmospheric water-vapor excision gaps in `wavelengths` so piecewise Savitzky-Golay derivatives never cross non-contiguous band jumps.

---

## Citation

If you use `MSPA` in your research, please cite:

```bibtex
@article{Osco2026MSPA,
  author  = {Osco, L.P., et al.},
  title   = {Morpho-Spectral Projection Analysis ({MSPA}) for Plant Reflectance Spectroscopy},
  journal = {Submitted Manuscript},
  year    = {2026}
}
```
