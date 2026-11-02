# MSPA: Morpho-Spectral Projection Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lucas-Osco/mspa/blob/main/MSPA_Colab_Tutorial.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn compatible](https://img.shields.io/badge/scikit--learn-compatible-orange.svg)](https://scikit-learn.org/)

**Morpho-Spectral Projection Analysis (`MSPA`)** is a supervised dimensionality reduction operator designed for contiguous one-dimensional reflectance, transmittance, and absorbance spectra ($\mathcal{T}: \mathbb{R}^D \to \mathbb{R}^K$).

Conventional dimensionality reduction techniques, such as Principal Component Analysis (`PCA`), Partial Least Squares (`PLS-DA`), `UMAP`, and Self-Organizing Maps (`SOM`), are **column-permutation invariant**: they treat adjacent wavelengths as unordered tabular columns, discarding the sequential differential geometry of the spectral curve. `MSPA` preserves physical wavelength contiguity while adhering strictly to the `scikit-learn` transformer API (`fit`, `transform`, `fit_transform`, `Pipeline`, and `set_output`).

---

## Interactive Google Colab Demo

You can test `MSPA` directly in your browser without installing anything locally using our interactive notebook with synthetic foliar reflectance spectra:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lucas-Osco/mspa/blob/main/MSPA_Colab_Tutorial.ipynb)

---

## Mathematical Structure of the Operator

Given an input spectral matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$ ordered along wavelengths $\lambda_1 < \lambda_2 < \dots < \lambda_D$, `MSPA` projects the spectrum into $K$ morpho-spectral variables through three complementary blocks:

1. **Module 1 — Beer-Lambert Continuum Removal & Inflection Kinematics (`MSPA_BIO_*`):** Extracts the maximum band depth ($D_{\max}$) and normalized integrated absorption area ($A_{\text{abs}}$) below the linear continuum hull across major biochemical absorption valleys, combined with first- and second-derivative inflection kinematics ($\lambda_{\text{RE}}$ centroid, maximum slope, and concavity amplitude) and inter-plateau radiative contrasts.
2. **Module 2 — Local Multi-Order Sub-Window Eigenmodes (`MSPA_WIN_*`):** Partitions the spectrum into contiguous radiative sub-windows and extracts the dominant orthogonal singular mode via direct Singular Value Decomposition (SVD) separately for **Order 0** (normalized amplitude), **Order 1** ($\partial\rho/\partial\lambda$, local slope), and **Order 2** ($\partial^2\rho/\partial\lambda^2$, local concavity) within each window.
3. **Module 3 — Global Fisher-Weighted Shrinkage Canonical Subspace (`MSPA_CAN_*`):** Concatenates the full differential state space $[\mathbf{X}^{(0)}, \mathbf{X}^{(1)}, \mathbf{X}^{(2)}]$, weights each channel by its inter/intra-class Fisher discriminability ratio, and extracts regularized canonical discriminant directions via analytical Ledoit-Wolf covariance shrinkage (with rank-safe singular-value-scaled orthogonal completion when $C - 1 < K_{\text{can}}$).

---

## Installation

Install directly from GitHub via `pip`:

```bash
pip install git+https://github.com/Lucas-Osco/mspa.git
```

Alternatively, copy the standalone [`mspa.py`](mspa.py) file into your project directory (requires only `numpy`, `scipy`, `scikit-learn`, and `pandas`).

---

## Comprehensive Usage Guide (Quick Start)

### 0. Generating or Loading Spectral Data

For testing or tutorials without proprietary datasets, `mspa` includes a physically plausible synthetic foliar reflectance generator (`generate_synthetic_vegetation_spectra`) covering $D = 1,580$ bands ($401\text{--}2400\text{ nm}$, with water-vapor windows excised at $1331\text{--}1490\text{ nm}$ and $1771\text{--}2030\text{ nm}$):

```python
from mspa import MSPA, generate_synthetic_vegetation_spectra
from sklearn.model_selection import train_test_split

# Generate 450 synthetic leaf spectra across 3 physiological classes
X_df, y, wavelengths = generate_synthetic_vegetation_spectra(n_samples=450, random_state=42)

# Split into training (80%) and holdout test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.20, stratify=y, random_state=42
)
```

---

### 1. Basic Train/Test Projection

Like `PLSRegression` or `LinearDiscriminantAnalysis`, `MSPA` is a **supervised** projection operator and requires the class labels `y_train` during `fit(X_train, y_train)`. During `transform(X_test)`, only the spectra are passed, preventing any data leakage:

```python
# Option A: Default benchmark configuration (K = 64 morpho-spectral variables)
reducer_64 = MSPA(n_components=64)
X_train_64 = reducer_64.fit_transform(X_train, y_train)
X_test_64 = reducer_64.transform(X_test)
print("Default (K=64) shapes:", X_train_64.shape, X_test_64.shape)  # (360, 64), (90, 64)

# Option B: Custom user-defined dimensionality (e.g., K = 20 or K = 96)
# If K is smaller than the full three-module basis, MSPA automatically ranks and
# retains the top K variables by their training Fisher discriminability ratio.
reducer_20 = MSPA(n_components=20)
X_train_20 = reducer_20.fit_transform(X_train, y_train)
X_test_20 = reducer_20.transform(X_test)
print("Custom (K=20) shapes :", X_train_20.shape, X_test_20.shape)  # (360, 20), (90, 20)
```

---

### 2. Automatic Intrinsic Dimensionality via Kneedle Breaking-Point (`n_components="auto"`)

If you do not want to fix the number of components manually, set `n_components="auto"`. `MSPA` will apply the maximum-curvature Kneedle algorithm to the cumulative Fisher-weighted multi-order singular value spectrum of `X_train` to determine the optimal number of canonical variables ($K^*$):

```python
reducer_auto = MSPA(n_components="auto")
X_train_auto = reducer_auto.fit_transform(X_train, y_train)
X_test_auto = reducer_auto.transform(X_test)

print("Automatically estimated K* :", reducer_auto.n_components_)  # e.g., 52
print("Projected test matrix shape:", X_test_auto.shape)            # (90, 52)
```

---

### 3. Input Formats: Pandas DataFrames vs. NumPy Arrays

`MSPA` accepts both `pandas.DataFrame` and `numpy.ndarray` inputs:

- **With a `pandas.DataFrame`:** If `wavelengths=None`, `MSPA` automatically parses numerical wavelengths from the column headers (e.g., `"WL_401"`, `"401"`, or `"401.0 nm"`).
- **With a `numpy.ndarray`:** Pass the 1D `wavelengths` array explicitly so `MSPA` can locate the optical sub-windows and detect atmospheric excision jumps.

```python
# Using raw NumPy arrays with an explicit wavelength vector
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()

reducer_np = MSPA(n_components=64, wavelengths=wavelengths)
Z_train_np = reducer_np.fit_transform(X_train_np, y_train)
Z_test_np = reducer_np.transform(X_test_np)
```

---

### 4. Exporting Semantic Feature Names as a Pandas DataFrame

Because `MSPA` implements the modern `scikit-learn` `get_feature_names_out()` interface, you can call `.set_output(transform="pandas")` to receive a labeled `DataFrame` indicating whether each column belongs to Module 3 (`MSPA_CAN_*`), Module 2 (`MSPA_WIN_*`), or Module 1 (`MSPA_BIO_*`):

```python
reducer_df = MSPA(n_components=64).set_output(transform="pandas")
df_train_mspa = reducer_df.fit_transform(X_train, y_train)

print(type(df_train_mspa))
print("Module 3 (Canonical) sample  :", df_train_mspa.columns[:3].tolist())
print("Module 2 (Windows) sample    :", df_train_mspa.columns[24:27].tolist())
print("Module 1 (Biophysical) sample:", df_train_mspa.columns[-3:].tolist())
```

---

### 5. Integration with `scikit-learn` Pipelines, Classical ML, and Tabular Foundation Models

`MSPA` can be placed directly inside a `scikit-learn` `Pipeline` for leak-free $k$-fold cross-validation with classical classifiers (`RandomForest`, `XGBoost`, `SVM`, `LDA`) or Tabular Foundation Models (`TabPFN`, `TabICL`):

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Example A: Pipeline with SVM (RBF Kernel) or Random Forest
pipe_svm = Pipeline([
    ("mspa", MSPA(n_components=64)),
    ("svm", SVC(C=1.0, kernel="rbf", probability=True, random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe_svm, X_df, y, cv=cv, scoring="f1_macro")
print(f"5-Fold CV Macro-F1 (MSPA + SVM-RBF): {scores.mean():.4f} +/- {scores.std():.4f}")

# Example B: Coupling MSPA with a Tabular Foundation Model (e.g., TabPFN / TabICL)
# Compressing 1,580 bands into 64 MSPA variables accelerates TFM inference by 11x-126x
# while preserving fine differential absorption morphology:
#
# from tabpfn import TabPFNClassifier
# pipe_tfm = Pipeline([
#     ("mspa", MSPA(n_components=64)),
#     ("tabpfn", TabPFNClassifier(device="cuda"))
# ])
# pipe_tfm.fit(X_train, y_train)
# y_prob = pipe_tfm.predict_proba(X_test)
```

---

### 6. Using `MSPA` on Other Spectral Ranges (VNIR-Only UAV Cameras or MIR/FTIR)

While `MSPA` activates eight biophysical leaf optical windows when input wavelengths fall within $380\text{--}2500\text{ nm}$, it adapts automatically to any contiguous spectral domain:

```python
import numpy as np

# Case 1: VNIR-only sensor (e.g., 400-1000 nm UAV hyperspectral camera with 300 bands)
# Automatically activates only the 6 optical windows covered by 400-1000 nm (ignoring SWIR)
wl_vnir = np.linspace(400, 1000, 300)
X_vnir = np.random.normal(size=(200, 300))
y_vnir = np.random.randint(0, 3, size=200)

mspa_vnir = MSPA(n_components="auto", wavelengths=wl_vnir).fit(X_vnir, y_vnir)
print("VNIR active windows:", [w[0] for w in mspa_vnir.active_windows_])

# Case 2: Mid-Infrared / FTIR (2500-10000 nm) or arbitrary wavenumbers (cm^-1)
# Automatically partitions contiguous segments into `n_fallback_windows` uniform sub-windows
wl_mir = np.linspace(2500, 10000, 500)
X_mir = np.random.normal(size=(200, 500))

mspa_mir = MSPA(n_components=64, wavelengths=wl_mir, n_fallback_windows=8)
Z_mir = mspa_mir.fit_transform(X_mir, y_vnir)
print("MIR projected shape:", Z_mir.shape)  # (200, 64)
```

---

## API Reference: Parameters and Fitted Attributes

| Parameter / Attribute | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_components` | `int` or `'auto'` | `64` | Number of output morpho-spectral variables, or `'auto'` for Kneedle breaking-point estimation. |
| `wavelengths` | `array-like` or `None` | `None` | Wavelengths corresponding to columns of `X`. Auto-parsed if `X` is a `DataFrame` with numeric headers. |
| `sg_window` | `int` | `11` | Window length for piecewise Savitzky-Golay 1st and 2nd derivative calculation. |
| `sg_poly` | `int` | `2` | Polynomial order for Savitzky-Golay derivative filtering. |
| `shrinkage` | `float` | `0.35` | Analytical Ledoit-Wolf covariance shrinkage coefficient $\gamma \in (0, 1)$ in Module 3. |
| `n_fallback_windows` | `int` | `8` | Number of uniform contiguous sub-windows when operating outside the $380\text{--}2500\text{ nm}$ range. |
| `random_state` | `int` | `42` | Random seed for deterministic orientation. |
| `n_components_` | `int` *(fitted)* | — | Effective number of retained morpho-spectral variables after fitting. |
| `feature_names_out_` | `ndarray` *(fitted)* | — | Array of semantic variable names (`MSPA_CAN_*`, `MSPA_WIN_*`, `MSPA_BIO_*`). |
| `contiguous_segments_` | `list` *(fitted)* | — | List of contiguous band index arrays separated by excised water-vapor gaps. |

---

## Citation

If you use `MSPA` in your research, please cite:

```bibtex
@article{Osco2026MSPA,
  author  = {Osco, Lucas Prado and others},
  title   = {Morpho-Spectral Projection Analysis ({MSPA}) for Plant Reflectance Spectroscopy},
  journal = {Submitted Manuscript},
  year    = {2026}
}
```
