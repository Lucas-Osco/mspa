# MSPA: Morpho-Spectral Projection Analysis for Plant Spectroscopy

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/github/Lucas-Osco/mspa/blob/main/MSPA_Colab_Tutorial.ipynb](https://colab.research.google.com/github/Lucas-Osco/mspa/blob/main/MSPA_Colab_Tutorial.ipynb))
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn compatible](https://img.shields.io/badge/scikit--learn-compatible-orange.svg)](https://scikit-learn.org/)

**Morpho-Spectral Projection Analysis (`MSPA`)** is a supervised dimensionality reduction operator formulated specifically for **plant reflectance spectroscopy** across the **Visible, Near-Infrared, and Shortwave Infrared (VNIR–SWIR, 350–2500 nm)** domain ($\mathcal{T}_{\text{MSPA}}: \mathbb{R}^D \to \mathbb{R}^K$).

Conventional dimensionality reduction methods routinely adopted in vegetation spectroscopy—such as Principal Component Analysis (`PCA`), Partial Least Squares Discriminant Analysis (`PLS-DA`), `UMAP`, and Self-Organizing Maps (`SOM`)—are **column-permutation invariant**: they treat adjacent wavelengths as unordered tabular columns, discarding the sequential differential geometry of biochemical absorption features. `MSPA` preserves physical wavelength contiguity while adhering strictly to the `scikit-learn` transformer API (`fit`, `transform`, `fit_transform`, `Pipeline`, and `set_output`).

<p align="center">
  <img src="mspa_architecture_overview.png" alt="MSPA Architectural Overview" width="96%">
</p>

---

## Interactive Google Colab Demo

Test `MSPA` directly in your browser without installing anything locally using our interactive notebook with continuous foliar reflectance spectra ($\rho(\lambda) \in [0, 1]$, $400\text{--}2500\text{ nm}$):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lucas-Osco/mspa/blob/main/MSPA_Colab_Tutorial.ipynb) **[Launch `MSPA_Colab_Tutorial.ipynb` in Google Colab](https://colab.research.google.com/github/Lucas-Osco/mspa/blob/main/MSPA_Colab_Tutorial.ipynb)**

---

## Why Standard Reducers Fail on Plant Spectra: The Column-Permutation Problem

Let $`\mathbf{X} \in \mathbb{R}^{N \times D}`$ be a foliar reflectance matrix ordered along physical wavelengths $`\lambda_1 < \lambda_2 < \dots < \lambda_D`$, and let $`\mathbf{\Pi} \in \{0, 1\}^{D \times D}`$ be an arbitrary permutation matrix that randomly shuffles the order of the wavelengths ($`\mathbf{X}_{\mathbf{\Pi}} = \mathbf{X}\mathbf{\Pi}`$).

In standard `PCA`, the sample covariance matrix of the shuffled spectrum is $`\mathbf{\Sigma}_{\mathbf{\Pi}} = \mathbf{\Pi}^\top \mathbf{\Sigma} \mathbf{\Pi}`$, which simply permutes the rows of the eigenvectors ($`\mathbf{V}_{\mathbf{\Pi}} = \mathbf{\Pi}^\top \mathbf{V}`$). Consequently, the projected scores remain **numerically identical**:

$$\mathbf{Z}_{\text{PCA}}(\mathbf{X}\mathbf{\Pi}) = (\mathbf{X}\mathbf{\Pi})(\mathbf{\Pi}^\top \mathbf{V}_K) = \mathbf{X}(\mathbf{\Pi}\mathbf{\Pi}^\top)\mathbf{V}_K = \mathbf{Z}_{\text{PCA}}(\mathbf{X})$$

The exact same invariance holds for `PLS-DA` (via inner products $`\mathbf{X}^\top \mathbf{Y}`$), `UMAP`, and `SOM` (via pairwise Euclidean distances $`\Vert{}\mathbf{x}_i \mathbf{\Pi} - \mathbf{x}_l \mathbf{\Pi}\Vert{}_2 = \Vert{}\mathbf{x}_i - \mathbf{x}_l\Vert{}_2`$). Because these methods are blind to wavelength adjacency, high-variance non-diagnostic plateaus (such as leaf albedo across the Near-Infrared plateau) dominate the projection and mask narrow diagnostic absorption features in the visible and SWIR regions.

You can verify this directly in Python:

```python
import numpy as np
from sklearn.decomposition import PCA
from mspa import MSPA, generate_synthetic_vegetation_spectra

X_df, y, wavelengths = generate_synthetic_vegetation_spectra(n_samples=450, random_state=42)
X_orig = X_df.to_numpy()
X_shuffled = X_orig[:, np.random.default_rng(777).permutation(X_orig.shape[1])]

# PCA is completely blind to wavelength order (Difference = 0.00e+00)
z_pca_1 = PCA(n_components=10, random_state=42).fit_transform(X_orig)
z_pca_2 = PCA(n_components=10, random_state=42).fit_transform(X_shuffled)
print("Max PCA difference after shuffling wavelengths :", np.max(np.abs(z_pca_1 - z_pca_2)))

# MSPA explicitly depends on contiguous curve topology (Difference = ~0.85)
z_mspa_1 = MSPA(n_components=64, wavelengths=wavelengths).fit_transform(X_orig, y)
z_mspa_2 = MSPA(n_components=64, wavelengths=wavelengths).fit_transform(X_shuffled, y)
print("Mean MSPA difference after shuffling wavelengths:", np.mean(np.abs(z_mspa_1 - z_mspa_2)))
```

---

## Mathematical Formulation: Inside the Three Modules of `MSPA`

Given Savitzky–Golay smoothed reflectance spectra $`\mathbf{X} \in \mathbb{R}^{N \times D}`$ ($`\rho(\lambda) \in [0, 1]`$), `MSPA` operates seamlessly on both continuous laboratory spectra (e.g., contact-probe leaf-clip measurements across $`400\text{--}2500\text{ nm}`$) and passive solar field spectra where deep atmospheric water-vapor windows are excised ($`\Delta\lambda_j > \max(5\delta_{\text{med}}, 10\text{ nm})`$). Within each contiguous spectral segment $`\mathcal{S}_b`$, analytical local least-squares Savitzky–Golay convolution ($`W_{\text{SG}}=11, p_{\text{SG}}=2`$) estimates the **1st derivative** ($`\mathbf{X}^{(1)} \approx \partial \mathbf{X} / \partial \lambda`$, local slope) and **2nd derivative** ($`\mathbf{X}^{(2)} \approx \partial^2 \mathbf{X} / \partial \lambda^2`$, local concavity).

The input spectrum $`\mathbf{x}_i \in \mathbb{R}^D`$ is then projected into $`\mathbf{z}_i = [\mathbf{z}_{i,\text{bio}}^\top, \mathbf{z}_{i,\text{win}}^\top, \mathbf{z}_{i,\text{can}}^\top]^\top \in \mathbb{R}^{K^*}`$ across three complementary modules:

### 1. Module 1 — Additive Continuum-Removed Geometry & Inflection Kinematics (`MSPA_BIO_*`, Variables 1–16)

In plant reflectance spectroscopy, biochemical absorption features manifest as local concave valleys bounded by higher-reflectance scattering shoulders. Across five major foliar absorption valleys established in leaf radiative transfer models ($\mathcal{V}_1$: Chlorophyll well $560\text{--}740\text{ nm}$; $\mathcal{V}_2, \mathcal{V}_3$: NIR water and mesophyll $910\text{--}1060\text{ nm}$ and $1130\text{--}1280\text{ nm}$; $\mathcal{V}_4$: SWIR 1 lignin–cellulose–water $1520\text{--}1760\text{ nm}$; and $\mathcal{V}_5$: SWIR 2 protein–cellulose $2050\text{--}2360\text{ nm}$), `MSPA` constructs the linear continuum chord $`c_{i,v}(\lambda_j)`$ connecting the valley shoulders ($`t_j \in [0, 1]`$):

$$c_{i,v}(\lambda_j) = x_i(\lambda_{v,\text{start}})(1 - t_j) + x_i(\lambda_{v,\text{end}})t_j, \qquad d_{i,v}(\lambda_j) = c_{i,v}(\lambda_j) - x_i(\lambda_j)$$

The additive residual profile $`d_{i,v}(\lambda_j)`$ removes local baseline tilts within each valley without introducing inter-regional variance coupling or division-by-zero instabilities. From each valley $`v`$, Module 1 extracts the **maximum contrast depth** ($`D_{\max, i, v}`$) and **normalized integrated depression area** ($`A_{\text{abs}, i, v}`$), alongside four **Red-Edge inflection kinematics** ($`680\text{--}760\text{ nm}`$, including the continuous inflection wavelength centroid $`\lambda_{\text{RE}, i}`$) and two inter-plateau radiative contrasts:

$$D_{\max, i, v} = \max_{\lambda_j \in \mathcal{V}_v} d_{i,v}(\lambda_j), \qquad A_{\text{abs}, i, v} = \frac{1}{L_v} \sum_{\lambda_j \in \mathcal{V}_v} d_{i,v}(\lambda_j), \qquad \lambda_{\text{RE}, i} = \frac{\sum_{\lambda_j \in \mathcal{W}_{\text{RE}}} \max(x_{i,j}^{(1)}, 0) \lambda_j}{\sum_{\lambda_j \in \mathcal{W}_{\text{RE}}} \max(x_{i,j}^{(1)}, 0) + \epsilon}$$

Because $`\max(\cdot)`$ and horizontal wavelength shifts ($`\lambda_{\text{RE}}`$) are **nonlinear functionals**, Module 1 captures absorption band shifts (e.g., chlorotic blue-shifts and red-shifts) that cannot be represented by fixed linear matrix projections.

### 2. Module 2 — Local Multi-Order Sub-Window Eigenmodes (`MSPA_WIN_*`, Variables 17–40)

To prevent high-variance NIR scattering plateaus from masking subtle variations in narrower regions, Module 2 partitions the $350\text{--}2500\text{ nm}$ vegetation spectrum into $`W = 8`$ contiguous radiative sub-windows $`\mathcal{W}_1, \dots, \mathcal{W}_8`$ (Blue/Carotenoids $400\text{--}510\text{ nm}$, Green/Anthocyanins $511\text{--}620\text{ nm}$, Red Chlorophyll Well $621\text{--}685\text{ nm}$, Red-Edge Ramp $686\text{--}765\text{ nm}$, NIR Mesophyll Plateau $766\text{--}910\text{ nm}$, NIR Water $911\text{--}1350\text{ nm}$, SWIR 1 $1450\text{--}1790\text{ nm}$, and SWIR 2 $1980\text{--}2450\text{ nm}$). Within each window $`w`$ and for each differential order $`m \in \{0, 1, 2\}`$ independently, the standardized sub-matrix $`\tilde{\mathbf{X}}_{\text{train}, w}^{(m)}`$ is decomposed via Singular Value Decomposition (SVD) to project each sample onto the dominant rank-1 right singular vector $`\mathbf{v}_{w,1}^{(m)}`$:

$$z_{i,\text{win}}^{(w, m)} = \left(\tilde{\mathbf{x}}_{i, w}^{(m)}\right)^\top \mathbf{v}_{w, 1}^{(m)}, \qquad w \in \{1, \dots, 8\}, \quad m \in \{0, 1, 2\}$$

This guarantees that every spectral region contributes its primary mode of **amplitude** ($`m=0`$), **slope** ($`m=1`$), and **concavity** ($`m=2`$) ($`8 \times 3 = 24`$ variables), forming a strictly block-diagonal sparse loading structure with zero cross-window variance leakage.

### 3. Module 3 — Global Fisher-Weighted Shrinkage Canonical Subspace (`MSPA_CAN_*`, Variables 41–64)

To capture global cross-wavelength interactions where signatures diverge across classes $`c \in \{1, \dots, C\}`$, Module 3 concatenates the full state space $`\mathbf{M}_i = [\mathbf{x}_i^{(0)\top}, \mathbf{x}_i^{(1)\top}, \mathbf{x}_i^{(2)\top}]^\top \in \mathbb{R}^{3D}`$ and weights each channel $`q \in \{1, \dots, 3D\}`$ by its inter-class to intra-class Fisher discriminability ratio $`\omega_q`$:

$$\tilde{\omega}_q = \sqrt{\frac{\text{Std}_{c}(\mu_{c,q})}{\frac{1}{C}\sum_{c=1}^{C} \sigma_{c,q} + \delta}}, \qquad \mathbf{H} = \tilde{\mathbf{M}} \, \text{diag}(\omega_1, \dots, \omega_{3D})$$

After orthogonal SVD compression of $`\mathbf{H}`$ into $`\mathbf{U}`$, canonical discriminant directions $`\mathbf{w}`$ are extracted via analytical Ledoit–Wolf covariance shrinkage ($`\gamma = 0.35`$):

$$\mathbf{\Sigma}_W^{(\gamma)} = (1 - \gamma)\mathbf{S}_W + \gamma \frac{\text{Tr}(\mathbf{S}_W)}{P}\mathbf{I}_P, \qquad \mathbf{\Sigma}_B \mathbf{w} = \Lambda \mathbf{\Sigma}_W^{(\gamma)} \mathbf{w}$$

When $`C - 1 < K_{\text{can}}`$ (e.g., binary classification $`C=2`$ where $`\text{rank}(\mathbf{\Sigma}_B) = 1`$), `MSPA` extracts regularized discriminant axes separately for Orders $`0, 1, 2`$, and $`\mathbf{H}`$, and completes the remaining axes using singular-value-scaled orthogonal vectors of $`\mathbf{H}`$ ($`\sqrt{\sigma_k^2 / \sigma_1^2}`$), preventing rank-1 noise overfitting.

---

## Installation

Install directly from GitHub via `pip`:

```bash
pip install git+https://github.com/Lucas-Osco/mspa.git
```

Alternatively, copy the standalone [`mspa.py`](mspa.py) file into your project directory (requires only `numpy`, `scipy`, `scikit-learn`, and `pandas`).

---

## Comprehensive Usage Guide (Quick Start)

### 0. Generating or Loading Plant Reflectance Spectra

For testing or tutorials without proprietary datasets, `mspa` includes a physically realistic synthetic foliar reflectance generator (`generate_synthetic_vegetation_spectra`). By default, it generates continuous leaf-clip laboratory spectra across $400\text{--}2500\text{ nm}$ ($D = 2,101$ bands, $\rho(\lambda) \in [0, 1]$, Savitzky–Golay smoothed); setting `excise_water_vapor=True` simulates passive solar field acquisitions with excised atmospheric water-vapor windows ($D = 1,580$ bands):

```python
from mspa import MSPA, generate_synthetic_vegetation_spectra
from sklearn.model_selection import train_test_split

# Generate 450 continuous leaf-clip reflectance spectra (400-2500 nm, D = 2,101 bands)
X_df, y, wavelengths = generate_synthetic_vegetation_spectra(
    n_samples=450, excise_water_vapor=False, random_state=42
)

# Split into training (80%) and holdout test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.20, stratify=y, random_state=42
)
```

---

### 1. Basic Train/Test Projection

Like `PLSRegression` or `LinearDiscriminantAnalysis`, `MSPA` is a **supervised** projection operator and requires the class labels `y_train` during `fit(X_train, y_train)`. During `transform(X_test)`, only the spectra are passed, preventing any data leakage:

```python
# Option A: Default configuration (K = 64 morpho-spectral variables)
reducer_64 = MSPA(n_components=64)
X_train_64 = reducer_64.fit_transform(X_train, y_train)
X_test_64 = reducer_64.transform(X_test)
print("Default (K=64) shapes:", X_train_64.shape, X_test_64.shape)  # (360, 64), (90, 64)

# Option B: Custom user-defined dimensionality (e.g., K = 20 or K = 96)
reducer_20 = MSPA(n_components=20)
X_train_20 = reducer_20.fit_transform(X_train, y_train)
X_test_20 = reducer_20.transform(X_test)
print("Custom (K=20) shapes :", X_train_20.shape, X_test_20.shape)  # (360, 20), (90, 20)
```

---

### 2. Autonomous Intrinsic Dimensionality via Kneedle Breaking-Point (`n_components="auto"`)

If you do not want to fix the number of components manually, set `n_components="auto"`. `MSPA` applies the maximum-curvature Kneedle algorithm to the cumulative Fisher-weighted multi-order singular value spectrum of `X_train` to determine the optimal number of canonical variables ($K^*$):

```python
reducer_auto = MSPA(n_components="auto")
X_train_auto = reducer_auto.fit_transform(X_train, y_train)
X_test_auto = reducer_auto.transform(X_test)

print("Automatically estimated K* :", reducer_auto.n_components_)  # e.g., 52
print("Projected test matrix shape:", X_test_auto.shape)            # (90, 52)
```

---

### 3. Exporting Semantic Feature Names, Variable Catalog, and Multi-Order Loadings

Because `MSPA` implements the `scikit-learn` `get_feature_names_out()` interface, you can call `.set_output(transform="pandas")` to receive a labeled `DataFrame` ordered across Module 1 (`MSPA_BIO_*`, vars 1–16), Module 2 (`MSPA_WIN_*`, vars 17–40), and Module 3 (`MSPA_CAN_*`, vars 41–64), as well as inspect the variable catalog and structure correlation loadings ($\mathbf{\Lambda}_{\text{MSPA}}$):

```python
reducer_df = MSPA(n_components=64).set_output(transform="pandas")
df_train_mspa = reducer_df.fit_transform(X_train, y_train)

print("Module 1 (Biophysical, 1-16) :", df_train_mspa.columns[:3].tolist())
print("Module 2 (Window SVD, 17-40) :", df_train_mspa.columns[16:19].tolist())
print("Module 3 (Canonical, 41-64)  :", df_train_mspa.columns[40:43].tolist())

# Inspect the full metadata catalog of all 64 extracted variables
catalog_df = reducer_df.get_variable_catalog()
print(catalog_df.head())

# Compute Multi-Order Structure Loadings (Order 0, Order 1, Order 2, and Composite)
loadings = reducer_df.compute_structure_loadings(X_train)
print("Composite loading matrix shape (D x K*):", loadings["Composite"].shape)
```

---

### 4. Integration with `scikit-learn` Pipelines, Classical ML, and Tabular Foundation Models

`MSPA` can be placed directly inside a `scikit-learn` `Pipeline` for leak-free $k$-fold cross-validation with classical classifiers (`RandomForest`, `XGBoost`, `SVM`, `LDA`) or Tabular Foundation Models (`TabPFN`, `TabICL`):

```python
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Example A: Pipeline with SVM (RBF Kernel)
pipe_svm = Pipeline([
    ("mspa", MSPA(n_components=64)),
    ("svm", SVC(C=1.0, kernel="rbf", probability=True, random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe_svm, X_df, y, cv=cv, scoring="f1_macro")
print(f"5-Fold CV Macro-F1 (MSPA + SVM-RBF): {scores.mean():.4f} +/- {scores.std():.4f}")

# Example B: Coupling MSPA with a Tabular Foundation Model (e.g., TabPFN / TabICL)
# Compressing contiguous spectra into 64 MSPA variables accelerates TFM inference
# by up to 126.5x while preserving fine differential absorption morphology:
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

### 5. Adapting Across Vegetation Sensors (Full-Range VNIR–SWIR vs. VNIR-Only UAV Cameras)

While `MSPA` activates all eight biophysical leaf optical sub-windows ($\mathcal{W}_1\text{--}\mathcal{W}_8$) and five absorption valleys ($\mathcal{V}_1\text{--}\mathcal{V}_5$) when input wavelengths span the full $350\text{--}2500\text{ nm}$ VNIR–SWIR range, it automatically adapts to sensors covering partial vegetation windows, such as UAV hyperspectral cameras or close-range phenotyping sensors restricted to the **VNIR ($400\text{--}1000\text{ nm}$)** domain:

```python
import numpy as np

# VNIR-only sensor (e.g., 400-1000 nm UAV camera with 300 bands)
# Automatically activates the 6 optical windows covered by 400-1000 nm (ignoring SWIR)
wl_vnir = np.linspace(400, 1000, 300)
X_vnir = np.random.uniform(0.05, 0.55, size=(200, 300))
y_vnir = np.random.randint(0, 3, size=200)

mspa_vnir = MSPA(n_components="auto", wavelengths=wl_vnir).fit(X_vnir, y_vnir)
print("VNIR active windows:", [w[0] for w in mspa_vnir.active_windows_])
print("VNIR active valleys:", [v[0] for v in mspa_vnir.active_valleys_])
```

---

## API Reference: Parameters, Methods, and Fitted Attributes

| Parameter / Attribute / Method | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_components` | `int` or `'auto'` | `64` | Number of output morpho-spectral variables, or `'auto'` for Kneedle breaking-point estimation. |
| `wavelengths` | `array-like` or `None` | `None` | Wavelengths (in nm, within $350\text{--}2500\text{ nm}$) corresponding to columns of `X`. Auto-parsed if `X` is a `DataFrame`. |
| `sg_window` | `int` | `11` | Window length for analytical local least-squares Savitzky–Golay 1st and 2nd derivative estimation. |
| `sg_poly` | `int` | `2` | Polynomial order for Savitzky–Golay derivative estimation. |
| `shrinkage` | `float` | `0.35` | Analytical Ledoit–Wolf covariance shrinkage coefficient $\gamma \in (0, 1)$ in Module 3. |
| `random_state` | `int` | `42` | Random seed for deterministic orientation. |
| `n_components_` | `int` *(fitted)* | — | Effective number of retained morpho-spectral variables after fitting. |
| `feature_names_out_` | `ndarray` *(fitted)* | — | Array of semantic variable names (`MSPA_BIO_*`, `MSPA_WIN_*`, `MSPA_CAN_*`). |
| `contiguous_segments_` | `list` *(fitted)* | — | List of contiguous band index arrays (single segment for laboratory spectra, or multiple segments if field water-vapor gaps are present). |
| `get_variable_catalog()` | `method` | — | Returns a `DataFrame` detailing the module, differential order, wavelength range, and Fisher score of each variable. |
| `compute_structure_loadings(X)` | `method` | — | Returns a `dict` with the multi-order structure loading matrices (`Order_0`, `Order_1`, `Order_2`, and `Composite`). |

---

## Citation

If you use `MSPA` in your research, please cite:

```bibtex
@article{Osco2026MSPA,
  author  = {Osco, Lucas Prado and Silva, Marcelo Santos da and Silva, Filipe Vieira and Zhao, Liang and Ramos, Ana Paula Marques and Ferreira, Ednaldo Jos{\'e} and Bendahan, Amaury Burlamaqui and Jorge, Lucio Andr{\'e} de Castro},
  title   = {Morpho-Spectral Projection Analysis ({MSPA}) for Plant Spectroscopy},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026}
}
```
