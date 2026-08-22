# Changelog

All notable changes to Millwright are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims at
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Python: a scikit-learn-shaped object API.** `mw.Frame` with `from_pandas` /
  `from_numpy` / `from_rows` ingest; composable transformer/estimator objects
  (`StandardScaler`, `MinMaxScaler`, `SimpleImputer`, `OneHotEncoder`,
  `RandomForest`, `LinearRegression`) added via `pipe.step(name, obj)` /
  `pipe.estimator(name, obj)`; `fit`/`predict`/`evaluate` accept a `Frame`.
- **Python: `pipeline.explain(...)`** returns SHAP feature importance
  (`Explainer.kernel()` configurable), and **`pipeline.export_onnx(path)`**
  writes the fitted pipeline to a single ONNX file. The `python` wheel now
  bundles the `model-selection`, `explain`, and `onnx` engines.

## [0.1.1]

### Added

- **`LogisticRegression`** — a native, core binary classifier with genuine
  `predict_proba`: the framework's first real `ProbaPredictor`.
- **`calibration` feature** — `PlattScaling`, `IsotonicRegression`,
  `reliability_curve`, and `CalibratedClassifier`, which wraps any
  `ProbaPredictor` and returns calibrated probabilities.
- **`anomaly` feature** — `Mahalanobis` and `KnnScore`, unified behind an
  `OutlierDetector` trait.
- **`eda` feature** — a polars-backed, dtype-aware `Table` (CSV/Parquet ingest
  that lowers to the numeric `Frame`) and a typed `Profile` with an HTML report,
  actionable alerts, and `suggest_pipeline()`.
- **Transformers** — `Winsorize`, `PowerTransform` (Yeo-Johnson),
  `ColumnTransformer`, and the supervised `TargetEncoder`.
- **Convenience** — `Frame::from_csv` (dependency-free numeric loader),
  `Table::head`.
- **Python** — `min_max_scaler`, `simple_imputer`, `one_hot`,
  `linear_regression`, and `evaluate()`.
- **Examples** — `explore` (ingest → profile → pipeline) and `trust`
  (calibration → reliability → anomaly detection).
- **Benchmarks** — `benches/throughput.rs` (criterion): the boundary conversion
  and core fit/predict, backing the "Rust speed" claim.
- **One-hot ingest** — `Table::to_frame_with` / `into_dataset_with` and a
  `CategoryEncoding` enum: lower nominal categories to 0/1 indicator columns
  instead of ordinal codes.
- **Schema-aware preprocessing** — `Frame` carries a per-column `Dtype`; `Table`
  marks categoricals as it lowers; scalers / `Winsorize` / `PowerTransform` pass
  categorical columns through untouched, and `OneHotEncoder` encodes by dtype
  rather than a value heuristic when the schema is known.
- **Real-data validation** — an end-to-end integration test on Quinlan's
  PlayTennis (`tests/real_data.rs`): CSV → profile → suggested pipeline → fit.

### Changed

- Exact-version pins on every engine crate; `Cargo.lock` committed; a
  feature-matrix CI (fmt, clippy `-D warnings`, docs, matrix, OS, examples,
  benches, publish dry-run, wheel).
- `selection.rs` split into `selection/{scoring,cv,search}`.
- MSRV is **1.95** (dep-dictated — `sysinfo` via tract, and polars); enforced by
  cargo via `rust-version` rather than a dedicated CI job (which would break on
  every transitive bump). The default install needs 1.85.
- De-staled the crate and module docs (no more "Phase 0 · the spine").

### Fixed

- Golden tests and the crate doctest build under every feature subset (they were
  unconditionally referencing backend-gated types).
- Float sorts use `f64::total_cmp`, closing a NaN-driven panic class.

## [0.1.0]

- Phases 0–8: the `Frame`/trait spine and smartcore backend, preprocessing and
  model selection, a second backend (linfa) and HPO, evaluation/diagnostics/
  explainability, ONNX export and inference, serving + drift monitoring + a model
  registry, time-series and out-of-core estimators, AutoML, and 1.0 hardening.
