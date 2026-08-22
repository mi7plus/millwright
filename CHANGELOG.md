# Changelog

All notable changes to Millwright are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims at
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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

### Changed

- Exact-version pins on every engine crate; `Cargo.lock` committed; a
  feature-matrix CI (fmt, clippy `-D warnings`, docs, MSRV, matrix, OS, wheel).
- MSRV is **1.91** (driven by polars, feature `eda`); the default install needs
  1.85.
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
