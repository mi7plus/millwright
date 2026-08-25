# Changelog

All notable changes to Millwright are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims at
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [2.2.3] - 2026-08-25

### Added
- First-class Python bindings and type stubs for voting, bagging, boosting,
  stacking, and AutoML, including ONNX export and ensemble diagnostics.
- Explicit classification/regression semantics for ensembles, avoiding
  accidental voting over integer-valued regression targets.
- Python logistic regression and probability prediction for soft voting,
  minute budgets, structured AutoML leaderboards, failure diagnostics, and
  fitted-winner access.
- Direct probability prediction and capability inspection on `AutoMLResult`,
  plus elapsed-time, trial-count, and budget-exhaustion diagnostics.

### Changed
- Split profiling, alert generation, classification metrics, and ONNX ensemble
  composition into focused helpers without changing their public behavior.

### Fixed
- Preserve domain-specific opset imports while composing ONNX ensemble graphs;
  CI now checks tree ensembles with the official ONNX checker.
- AutoML stacking inherits the configured CV strategy, and failed ensemble
  candidates are retained as inspectable diagnostics instead of disappearing.
- AutoML continues past failed individual candidates, validates budgets, and
  reports those failures; classifier candidate construction now shares one
  feature-aware model matrix instead of three duplicated paths.
- Ensembles validate classification labels, learning rates, and model output
  dimensions, returning typed errors instead of rounding invalid labels or
  risking indexing panics.
- Updated Python CI to use a patched pytest release while preserving Python 3.9
  ABI support with a dependency-free wheel smoke test.
- Updated website, guide, examples, and deployment notes for the current AutoML
  and native ONNX-ML capabilities.
- Reject non-finite cross-validation scores so invalid candidates and ensembles
  cannot win an AutoML search.

## [2.2.2] - 2026-08-24

### Added
- AutoML ensemble-family selection across voting, bagging, boosting, and
  stacking, with configurable ensemble size and explicit winner introspection.
- Portable ONNX graph composition and round-trip inference for every generated
  ensemble family, including probability-producing logistic pipelines.

### Changed
- Soft voting now averages genuine model probabilities instead of fractions of
  hard class predictions.
- AutoML can prefer an ensemble on tied scores and exposes ensemble winners via
  `is_ensemble()` and `best_ensemble()`.

### Fixed
- AutoML ensemble winners can now be exported to ONNX and served through the
  same inference runtime as single pipelines.
- Added deterministic coverage of the ensemble-winner and ONNX export branches.

## [2.2.1] - 2026-08-24

### Added
- Enforced line, function, and region coverage floors, plus nightly LLVM branch
  coverage artifacts.
- Expanded the installed-wheel Python suite to cover validation, classification,
  regression, search, explainability, profiling, and ONNX round trips.
- Added protected MSRV, branch-coverage, Windows, and macOS checks on `main`.

### Changed
- Isolated ONNX native execution, profile rendering, advanced transforms, and
  Python data/EDA bindings into focused production modules.
- Hardened Dependabot so compatibility-boundary and exact-pinned engine upgrades
  remain deliberate, while safe GitHub Actions updates stay automated.
- Updated GitHub Actions and kept release publishing on trusted OIDC workflows.

### Fixed
- Version synchronization now includes the root `Cargo.lock` package entry and
  the maintained Python version assertion.
- Removed the obsolete standalone Python smoke script after migrating CI to the
  pytest behavioral suite.
- Corrected the MSRV workflow selector so dependency automation cannot rewrite
  Rust 1.95 as an action version.

## [2.2.0] - 2026-08-24

### Added
- Complete Rust and Python ML lifecycle: typed ingest and EDA, preprocessing,
  model selection and HPO, ensembles, diagnostics and explainability, ONNX
  portability, serving, registry, drift monitoring, specialized estimators, and
  AutoML.
- Cross-platform Python abi3 wheels and synchronized crates.io/PyPI releases.

### Changed
- Hardened feature isolation, packaging, MSRV, SemVer, advisory, example, and
  benchmark checks across the full CI matrix.
- Updated the Python extension to PyO3 0.29 and documented the stable 2.2 API.

### Security
- Added explicit dependency advisory handling and removed vulnerable PyO3
  releases from the resolved graph.

## [0.2.1] - 2026-08-23

### Added
- **Exported forests now serve in Millwright.** `InferenceModel` runs linear / NN
  ONNX graphs through tract as before, and evaluates the ONNX-ML tree-ensemble
  ops tract doesn't implement (from an exported `RandomForest`) with a small
  native interpreter. A model exported by Millwright always round-trips back into
  `InferenceModel`/`Server` — forests included — while the artifact stays portable
  to any ONNX runtime.
- **Imputers and one-hot encoders are ONNX-exportable.** A `SimpleImputer` step
  exports as `Where(IsNaN(x), fill, x)`; a `OneHotEncoder` step exports as a
  per-column `Gather` → `Round` → `Equal` → `Cast` → `Concat` expansion (with the
  graph input re-declared at the raw feature width). So the whole realistic
  `impute → one-hot → scale → model` pipeline exports and serves as one graph —
  through tract (linear) or the native interpreter (behind a forest), verified
  identical to the in-memory pipeline. The pipeline export generalized from
  folding one affine map to splicing an ordered chain of transformer "prefixes"
  (new `Transformer::onnx_prefix`). Only steps with no ONNX form now error.
- **`SearchResult::export_onnx`** — a `GridSearch`/`RandomSearch` winner can now
  be exported to ONNX (it could only `predict` before). Backed by a new
  object-safe `Estimator::to_onnx_proto` on `Pipeline`.
- **`Server::from_registry(&reg, name, tag)`** and
  **`DriftMonitor::from_registry(&version)`** — serve a tagged artifact straight
  from a registry, and build a PSI monitor from the version's stored reference
  distribution.

## [0.2.0] - 2026-08-23

### Added
- **rayon parallelism.** Cross-validation folds and search candidates now
  evaluate in parallel: `cross_val_score` is fold-parallel, `Bagging` fits its
  base estimators in parallel, and `AutoML::parallel()` adds candidate-level
  parallelism. The core contract traits gained `Send + Sync` bounds (every
  concrete model already satisfied them). Results stay seed-reproducible.
- **`Boosting`** — SAMME adaptive boosting over any weak learner (an
  `alpha`-weighted vote of models each reweighted toward the last round's
  mistakes), joining `Voting`/`Bagging`/`Stacking`.
- **Three more models over the smartcore backend:** `Knn` (k-nearest-neighbours),
  `Svc` (support vector classifier, linear or RBF, one-vs-one for multiclass),
  and `NaiveBayes` (Gaussian). All implement the same `Estimator`/`Predictor`
  contract, so they drop into pipelines, ensembles, and search unchanged — and
  they are exposed to Python too (`mw.Knn`, `mw.Svc` / `mw.Svc.rbf()`,
  `mw.NaiveBayes`, plus `pipe.knn()` / `pipe.svc()` / `pipe.naive_bayes()`).
- **Python: a scikit-learn-shaped object API.** `mw.Frame` with `from_pandas` /
  `from_numpy` / `from_rows` ingest; composable transformer/estimator objects
  (`StandardScaler`, `MinMaxScaler`, `SimpleImputer`, `OneHotEncoder`,
  `RandomForest`, `LinearRegression`) added via `pipe.step(name, obj)` /
  `pipe.estimator(name, obj)`; `fit`/`predict`/`evaluate` accept a `Frame`.
- **Python: `pipeline.explain(...)`** returns SHAP feature importance
  (`Explainer.kernel()` configurable), and **`pipeline.export_onnx(path)`**
  writes the fitted pipeline to a single ONNX file. The `python` wheel now
  bundles the `model-selection`, `explain`, and `onnx` engines.
- **Python: `GridSearch` / `KFold` / `StratifiedKFold`** over a pipeline, with
  a `SearchResult` (`best_score`, `best_params()`, `predict()`).
- **`InferenceModel` is now an `Estimator` + `Predictor`**, so a pre-trained
  ONNX model (from scikit-learn, PyTorch, …) can be dropped into a `Pipeline`
  as a frozen estimator behind Millwright's preprocessing — in Rust and, via
  `mw.OnnxModel(path)`, from Python.
- **Python: `mw.Table` + `mw.Profile`** — dtype-aware CSV/Parquet ingest and
  automated EDA (`Profile.of(table_or_frame).to_html(path)`). The wheel now
  bundles the `eda` (polars) engine, so it is larger than the pure-model build.
- **`Table::from_frame`** (Rust): build a numeric `Table` from a `Frame`, so the
  numeric world can round-trip back into the typed one (e.g. to profile it).
- **Richer EDA in `Profile`** — excess `kurtosis` and z-score outlier counts per
  numeric column; a **Spearman** rank-correlation matrix beside the Pearson one;
  a **co-missing** map (columns whose null patterns correlate); high-cardinality
  categoricals now suggest `TargetEncoder`; and `suggest_pipeline` adds a
  train-time **SMOTE** balancer on class imbalance (with `preprocessing`).
- **AutoML is seeded by EDA.** With the `eda` engine on, the search fixes its
  preprocessing to `Profile::suggest_pipeline()` and varies only the model,
  pruning the space (it falls back to a scaler sweep without `eda`).

### Fixed
- **Cross-validated F1 no longer returns `NaN`.** A fold whose predictions
  contain no true positives left smartcore's F1 evaluating `0/0`; a single NaN
  fold poisoned the CV mean and a search's `best_score`. An undefined F1 is now
  0.0 (scikit-learn's `zero_division=0` convention), so `GridSearch(...).scoring(F1)`
  yields finite, comparable scores.

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
- MSRV is **1.95** (dep-dictated — `sysinfo` via tract, and polars), declared
  through `rust-version` and exercised by a dedicated CI job.
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
