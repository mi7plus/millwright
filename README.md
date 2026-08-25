# millwright

[![crates.io](https://img.shields.io/crates/v/millwright.svg)](https://crates.io/crates/millwright)
[![docs.rs](https://img.shields.io/docsrs/millwright)](https://docs.rs/millwright)
[![PyPI](https://img.shields.io/pypi/v/millwright.svg)](https://pypi.org/project/millwright/)
[![CI](https://github.com/mi7plus/millwright/actions/workflows/ci.yml/badge.svg)](https://github.com/mi7plus/millwright/actions/workflows/ci.yml)
[![downloads](https://img.shields.io/crates/d/millwright.svg)](https://crates.io/crates/millwright)
[![license](https://img.shields.io/crates/l/millwright.svg)](LICENSE)

A unified ML framework for Rust — *fit. predict. serve. watch.*

- **Tutorial:** the hands-on docs site at <https://millwright-rs.dev/docs/>
  (source in [`docs/`](docs/index.html); a short [`GUIDE.md`](GUIDE.md) has the
  quickstart).
- **Contributing:** see [`CONTRIBUTING.md`](CONTRIBUTING.md).
- **Design brief:** the *why*, at **<https://millwright-rs.dev/>**.

It assembles focused Rust ML engines behind one stable data model and trait
contract, so training, evaluation, export, serving, and monitoring compose.

## Status: Phases 0–8 — done

### Phase 0 · the spine

`fit · transform · predict · Pipeline` end to end over a real backend:

- **`Frame` / `Dataset`** — the contiguous, row-major `f64` boundary type
  (`src/frame.rs`).
- **The four traits** — object-safe `Transformer`, `Estimator`, `Predictor`,
  `ProbaPredictor`, plus a blanket `Model` (`src/traits.rs`).
- **The first backend** — a smartcore adapter (`RandomForest`,
  `LinearRegression`) converting `Frame → DenseMatrix` at the edge only
  (`src/backends/smartcore.rs`).
- **`Pipeline`** — named steps + a final model, `"step__param"` addressing;
  pipelines nest (`src/pipeline.rs`).

### Phase 1 · prep & select — *a real, tunable, ensemble-ready workflow*

- **Preprocessing** (`src/transform.rs`, core): `SimpleImputer`, `StandardScaler`,
  `MinMaxScaler`, `OneHotEncoder`, `Winsorize` (clip outliers), `PowerTransform`
  (Yeo-Johnson), `ColumnTransformer` (per-subset transforms), and the supervised
  `TargetEncoder`.
- **Balancing** (`src/balance.rs`, via [`imbalance-rs`]): `Smote`,
  `RandomOverSampler` as train-time `Balancer`s — `Pipeline::balance(...)`,
  applied only during `fit`.
- **Model selection** (`src/selection/`, via [`model-selection-rs`]):
  `KFold` / `StratifiedKFold`, a `Metric` enum (accuracy, F1, MAE, MSE, RMSE, R²),
  and `GridSearch` / `RandomSearch` over a whole pipeline, tuned by path. `grid!`
  macro included.
- **Ensembles** (`src/ensemble.rs`, core): `Voting` (hard/soft), `Bagging`,
  `Boosting`, and leak-free `Stacking` riding the same CV engine — all `Model`s
  themselves, so they compose, tune, and nest. Set `EnsembleTask::Regression`
  explicitly when a regression target happens to contain only integers.

### Phase 2 · backends & HPO — *two backends, one contract*

- **The second backend** (`src/backends/linfa.rs`, via [`linfa`], feature
  `linfa-backend`): `KMeans`, `GaussianMixture`, `Dbscan` (as a new `Clusterer`
  contract) and `Pca` (as a `Transformer`) — each converting `Frame → ndarray`
  at the edge, proving the boundary conversion against a whole other engine.
- **Bayesian search** (`src/selection/`, via [`hyperopt-rs`], feature `hpo`):
  `BayesSearch` runs TPE search over a `SearchSpace` and returns the *same*
  `SearchResult` as grid/random search — one search API, three strategies.

### Phase 3 · insight — *trust the model, not just run it*

- **Evaluation reports** (`src/evaluate.rs`, core): `model.evaluate(&test)` bundles
  task-appropriate metrics into a `Report` (accuracy/precision/recall/F1 or
  MAE/MSE/RMSE/R²).
- **Regression diagnostics** (`src/diagnostics.rs`, via [`regression-diagnostics`],
  feature `diagnostics`): `Diagnostics::of(&data)` runs OLS and exposes
  `summary()`, R², per-column VIF, residuals, and Cook's distance.
- **Explainability** (`src/explain.rs`, via [`shap-rs`], feature `explain`):
  `model.explain(&Explainer::kernel(), &frame)` gives per-row SHAP values and
  global importance, plus `permutation_importance(...)`.
- **Report figures** (`src/viz.rs`, via [`plotters-statistical`], feature `viz`):
  `viz::roc_svg(...)` and `viz::residuals_svg(...)` render self-contained SVGs
  (pure-Rust backend, no system fonts).
- **Probabilities** (`src/logistic.rs`, core): `LogisticRegression` is a native,
  probability-capable classifier — the first real `ProbaPredictor`.
- **Calibration** (`src/calibration.rs`, feature `calibration`): `PlattScaling` /
  `IsotonicRegression` and `reliability_curve`, plus `CalibratedClassifier`,
  which wraps any `ProbaPredictor` and returns calibrated probabilities.
- **Anomaly detection** (`src/anomaly.rs`, feature `anomaly`): `Mahalanobis` and
  `KnnScore`, unified behind an `OutlierDetector` trait.

[`imbalance-rs`]: https://crates.io/crates/imbalance-rs
[`model-selection-rs`]: https://crates.io/crates/model-selection-rs
[`linfa`]: https://crates.io/crates/linfa
[`hyperopt-rs`]: https://crates.io/crates/hyperopt-rs
### Phase 4 · portability & Python — *train once; run in Rust, Python, or any ONNX runtime*

- **ONNX export** (`src/onnx.rs`, via [`onnx-export-rs`], feature `onnx`):
  `model.export_onnx(path)` for `RandomForest` (ONNX-ML tree ensemble) and
  `LinearRegression`; **whole-pipeline export** folds affine scalers into the
  estimator's graph as one `.onnx`.
- **Inference** (via [`tract`], feature `onnx`): `InferenceModel::load(path)`
  loads and runs any ONNX file. tract executes the linear/affine/pipeline graphs
  (a full round-trip); tree-ensemble ONNX-ML artifacts run in external runtimes
  like onnxruntime.
- **Python bindings** (`src/python.rs`, via [`pyo3`], feature `python`): a
  `Pipeline` class over the same Rust core, shipped on [PyPI](https://pypi.org/project/millwright/)
  as an abi3 wheel.

```bash
pip install millwright
```

```python
import millwright as mw
pipe = mw.Pipeline()
pipe.standard_scaler()
pipe.random_forest(n_trees=100, max_depth=8)
pipe.fit(rows, labels)          # list[list[float]], list[float]
preds = pipe.predict(rows)      # runs the Rust engine
```

To build from source (contributors), from a virtualenv: `maturin develop --features python`.

[`regression-diagnostics`]: https://crates.io/crates/regression-diagnostics
[`shap-rs`]: https://crates.io/crates/shap-rs
[`plotters-statistical`]: https://crates.io/crates/plotters-statistical
### Phase 5 · operations — *past where scikit-learn stops*

- **Registry** (`src/registry.rs`, feature `registry`): `Registry::local(path)`
  versions a model's ONNX artifact, content-addressed (identical models dedupe),
  with metadata + reference distribution, movable tags, and `rollback`.
- **Drift monitor** (`src/monitor.rs`, via [`driftwatch`], feature `monitor`):
  `DriftMonitor::psi(reference)` watches the prediction stream — `observe` +
  `report` give live PSI and a drift verdict.
- **Server** (`src/serve.rs`, via [`axum`], feature `serve`): `Server::from_onnx`
  exposes `POST /predict` (validated) over the tract runtime; with a monitor
  attached, every request feeds it and `GET /metrics` reports drift.

```rust
Server::from_onnx(reg.onnx_path("churn", "prod")?)?
    .route("/predict")
    .with_monitor(DriftMonitor::psi(&reference)?)
    .serve("0.0.0.0:8080").await?;
```

[`onnx-export-rs`]: https://crates.io/crates/onnx-export-rs
[`tract`]: https://crates.io/crates/tract-onnx
[`pyo3`]: https://pyo3.rs/
### Phase 6 · specialized — *the long tail of real workloads*

Same contract, different data shapes — each gets its own trait.

- **Time series** (`src/backends/chronos.rs`, via [`chronos-ts`], feature
  `timeseries`): `AutoArima` implements a `Forecaster` — `fit(&series)` then
  `forecast(steps)`.
- **Out-of-core** (`src/backends/incremental.rs`, via [`incremental-rs`], feature
  `incremental`): `IncrementalLinear` implements `PartialFit` + `Predictor` —
  `partial_fit(&batch)` learns one batch at a time.

These two crates pin `ndarray 0.15` while the rest of the stack uses `0.16`;
Cargo links both, and the boundary conversion happens only inside these
adapters — the "two ndarray worlds" the design settles, now exercised for real.

[`driftwatch`]: https://crates.io/crates/driftwatch
[`axum`]: https://crates.io/crates/axum
### Phase 7 · synthesis — *auto-sklearn, but the output actually deploys*

- **AutoML** (`src/automl.rs`, feature `automl`): `AutoML::classifier()` /
  `regressor()` searches preprocessing × model × hyperparameters under a
  `Budget` (trials or minutes), auto-ensembles the top candidates, and returns a
  ranked leaderboard plus the best fitted model. **No new crate** — it
  orchestrates the model-selection, ensemble, and backend machinery already
  built. A single-pipeline winner flows straight into `export_onnx`, so unlike a
  TPOT object the result deploys.

```rust
let result = AutoML::classifier()
    .budget(Budget::trials(40))
    .metric(Metric::F1)
    .cv(StratifiedKFold::new(5))
    .fit(&train)?;
println!("{}", result.leaderboard());
result.export_onnx("model.onnx")?;   // deployable
```

[`chronos-ts`]: https://crates.io/crates/chronos-ts
[`incremental-rs`]: https://crates.io/crates/incremental-rs

### Phase 8 · harden → 1.0 — *a framework you can bet on*

Pin, prove, document — owning the one real risk of assembling young,
single-author engine crates.

- **Exact-version pins** (`Cargo.toml`): every engine — the ecosystem crates plus
  the smartcore and linfa families — is pinned to an exact `=x.y.z`, so a stray
  `cargo update` can't move a fragile engine under the stable trait contract.
  General infrastructure (serde, tokio, axum, …) stays on caret ranges to avoid
  forcing conflicts downstream.
- **Committed `Cargo.lock`**: the whole ~300-package graph is reproducible; CI
  builds with `--locked`.
- **Golden-output tests** (`tests/golden.rs`): lock the numeric behaviour of the
  engines on fixed inputs — exact for the deterministic paths (OLS, affine
  transforms, metric formulas), well-separated class labels for the stochastic
  ones. An engine bump that moves a number shows up as a diff.
- **Feature-matrix CI** (`.github/workflows/ci.yml`): `fmt`, `clippy -D warnings`,
  docs, and the test suite across the feature matrix — from `--no-default-features`
  through each feature to `full` — plus Windows/macOS, the runnable examples, a
  benchmark compile-check, a `cargo publish --dry-run`, and a maturin wheel. The
  MSRV (`rust-version = 1.95`, dep-dictated) is enforced by cargo for consumers.
- **The tutorial** ([`GUIDE.md`](GUIDE.md) + [`guide.html`](guide.html)): the
  design brief's lifecycle, re-cast as a hands-on guide.

### Ingest & EDA — *the lifecycle starts where the data does*

The front of the lifecycle, behind the `eda` feature (via [`polars`]).

- **`Table`** (`src/table.rs`): a dtype-aware, polars-backed table —
  `Table::from_csv` / `from_parquet` read real string/categorical/datetime/null
  columns. It *lowers* to the numeric world: `table.to_frame()` and
  `table.into_dataset("target")` (categoricals label-encoded, nulls → `NaN`), so
  `Frame` stays the numeric boundary everything else already speaks.
- **`Profile`** (`src/profile.rs`): `Profile::of(&table)` returns a *typed* EDA —
  overview, per-column numeric/categorical profiles, missingness, Pearson
  correlations (high-|r| pairs flagged), IQR outliers, and target relationship
  (class balance or feature-target correlation). It renders a self-contained
  `to_html(path)` report, lists `alerts()` that name the fix, and — the loop
  scikit-learn can't close — `suggest_pipeline()` drafts the preprocessing from
  those findings; you just add the model.

```rust
let table = Table::from_csv("customers.csv")?;
let profile = Profile::of_with_target(&table, "churned")?;
profile.to_html("eda.html")?;

let train = table.into_dataset("churned")?;
let mut pipe = profile.suggest_pipeline()      // impute · encode · scale, from the alerts
    .estimator("rf", RandomForest::new());
pipe.fit(&train)?;
```

[`polars`]: https://crates.io/crates/polars

### Quickstart

```rust
use millwright::grid;
use millwright::prelude::*;

let pipe = Pipeline::new()
    .step("impute", SimpleImputer::median())
    .step("scale", StandardScaler::new())
    .balance(Smote::new())                 // train-time only
    .estimator("rf", RandomForest::new());

let search = GridSearch::new(pipe, grid! { "rf__max_depth" => [4, 8, 16] })
    .cv(StratifiedKFold::new(5))
    .scoring(Metric::F1)
    .fit(&train)?;

println!("best F1 = {:.3}", search.best_score());
let preds = search.predict(&test)?;
```

Run the end-to-end examples:

```bash
cargo run --example spine
```

```bash
cargo run --example explore --features "eda smartcore-backend"
```

```bash
cargo run --example trust --features "calibration anomaly"
```

```bash
cargo run --example workflow
```

```bash
cargo run --example backends --features "smartcore-backend linfa-backend hpo"
```

```bash
cargo run --example insight --features "smartcore-backend diagnostics explain viz"
```

```bash
cargo run --example portability --features "smartcore-backend onnx"
```

```bash
cargo run --example operations --features "smartcore-backend onnx registry monitor serve"
```

```bash
cargo run --example specialized --features "timeseries incremental"
```

```bash
cargo run --example automl --features "smartcore-backend automl onnx"
```

### Building on Windows

The default toolchain is MSVC. If a Unix `link.exe` (e.g. from Git/Laragon) is
ahead of MSVC's on `PATH`, linking fails with an "extra operand" error. Build
from a *Developer Command Prompt / PowerShell for VS 2022*, or run `vcvars64.bat`
first, so the MSVC linker is found before the shadowing one.

## Roadmap

Phases 0–8 are done — the full lifecycle plus 1.0 hardening (exact-version pins,
a committed lockfile, golden-output tests, and a feature-matrix CI). The design
brief lays out the arc; the tutorial ([`GUIDE.md`](GUIDE.md)) is the how.
