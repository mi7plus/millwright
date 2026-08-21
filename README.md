# millwright

A unified ML framework for Rust — *ten crates, one lifecycle.*

See [`Millwright.html`](index.html) / [`millwright-design-brief.pdf`](millwright-design-brief.pdf)
for the full design brief.

## Status: Phases 0–3 — done

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
  `MinMaxScaler`, `OneHotEncoder`.
- **Balancing** (`src/balance.rs`, via [`imbalance-rs`]): `Smote`,
  `RandomOverSampler` as train-time `Balancer`s — `Pipeline::balance(...)`,
  applied only during `fit`.
- **Model selection** (`src/selection.rs`, via [`model-selection-rs`]):
  `KFold` / `StratifiedKFold`, a `Metric` enum (accuracy, F1, MAE, MSE, RMSE, R²),
  and `GridSearch` / `RandomSearch` over a whole pipeline, tuned by path. `grid!`
  macro included.
- **Ensembles** (`src/ensemble.rs`, core): `Voting` (hard/soft), `Bagging`, and
  leak-free `Stacking` riding the same CV engine — all `Model`s themselves, so
  they compose, tune, and nest.

### Phase 2 · backends & HPO — *two backends, one contract*

- **The second backend** (`src/backends/linfa.rs`, via [`linfa`], feature
  `linfa-backend`): `KMeans`, `GaussianMixture`, `Dbscan` (as a new `Clusterer`
  contract) and `Pca` (as a `Transformer`) — each converting `Frame → ndarray`
  at the edge, proving the boundary conversion against a whole other engine.
- **Bayesian search** (`src/selection.rs`, via [`hyperopt-rs`], feature `hpo`):
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

[`imbalance-rs`]: https://crates.io/crates/imbalance-rs
[`model-selection-rs`]: https://crates.io/crates/model-selection-rs
[`linfa`]: https://crates.io/crates/linfa
[`hyperopt-rs`]: https://crates.io/crates/hyperopt-rs
[`regression-diagnostics`]: https://crates.io/crates/regression-diagnostics
[`shap-rs`]: https://crates.io/crates/shap-rs
[`plotters-statistical`]: https://crates.io/crates/plotters-statistical

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
cargo run --example workflow
```

```bash
cargo run --example backends --features "smartcore-backend linfa-backend hpo"
```

```bash
cargo run --example insight --features "smartcore-backend diagnostics explain viz"
```

### Building on Windows

The default toolchain is MSVC. If a Unix `link.exe` (e.g. from Git/Laragon) is
ahead of MSVC's on `PATH`, linking fails with an "extra operand" error. Build
from a *Developer Command Prompt / PowerShell for VS 2022*, or run `vcvars64.bat`
first, so the MSVC linker is found before the shadowing one.

## Roadmap

Phases 0–3 are done. Phases 4–8 — ONNX & Python, serving & monitoring, time
series & out-of-core, AutoML, and 1.0 hardening — are laid out in the design
brief.
