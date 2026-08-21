# millwright

A unified ML framework for Rust — *ten crates, one lifecycle.*

See [`Millwright.html`](Millwright.html) / [`millwright-design-brief.pdf`](millwright-design-brief.pdf)
for the full design brief.

## Status: Phase 0 (spine) + Phase 1 (prep & select) — done

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

[`imbalance-rs`]: https://crates.io/crates/imbalance-rs
[`model-selection-rs`]: https://crates.io/crates/model-selection-rs

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

### Building on Windows

The default toolchain is MSVC. If a Unix `link.exe` (e.g. from Git/Laragon) is
ahead of MSVC's on `PATH`, linking fails with an "extra operand" error. Build
from a *Developer Command Prompt / PowerShell for VS 2022*, or run `vcvars64.bat`
first, so the MSVC linker is found before the shadowing one.

## Roadmap

Phases 0 (spine) and 1 (prep & select) are done. Phases 2–8 — more backends &
Bayesian HPO, diagnostics & explainability, ONNX & Python, serving & monitoring,
time series & out-of-core, AutoML, and 1.0 hardening — are laid out in the design
brief.
