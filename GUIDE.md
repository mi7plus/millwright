# The Millwright Guide

*A unified ML framework for Rust — ten crates, one lifecycle.*

This is the hands-on tutorial: it walks the whole machine-learning lifecycle —
from a raw table of numbers to a served, drift-monitored model — using one data
model, one trait contract, and one pipeline. Every capability is a cargo feature
over a single crate, so you compile only what you use.

If you want the *why* — the design rationale, the architecture diagram, the
honest boundaries — read the [design brief](https://millwright-rs.dev/). This
guide is the *how*.

Every code block below is real API, mirrored from the runnable programs in
[`examples/`](examples). Each example names the features it needs; run them as
you read:

```bash
cargo run --example spine
```

---

## Contents

1. [Install & features](#install--features)
2. [Ingest & EDA: `Table` and `Profile`](#ingest--eda-table-and-profile)
3. [The data model: `Frame` and `Dataset`](#the-data-model-frame-and-dataset)
4. [The contract: four traits](#the-contract-four-traits)
5. [Pipelines: compose, then tune by path](#pipelines-compose-then-tune-by-path)
6. [Preprocessing & balancing](#preprocessing--balancing)
7. [Model selection: cross-validation & search](#model-selection-cross-validation--search)
8. [Ensembles: combine, even across backends](#ensembles-combine-even-across-backends)
9. [A second backend: linfa](#a-second-backend-linfa)
10. [Insight: evaluate, diagnose, explain, visualize](#insight-evaluate-diagnose-explain-visualize)
11. [Portability: ONNX in and out](#portability-onnx-in-and-out)
12. [Operations: registry, drift, serving](#operations-registry-drift-serving)
13. [Specialized shapes: time series & out-of-core](#specialized-shapes-time-series--out-of-core)
14. [AutoML: the framework, pointed at itself](#automl-the-framework-pointed-at-itself)
15. [Python](#python)
16. [Reproducibility: pins, lockfile, golden tests, CI](#reproducibility-pins-lockfile-golden-tests-ci)

---

## Install & features

Add the crate and pull the capabilities you need. The `default` install is a
lean, useful core — the smartcore backend, preprocessing, model selection, and
ensembles:

```toml
[dependencies]
millwright = "0.1"
```

Everything past the core is feature-gated. A serving binary never compiles SHAP;
a notebook never compiles `axum`.

```toml
# just the spine — one backend, the four traits, pipelines
millwright = { version = "0.1", default-features = false, features = ["smartcore-backend"] }

# the whole lifecycle
millwright = { version = "0.1", features = ["full"] }
```

| Feature | Adds |
| --- | --- |
| `smartcore-backend` *(default)* | `RandomForest`, `LinearRegression` |
| `preprocessing` *(default)* | `Smote`, `RandomOverSampler` balancers (imputers/scalers/encoders are core) |
| `model-selection` *(default)* | `KFold`/`StratifiedKFold`, `GridSearch`/`RandomSearch`, metrics |
| `ensemble` *(default)* | `Voting`, `Bagging`, `Stacking` |
| `eda` | `Table` (polars CSV/Parquet ingest) + `Profile` (typed EDA) |
| `linfa-backend` | `KMeans`, `GaussianMixture`, `Dbscan`, `Pca` |
| `hpo` | `BayesSearch` (TPE) over a `SearchSpace` |
| `diagnostics` | OLS `Diagnostics`: VIF, residuals, Cook's distance |
| `explain` | `Explainer` (SHAP) + `permutation_importance` |
| `viz` | ROC / residual SVG figures |
| `onnx` | `export_onnx` + `InferenceModel` (tract) |
| `registry` | versioned model `Registry` |
| `monitor` | `DriftMonitor` (PSI) |
| `serve` | `Server` — `POST /predict`, `GET /metrics` |
| `timeseries` | `AutoArima` forecaster |
| `incremental` | `IncrementalLinear` (`partial_fit`) |
| `automl` | `AutoML` search |
| `python` | the `pip install millwright` package |
| `full` | every Rust-facing feature above |

One import brings the whole framework into scope:

```rust
use millwright::prelude::*;
```

---

## Ingest & EDA: `Table` and `Profile`

The lifecycle starts where the data does. Behind the `eda` feature, a
polars-backed **`Table`** reads real CSV/Parquet — strings, categories, dates,
booleans, nulls — and a **`Profile`** reports it and drafts the preprocessing.
`Frame` stays the numeric boundary; `Table` is the typed world *in front of* it.

```rust
use millwright::prelude::*;

let table = Table::from_csv("customers.csv")?;   // or ::from_parquet(...)
println!("{:?}  {:?}", table.shape(), table.column_names());

// a typed profile — not just an HTML blob
let profile = Profile::of_with_target(&table, "churned")?;
println!("{}", profile.summary());
for alert in profile.alerts() {
    println!("{alert}");   // e.g. "[city] categorical (3 levels) → OneHotEncoder"
}
profile.to_html("eda_report.html")?;             // a shareable report
```

The profile is *typed* — `overview()`, `columns()`, `missingness()`,
`correlations()`, `target()`, `alerts()` are all fields you can branch on, with
the HTML report as just one renderer. And because Millwright owns EDA *and* the
pipeline, the profile drafts the starting preprocessing from its own findings —
the loop scikit-learn can't close:

```rust
// lower the typed table to the numeric world for modelling
let train = table.into_dataset("churned")?;      // categoricals encoded, nulls → NaN

// EDA drafts the pipeline; you just add the model
let mut pipe = profile.suggest_pipeline()        // impute · encode · scale, from the alerts
    .estimator("rf", RandomForest::new());
pipe.fit(&train)?;
```

Run the whole thing — CSV in, report + fitted model out:

```bash
cargo run --example explore --features "eda smartcore-backend"
```

## The data model: `Frame` and `Dataset`

Everything the *public* API speaks is a `Frame`: a contiguous, row-major `f64`
buffer plus a schema. It is the boundary type — the thing that lets a
`linfa 0.15`-era model and a `smartcore` `DenseMatrix` meet in one signature
without your code ever naming their array versions. Each backend converts
`Frame` ⇄ its native type *inside the adapter only*.

```rust
use millwright::prelude::*;

// rows of features, with column names
let x = Frame::from_rows(
    vec![
        vec![0.0, 0.1],
        vec![0.4, 0.2],
        vec![9.0, 9.1],
        vec![9.4, 8.7],
    ],
    vec!["a".into(), "b".into()],
)?;

assert_eq!(x.shape(), (4, 2));   // (rows, cols)
let _first_row: &[f64] = x.row(0);
let _first_col: Vec<f64> = x.column(0);
```

A `Dataset` is a `Frame` paired with a target vector — what a supervised
`Estimator` fits on:

```rust
let train = Dataset::new(x.clone(), vec![0.0, 0.0, 1.0, 1.0])?;
let _features = train.features();  // &Frame
let _target = train.target();      // &[f64]
```

The task (classification vs. regression) is inferred from the target: an
all-integral target is treated as class labels, anything else as regression.

---

## The contract: four traits

Four object-safe traits are the whole core. Object-safe means a `Pipeline` can
hold a heterogeneous `Vec<Box<dyn …>>` — everything composes because everything
speaks the same contract.

| Trait | Method | Meaning |
| --- | --- | --- |
| `Transformer` | `fit(&mut self, &Frame)` → `transform(&Frame) -> Frame` | learn column stats, reshape features |
| `Estimator` | `fit(&mut self, &Dataset)` | learn a model from features + target |
| `Predictor` | `predict(&Frame) -> Vec<f64>` | one prediction per row |
| `ProbaPredictor` | `predict_proba(&Frame) -> Frame` | class probabilities |

A blanket `Model` trait ties `Estimator + Predictor` together, and a blanket
`Evaluate` gives every predictor a `.evaluate(&test)`. You rarely name these
directly — you compose the types that implement them.

```rust
let mut rf = RandomForest::new().n_trees(50);
rf.fit(&train)?;              // Estimator::fit(&Dataset)
let preds = rf.predict(&x)?;  // Predictor::predict(&Frame) -> Vec<f64>, one per row
```

---

## Pipelines: compose, then tune by path

A `Pipeline` is a sequence of named transformer steps and one final estimator,
as a single object that is *itself* a `Model`. Fitting runs each transform in
turn, then fits the estimator on the transformed frame; predicting replays the
fitted transforms before the model.

```rust
let mut pipe = Pipeline::new()
    .step("scale", StandardScaler::new())
    .estimator("rf", RandomForest::new());

// Tune a parameter deep in the chain by path — the scikit-learn "step__param"
// convention. Steps are addressable by name.
pipe.set_param("rf__n_trees", ParamValue::Int(50))?;
pipe.set_param("rf__max_depth", ParamValue::Int(4))?;

pipe.fit(&train)?;
let preds = pipe.predict(&x)?;
```

That path addressing is what lets a *search* reach any parameter anywhere in the
chain (next section). Pipelines nest, so a pipeline can be a step in another.

---

## Preprocessing & balancing

The core transformers need no extra dependencies and are always available:

```rust
// impute missing values, scale, and one-hot encode
let mut pipe = Pipeline::new()
    .step("impute", SimpleImputer::median())     // or ::mean() / ::constant(0.0)
    .step("scale",  StandardScaler::new())        // or MinMaxScaler::new()
    .step("encode", OneHotEncoder::infer())       // or ::columns(["city"])
    .estimator("rf", RandomForest::new());
```

*Balancers* (from `imbalance-rs`, feature `preprocessing`) are train-time only:
they resample the training set during `fit` and are skipped at predict time, so
they never distort inference.

```rust
let pipe = Pipeline::new()
    .step("impute", SimpleImputer::median())
    .step("scale",  StandardScaler::new())
    .balance(Smote::new().k_neighbors(3).random_state(0))  // train-time only
    .estimator("rf", RandomForest::new());
```

---

## Model selection: cross-validation & search

Search runs over the *whole pipeline*, cross-validated, tuning parameters by
path. The `grid!` macro spells the grid; `StratifiedKFold` keeps class balance
across folds; `Metric` picks the score.

```rust
use millwright::grid;

let search = GridSearch::new(pipe, grid! { "rf__max_depth" => [2, 4, 8] })
    .cv(StratifiedKFold::new(4))
    .scoring(Metric::F1)
    .fit(&train)?;

for (params, score) in search.leaderboard() {
    println!("  {score:.3}  {params:?}");
}
println!("best F1 = {:.3}", search.best_score());
let preds = search.predict(&probe)?;   // the refit best model
```

`RandomSearch` swaps the grid for random draws; with the `hpo` feature,
`BayesSearch` runs TPE over a `SearchSpace` and returns the *same* `SearchResult`
— one search API, three strategies:

```rust
let space = SearchSpace::new().int("max_depth", 1, 12);
let search = BayesSearch::new(RandomForest::new(), space)
    .n_trials(12)
    .seed(0)
    .cv(StratifiedKFold::new(4))
    .scoring(Metric::F1)
    .fit(&train)?;
println!("best F1 = {:.3} at {:?}", search.best_score(), search.best_params());
```

---

## Ensembles: combine, even across backends

Because every model is a `Predictor`, combining models is just another
`Predictor` that holds several — no new machinery, and it works *across
backends*.

```rust
// soft (mean-probability) vote across two forests
let mut vote = Voting::soft()
    .add("rf_shallow", RandomForest::new().max_depth(2))
    .add("rf_deep",    RandomForest::new().max_depth(8));
vote.fit(&train)?;

// stacking: a meta-learner on leak-free out-of-fold base predictions
let mut stack = Stacking::meta(RandomForest::new().n_trees(50))
    .base("rf",  RandomForest::new().n_trees(30))
    .base("rf2", RandomForest::new().max_depth(3))
    .cv(StratifiedKFold::new(4));   // folds come from the CV engine → leak-free
stack.fit(&train)?;

// bagging fans out over any estimator
let bag = Bagging::of(RandomForest::new()).n_estimators(20);
```

An ensemble *is* an `Estimator`, so you can grid-search a member straight
through it (`"rf__max_depth"`), pipeline it, and export it.

---

## A second backend: linfa

The `linfa-backend` feature adds unsupervised models through the *same* boundary
`Frame` — the proof that the two-`ndarray`-worlds problem is settled by design.
Clustering models implement a `Clusterer` contract; `Pca` is a `Transformer`.

```rust
let mut km = KMeans::new(2);
km.fit(&x)?;
println!("k-means labels: {:?}", km.predict(&x)?);

let dbscan = Dbscan::new(3).tolerance(1.0);
println!("dbscan: {:?}", dbscan.fit_predict(&x)?);

let mut pca = Pca::new(1);
let reduced = pca.fit_transform(&x)?;   // a Frame with fewer columns
```

---

## Insight: evaluate, diagnose, explain, visualize

Any `Predictor` can score itself on a labelled `Dataset` (core, no features
needed):

```rust
let mut rf = RandomForest::new().n_trees(60);
rf.fit(&train)?;
print!("{}", rf.evaluate(&test)?);   // accuracy / precision / recall / F1 (or MAE/MSE/RMSE/R²)
```

With `explain`, get SHAP values and permutation importance on the fitted model:

```rust
let shap = rf.explain(&Explainer::kernel().nsamples(80), test.features())?;
println!("SHAP importance: {:?}", shap.importance());
let perm = permutation_importance(&rf, &test, 8, 0)?;
```

With `diagnostics`, run OLS regression diagnostics — VIF, residuals, influence:

```rust
let diag = Diagnostics::of(&reg)?;
println!("R² = {:.4}, max Cook's D = {:.4}", diag.r_squared(), diag.max_cooks_distance());
println!("VIF = {:?}", diag.vif());
```

With `viz`, render self-contained SVG figures (a pure-Rust backend — no system
fonts):

```rust
let scores = rf.predict(test.features())?;
let auc = viz::roc_svg(test.target(), &scores, "roc.svg", (520, 420))?;
viz::residuals_svg(reg.target(), &y_pred, "residuals.svg", (520, 420))?;
```

---

## Portability: ONNX in and out

With `onnx`, any model — or a whole pipeline — exports to a single `.onnx` file.
Whole-pipeline export folds leading affine transformers (scalers) into the
estimator's graph, producing one self-contained graph: raw features in,
predictions out. `InferenceModel::load` runs any ONNX file back through tract.

```rust
let mut pipe = Pipeline::new()
    .step("scale", StandardScaler::new())
    .estimator("lr", LinearRegression::new());
pipe.fit(&train)?;
let native = pipe.predict(&probe)?;

pipe.export_onnx("pipeline.onnx")?;                 // scaler + model, one graph
let model = InferenceModel::load("pipeline.onnx")?;
let via_onnx = model.predict(&probe)?;              // matches `native`
```

Linear/affine/pipeline graphs run *inside* tract for a full round-trip. A random
forest exports to a valid ONNX-ML tree-ensemble artifact for external runtimes
(onnxruntime); tract implements NN ops, not the ONNX-ML tree ops.

---

## Operations: registry, drift, serving

This is where Millwright runs past where scikit-learn stops.

**Registry** (`registry`) versions the ONNX artifact, content-addressed, with
metadata and a reference distribution; tags are movable pointers you can roll
back:

```rust
let reg = Registry::local("./models");
let v1 = reg.register("demand", &model, Metadata {
    metrics: vec![("r2".into(), 1.0)],
    reference: reference.clone(),   // the training distribution to watch against
    note: "baseline".into(),
})?;
reg.tag("demand", &v1.id, "prod")?;
// … promote a v2, then revert in one line:
let reverted = reg.rollback("demand", "prod")?;
```

**Drift monitor** (`monitor`) watches the prediction stream for PSI drift against
that reference:

```rust
let monitor = DriftMonitor::psi(&reference)?;
monitor.observe(&model.predict(probe)?);
println!("{:?}", monitor.report()?);   // live PSI + a drift verdict
```

**Server** (`serve`) exposes a validated `POST /predict` over the tract runtime;
attach a monitor and every request feeds it, with drift at `GET /metrics`:

```rust
Server::from_onnx(reg.onnx_path("demand", "prod")?)?
    .route("/predict")
    .with_monitor(DriftMonitor::psi(&reference)?)
    .serve("0.0.0.0:8080").await?;
```

---

## Specialized shapes: time series & out-of-core

Same contract, different data shapes — each gets its own trait.

**Time series** (`timeseries`): `AutoArima` implements a `Forecaster`.

```rust
let mut arima = AutoArima::new().max_p(3).max_q(3);
arima.fit(&series)?;                 // &[f64]
let forecast = arima.forecast(6)?;   // six steps ahead
```

**Out-of-core** (`incremental`): `IncrementalLinear` implements `PartialFit`,
learning one batch at a time without ever holding the whole dataset in memory.

```rust
let mut model = IncrementalLinear::with_rate(0.05, 0.0);
for batch in batches {              // each batch is a Dataset
    model.partial_fit(&batch)?;
}
let preds = model.predict(&probe)?;
```

These two crates pin `ndarray 0.15` while the rest of the stack uses `0.16`;
Cargo links both, and the conversion happens only inside these adapters — the
"two `ndarray` worlds" the design settles, exercised for real.

---

## AutoML: the framework, pointed at itself

Profiling, preprocessing, cross-validation, search, and ensembling are exactly
what an AutoML engine needs — so `AutoML` is not a bolt-on, it is the framework
orchestrating its own parts. Point it at data and a budget; get back a
leaderboard and the best *deployable* model.

```rust
let result = AutoML::classifier()      // or ::regressor()
    .budget(Budget::trials(20))         // or Budget::minutes(10)
    .metric(Metric::F1)
    .cv(StratifiedKFold::new(5))
    .seed(0)
    .fit(&train)?;

println!("{}", result.leaderboard());
println!("winner: {} (F1 = {:.3})", result.best_label(), result.best_score());
let preds = result.predict(&probe)?;

// A single-pipeline winner is deployable — unlike a TPOT object.
result.export_onnx("model.onnx")?;
```

---

## Python

The `python` feature ships a `pip install millwright` package: the same Rust
engine behind a Pythonic API, built with maturin into an abi3 wheel.

```python
import millwright as mw

pipe = mw.Pipeline()
pipe.standard_scaler()
pipe.random_forest(n_trees=100, max_depth=8)

pipe.fit(rows, labels)          # list[list[float]], list[float]
preds = pipe.predict(rows)      # runs the Rust engine
```

Build it from a virtualenv:

```bash
maturin develop --features python
```

The `python` feature is deliberately *not* part of `full`: pyo3's
`extension-module` defers libpython symbols, so a plain `cargo test` can't link a
test binary with it. It is built and tested the way it ships — as a wheel, via
maturin.

---

## Reproducibility: pins, lockfile, golden tests, CI

Millwright assembles young, single-author engine crates. That is its real risk,
and Phase 8 owns it directly — *a framework you can bet on*:

- **Exact-version pins.** Every engine (the ecosystem crates plus the smartcore
  and linfa families) is pinned to an exact `=x.y.z` in `Cargo.toml`. A stray
  `cargo update` can never silently move a fragile engine underneath the stable
  trait contract; bumps are deliberate, one line, one commit. General-purpose
  infrastructure (serde, tokio, axum, …) stays on caret ranges to avoid forcing
  version conflicts downstream.
- **Committed `Cargo.lock`.** The whole ~300-package graph is reproducible. CI
  builds with `--locked`, so a drifted lockfile is a hard error.
- **Golden-output tests.** [`tests/golden.rs`](tests/golden.rs) locks the
  *numeric* behaviour of the engines on fixed inputs — exact for the
  deterministic paths (OLS, the affine transforms, the metric formulas),
  well-separated class labels for the stochastic ones. If an engine bump moves a
  number, the diff makes it impossible to miss.
- **Feature-matrix CI.** [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
  runs `fmt`, `clippy -D warnings`, docs, an MSRV build, and the test suite
  across the feature matrix — from `--no-default-features` (bare core) through
  each feature to `full`, plus Windows/macOS on the default install and a maturin
  wheel build for Python.

Run the suite yourself:

```bash
cargo test --features full
```

```bash
cargo test --locked --no-default-features --features smartcore-backend
```

---

*Built the trade way — proven parts, assembled into one machine, and kept
running.* For the full rationale and architecture, see the
[design brief](https://millwright-rs.dev/).
