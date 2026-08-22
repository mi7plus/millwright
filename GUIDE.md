# The Millwright Guide

*A unified ML framework for Rust — ten crates, one lifecycle.*

The full hands-on tutorial now lives as a browsable, multi-page site:

### → **[millwright-rs.dev/docs/](https://millwright-rs.dev/docs/)**  (source in [`docs/`](docs/index.html))

It walks the whole lifecycle, one topic per page: **Data & EDA** · **Pipelines &
Models** · **Insight** (evaluate, explain, calibrate, detect) · **Deploy**
(ONNX, serving, registry, AutoML) · **Python**.

## Quickstart

```toml
[dependencies]
millwright = "0.1"          # or features = ["full"] for the whole lifecycle
```

```rust
use millwright::prelude::*;

// features as rows + a target -> a Dataset
let train = Dataset::new(x, y)?;

// standardize, then a random forest — one composable object
let mut pipe = Pipeline::new()
    .step("scale", StandardScaler::new())
    .estimator("rf", RandomForest::new());

pipe.fit(&train)?;
let preds = pipe.predict(&test)?;
```

## Where to go

- **Tutorial:** <https://millwright-rs.dev/docs/> — the full, hands-on walk-through.
- **API reference:** [docs.rs/millwright](https://docs.rs/millwright).
- **Design brief (the *why*):** <https://millwright-rs.dev/>.
- **Examples:** [`examples/`](examples) — a runnable program for each feature group.
- **Python:** [`pip install millwright`](https://pypi.org/project/millwright/).
