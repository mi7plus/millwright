# millwright

A unified ML framework for Rust — *ten crates, one lifecycle.*

See [`Millwright.html`](Millwright.html) / [`millwright-design-brief.pdf`](millwright-design-brief.pdf)
for the full design brief.

## Status: Phase 0 · the spine

The smallest thing that proves the design — `fit · transform · predict · Pipeline`
working end to end over a real backend. Shipped:

- **`Frame` / `Dataset`** — the contiguous, row-major `f64` boundary type the
  public API speaks (`src/frame.rs`).
- **The four traits** — object-safe `Transformer`, `Estimator`, `Predictor`,
  `ProbaPredictor`, plus a blanket `Model` (`src/traits.rs`).
- **The first backend** — a smartcore adapter (`RandomForest`,
  `LinearRegression`) that converts `Frame → DenseMatrix` at the edge only
  (`src/backends/smartcore.rs`).
- **`Pipeline`** — named transformer steps + a final model, with `"step__param"`
  path addressing; pipelines nest (`src/pipeline.rs`).

One core transformer, `StandardScaler`, ships now to exercise `transform`; the
full preprocessing suite is Phase 1.

### Quickstart

```rust
use millwright::prelude::*;

let x = Frame::from_rows(
    vec![vec![0.0, 0.1], vec![9.0, 9.1]],
    vec!["a".into(), "b".into()],
)?;
let train = Dataset::new(x.clone(), vec![0.0, 1.0])?;

let mut pipe = Pipeline::new()
    .step("scale", StandardScaler::new())
    .estimator("rf", RandomForest::new());

pipe.set_param("rf__n_trees", ParamValue::Int(50))?;
pipe.fit(&train)?;
let preds = pipe.predict(&x)?;
```

Run the end-to-end example:

```bash
cargo run --example spine
```

### Building on Windows

The default toolchain is MSVC. If a Unix `link.exe` (e.g. from Git/Laragon) is
ahead of MSVC's on `PATH`, linking fails with an "extra operand" error. Build
from a *Developer Command Prompt / PowerShell for VS 2022*, or run `vcvars64.bat`
first, so the MSVC linker is found before the shadowing one.

## Roadmap

Phase 0 (spine) is done. Phases 1–8 — preprocessing & CV, more backends & HPO,
diagnostics & explainability, ONNX & Python, serving & monitoring, time series &
out-of-core, AutoML, and 1.0 hardening — are laid out in the design brief.
