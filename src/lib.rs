//! # Millwright
//!
//! A unified ML framework for Rust — *ten crates, one lifecycle.*
//!
//! This is **Phase 0 · the spine**: the smallest thing that proves the design.
//! It ships four pieces that everything later builds on:
//!
//! - [`Frame`](frame::Frame) / [`Dataset`](frame::Dataset) — the boundary data
//!   model the public API speaks.
//! - the four traits — [`Transformer`](traits::Transformer),
//!   [`Estimator`](traits::Estimator), [`Predictor`](traits::Predictor),
//!   [`ProbaPredictor`](traits::ProbaPredictor).
//! - the first backend — a [`smartcore`](backends::smartcore) adapter.
//! - [`Pipeline`](pipeline::Pipeline) composition, with `"step__param"`
//!   addressing.
//!
//! Everything is object-safe, so a pipeline holds a heterogeneous chain of
//! boxed steps and a boxed model.
//!
//! ```no_run
//! use millwright::prelude::*;
//!
//! # fn main() -> millwright::Result<()> {
//! let x = Frame::from_rows(
//!     vec![vec![0.0, 0.0], vec![9.0, 9.0]],
//!     vec!["a".into(), "b".into()],
//! )?;
//! let train = Dataset::new(x.clone(), vec![0.0, 1.0])?;
//!
//! let mut pipe = Pipeline::new()
//!     .step("scale", StandardScaler::new())
//!     .estimator("rf", RandomForest::new());
//!
//! pipe.fit(&train)?;
//! let preds = pipe.predict(&x)?;
//! # let _ = preds;
//! # Ok(())
//! # }
//! ```

pub mod backends;
pub mod error;
pub mod frame;
pub mod pipeline;
pub mod traits;
pub mod transform;

pub use error::{Error, Result};

/// The one import that brings the whole spine into scope.
pub mod prelude {
    pub use crate::error::{Error, Result};
    pub use crate::frame::{Dataset, Frame};
    pub use crate::pipeline::Pipeline;
    pub use crate::traits::{Estimator, Model, ParamValue, Predictor, ProbaPredictor, Transformer};
    pub use crate::transform::StandardScaler;

    #[cfg(feature = "smartcore-backend")]
    pub use crate::backends::smartcore::{LinearRegression, RandomForest};
}
