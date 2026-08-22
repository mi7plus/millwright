//! # Millwright
//!
//! A unified ML framework for Rust — *ten crates, one lifecycle.*
//!
//! Millwright assembles proven Rust crates into one composable ML lifecycle —
//! ingest and profile data, build and tune pipelines, evaluate and explain
//! models, export to ONNX, and serve with drift monitoring — behind one data
//! model, one trait contract, and feature-gated backends.
//!
//! - [`Frame`](frame::Frame) / [`Dataset`](frame::Dataset) — the numeric
//!   boundary type the public API speaks. `Table` (feature `eda`) is the typed,
//!   polars-backed front that lowers into it.
//! - the **core contract** — object-safe [`Transformer`](traits::Transformer),
//!   [`Estimator`](traits::Estimator), [`Predictor`](traits::Predictor), and
//!   [`ProbaPredictor`](traits::ProbaPredictor) — plus specialized
//!   [`Clusterer`](traits::Clusterer), [`Forecaster`](traits::Forecaster),
//!   [`PartialFit`](traits::PartialFit), and [`Balancer`](traits::Balancer) for
//!   the shapes that need them.
//! - [`Pipeline`](pipeline::Pipeline) composition with `"step__param"`
//!   addressing, over feature-gated backends (smartcore, linfa, …).
//!
//! Everything is object-safe, so a pipeline holds a heterogeneous chain of
//! boxed steps and a boxed model. Each capability is a cargo feature; `default`
//! is a lean core and `full` lights up the whole lifecycle. See the
//! [guide](https://millwright-rs.dev/guide.html) for a tour.
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
//!     .estimator("lr", LogisticRegression::new()); // a core, probability-capable model
//!
//! pipe.fit(&train)?;
//! let preds = pipe.predict(&x)?;
//! # let _ = preds;
//! # Ok(())
//! # }
//! ```

pub mod backends;
pub mod error;
pub mod evaluate;
pub mod frame;
pub mod logistic;
pub mod pipeline;
pub mod traits;
pub mod transform;

#[cfg(feature = "anomaly")]
pub mod anomaly;
#[cfg(feature = "automl")]
pub mod automl;
#[cfg(feature = "preprocessing")]
pub mod balance;
#[cfg(feature = "calibration")]
pub mod calibration;
#[cfg(feature = "diagnostics")]
pub mod diagnostics;
#[cfg(feature = "ensemble")]
pub mod ensemble;
#[cfg(feature = "explain")]
pub mod explain;
#[cfg(feature = "monitor")]
pub mod monitor;
#[cfg(feature = "onnx")]
pub mod onnx;
#[cfg(feature = "eda")]
pub mod profile;
#[cfg(feature = "registry")]
pub mod registry;
#[cfg(feature = "model-selection")]
pub mod selection;
#[cfg(feature = "serve")]
pub mod serve;
#[cfg(feature = "eda")]
pub mod table;
#[cfg(feature = "viz")]
pub mod viz;

#[cfg(any(feature = "model-selection", feature = "ensemble", feature = "explain"))]
mod rng;

#[cfg(feature = "python")]
mod python;

pub use error::{Error, Result};

/// The one import that brings the whole framework into scope.
pub mod prelude {
    pub use crate::error::{Error, Result};
    pub use crate::evaluate::{Evaluate, Report, Task};
    pub use crate::frame::{Dataset, Frame};
    pub use crate::logistic::LogisticRegression;
    pub use crate::pipeline::Pipeline;
    pub use crate::traits::{
        Balancer, Clusterer, Estimator, Forecaster, Model, ParamValue, PartialFit, Predictor,
        ProbaPredictor, Transformer,
    };
    pub use crate::transform::{
        ColumnTransformer, ImputeStrategy, MinMaxScaler, OneHotEncoder, PowerTransform,
        SimpleImputer, StandardScaler, TargetEncoder, Winsorize,
    };

    #[cfg(feature = "smartcore-backend")]
    pub use crate::backends::smartcore::{LinearRegression, RandomForest};

    #[cfg(feature = "linfa-backend")]
    pub use crate::backends::linfa::{Dbscan, GaussianMixture, KMeans, Pca};

    #[cfg(feature = "timeseries")]
    pub use crate::backends::chronos::AutoArima;

    #[cfg(feature = "incremental")]
    pub use crate::backends::incremental::IncrementalLinear;

    #[cfg(feature = "preprocessing")]
    pub use crate::balance::{RandomOverSampler, Smote};

    #[cfg(feature = "model-selection")]
    pub use crate::selection::{
        CrossValidator, GridSearch, KFold, Metric, ParamGrid, RandomSearch, SearchResult,
        StratifiedKFold,
    };

    #[cfg(feature = "hpo")]
    pub use crate::selection::{BayesSearch, SearchSpace};

    #[cfg(feature = "diagnostics")]
    pub use crate::diagnostics::Diagnostics;

    #[cfg(feature = "calibration")]
    pub use crate::calibration::{
        reliability_curve, CalibratedClassifier, CalibrationMethod, IsotonicRegression,
        PlattScaling, ReliabilityBin,
    };

    #[cfg(feature = "anomaly")]
    pub use crate::anomaly::{KnnScore, Mahalanobis, OutlierDetector};

    #[cfg(feature = "eda")]
    pub use crate::profile::{Alert, ColumnProfile, Profile, TargetKind, TargetProfile};
    #[cfg(feature = "eda")]
    pub use crate::table::{ColKind, Table};

    #[cfg(feature = "explain")]
    pub use crate::explain::{permutation_importance, Explain, Explainer, Explanation};

    #[cfg(feature = "viz")]
    pub use crate::viz;

    #[cfg(feature = "onnx")]
    pub use crate::onnx::{ExportOnnx, InferenceModel};

    #[cfg(feature = "registry")]
    pub use crate::registry::{Metadata, Registry, Version};

    #[cfg(feature = "monitor")]
    pub use crate::monitor::{DriftMonitor, DriftStatus};

    #[cfg(feature = "serve")]
    pub use crate::serve::Server;

    #[cfg(feature = "ensemble")]
    pub use crate::ensemble::{Bagging, Voting, VotingKind};

    #[cfg(feature = "automl")]
    pub use crate::automl::{AutoML, AutoMLResult, Budget};
    #[cfg(all(feature = "ensemble", feature = "model-selection"))]
    pub use crate::ensemble::Stacking;
}
