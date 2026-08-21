//! The single error type for the framework spine.
//!
//! Every fallible operation in Millwright returns [`Result`], so a whole
//! `fit → transform → predict → Pipeline` chain threads one error type from
//! end to end.

use std::fmt;

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur while building or running the spine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A buffer length or row/column count did not line up.
    Shape(String),
    /// A column name was missing, or two frames disagreed on their schema.
    Schema(String),
    /// A `predict`/`transform` was attempted before `fit`.
    NotFitted(String),
    /// A backend engine (e.g. smartcore) rejected the operation.
    Backend(String),
    /// A parameter path or value was not understood by a step.
    Param(String),
    /// The pipeline itself was assembled in an invalid way.
    Pipeline(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Shape(m) => write!(f, "shape error: {m}"),
            Error::Schema(m) => write!(f, "schema error: {m}"),
            Error::NotFitted(m) => write!(f, "not fitted: {m}"),
            Error::Backend(m) => write!(f, "backend error: {m}"),
            Error::Param(m) => write!(f, "parameter error: {m}"),
            Error::Pipeline(m) => write!(f, "pipeline error: {m}"),
        }
    }
}

impl std::error::Error for Error {}
