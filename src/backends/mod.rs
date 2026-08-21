//! Backend adapters — proven engines behind the four traits.
//!
//! Each adapter converts a [`Frame`](crate::frame::Frame) to its engine's
//! native array type *inside the adapter only*, so the version war between
//! array crates never reaches user code. Phase 0 ships the first backend,
//! [`smartcore`](self::smartcore); later phases add linfa, chronos-ts, and a
//! tract-based ONNX runner behind their own feature gates.

#[cfg(feature = "smartcore-backend")]
pub mod smartcore;
