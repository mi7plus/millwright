//! Cross-validation splitters and the `cross_val_score` primitive.

use ndarray::Array1;

use model_selection_rs::splitters::{
    CvSplitter, KFold as MsKFold, StratifiedKFold as MsStratifiedKFold,
};

use crate::error::{Error, Result};
use crate::frame::Dataset;
use crate::traits::Model;

use super::scoring::Metric;

fn ms_err(e: impl std::fmt::Display) -> Error {
    Error::Backend(format!("model-selection: {e}"))
}

/// A cross-validation strategy that yields `(train, test)` row-index splits for
/// a dataset. Wraps the `model-selection-rs` splitters, supplying labels
/// automatically where a splitter needs them.
pub trait CrossValidator: CrossValidatorClone {
    /// Produce the `(train_indices, test_indices)` splits for `dataset`.
    fn splits(&self, dataset: &Dataset) -> Result<Vec<(Vec<usize>, Vec<usize>)>>;
    /// The number of splits.
    fn n_splits(&self) -> usize;
}

/// Clone support for boxed cross-validators (so a stacking ensemble that owns
/// one can itself be `Clone`, and thus a [`Model`]).
pub trait CrossValidatorClone {
    /// Clone into a fresh box.
    fn clone_box(&self) -> Box<dyn CrossValidator>;
}

impl<T> CrossValidatorClone for T
where
    T: CrossValidator + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn CrossValidator> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn CrossValidator> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Plain K-fold cross-validation.
#[derive(Clone, Copy, Debug)]
pub struct KFold {
    k: usize,
}

impl KFold {
    /// K-fold with `k` folds.
    pub fn new(k: usize) -> Self {
        KFold { k }
    }
}

impl CrossValidator for KFold {
    fn splits(&self, dataset: &Dataset) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        MsKFold::new(self.k)
            .map_err(ms_err)?
            .split(dataset.features().nrows())
            .map_err(ms_err)
    }
    fn n_splits(&self) -> usize {
        self.k
    }
}

/// Stratified K-fold — preserves per-class proportions in every fold.
///
/// Labels come from the dataset's target automatically (integral-coded), so
/// this reads exactly like the design brief: `StratifiedKFold::new(5)`.
#[derive(Clone, Copy, Debug)]
pub struct StratifiedKFold {
    k: usize,
}

impl StratifiedKFold {
    /// Stratified K-fold with `k` folds.
    pub fn new(k: usize) -> Self {
        StratifiedKFold { k }
    }
}

impl CrossValidator for StratifiedKFold {
    fn splits(&self, dataset: &Dataset) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let labels: Array1<i64> = Array1::from(
            dataset
                .target()
                .iter()
                .map(|v| v.round() as i64)
                .collect::<Vec<_>>(),
        );
        MsStratifiedKFold::new(self.k, &labels)
            .map_err(ms_err)?
            .split(labels.len())
            .map_err(ms_err)
    }
    fn n_splits(&self) -> usize {
        self.k
    }
}

/// Cross-validate one already-configured model, returning the mean fold score.
///
/// A fresh clone of `model` is fit on each fold's training rows and scored on
/// its test rows, so the passed-in model is left untouched.
pub fn cross_val_score(
    model: &dyn Model,
    dataset: &Dataset,
    cv: &dyn CrossValidator,
    metric: Metric,
) -> Result<f64> {
    let splits = cv.splits(dataset)?;
    if splits.is_empty() {
        return Err(Error::Pipeline(
            "cross-validation produced no splits".into(),
        ));
    }
    let mut total = 0.0;
    for (train, test) in &splits {
        let mut m = model.clone_box();
        m.fit(&dataset.select(train))?;
        let preds = m.predict(&dataset.features().select_rows(test))?;
        let truth: Vec<f64> = test.iter().map(|&i| dataset.target()[i]).collect();
        total += metric.score(&truth, &preds);
    }
    Ok(total / splits.len() as f64)
}
