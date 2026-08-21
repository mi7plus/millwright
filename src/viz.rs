//! Report figures — ROC and residual charts rendered to SVG.
//!
//! Uses [`plotters-statistical`](https://docs.rs/plotters-statistical) series on
//! top of plotters' pure-Rust SVG backend (no system fonts or libraries), so a
//! report figure is one call that writes a self-contained `.svg`.

use std::path::Path;

use plotters::prelude::*;
use plotters_statistical::RocCurve;

use crate::error::{Error, Result};

fn viz_err(e: impl std::fmt::Display) -> Error {
    Error::Backend(format!("viz: {e}"))
}

/// Render a ROC curve (with its AUC) for binary scores to an SVG file.
///
/// `y_true` is `0.0` / `1.0` ground truth; `y_score` the positive-class score
/// (probability or ranking). Returns the AUC.
pub fn roc_svg(
    y_true: &[f64],
    y_score: &[f64],
    path: impl AsRef<Path>,
    size: (u32, u32),
) -> Result<f64> {
    if y_true.len() != y_score.len() {
        return Err(Error::Shape(
            "roc_svg: y_true and y_score differ in length".into(),
        ));
    }
    let labels: Vec<bool> = y_true.iter().map(|v| *v >= 0.5).collect();
    let roc = RocCurve::from_scores(y_score, &labels).map_err(viz_err)?;
    let auc = roc.auc();

    let root = SVGBackend::new(path.as_ref(), size).into_drawing_area();
    root.fill(&WHITE).map_err(viz_err)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(format!("ROC (AUC = {auc:.3})"), ("sans-serif", 18))
        .margin(12)
        .x_label_area_size(32)
        .y_label_area_size(36)
        .build_cartesian_2d(0.0f64..1.0, 0.0f64..1.0)
        .map_err(viz_err)?;
    chart
        .configure_mesh()
        .x_desc("false positive rate")
        .y_desc("true positive rate")
        .draw()
        .map_err(viz_err)?;
    // chance baseline, then the ROC curve
    chart
        .draw_series(LineSeries::new(vec![(0.0, 0.0), (1.0, 1.0)], RED.mix(0.4)))
        .map_err(viz_err)?;
    chart.draw_series(std::iter::once(roc)).map_err(viz_err)?;
    root.present().map_err(viz_err)?;
    Ok(auc)
}

/// Render a residuals-vs-fitted scatter (with a zero line) to an SVG file — the
/// standard regression diagnostic plot.
pub fn residuals_svg(
    y_true: &[f64],
    y_pred: &[f64],
    path: impl AsRef<Path>,
    size: (u32, u32),
) -> Result<()> {
    if y_true.len() != y_pred.len() {
        return Err(Error::Shape("residuals_svg: length mismatch".into()));
    }
    let points: Vec<(f64, f64)> = y_pred
        .iter()
        .zip(y_true)
        .map(|(p, t)| (*p, t - p))
        .collect();
    let (mut xmin, mut xmax) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut rmin, mut rmax) = (f64::INFINITY, f64::NEG_INFINITY);
    for (x, r) in &points {
        xmin = xmin.min(*x);
        xmax = xmax.max(*x);
        rmin = rmin.min(*r);
        rmax = rmax.max(*r);
    }
    // pad the ranges so points aren't on the border
    let xpad = ((xmax - xmin) * 0.05).max(1e-6);
    let rpad = ((rmax - rmin) * 0.15).max(1e-6);

    let root = SVGBackend::new(path.as_ref(), size).into_drawing_area();
    root.fill(&WHITE).map_err(viz_err)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Residuals vs fitted", ("sans-serif", 18))
        .margin(12)
        .x_label_area_size(32)
        .y_label_area_size(40)
        .build_cartesian_2d(xmin - xpad..xmax + xpad, rmin - rpad..rmax + rpad)
        .map_err(viz_err)?;
    chart
        .configure_mesh()
        .x_desc("fitted")
        .y_desc("residual")
        .draw()
        .map_err(viz_err)?;
    chart
        .draw_series(LineSeries::new(
            vec![(xmin - xpad, 0.0), (xmax + xpad, 0.0)],
            RED.mix(0.5),
        ))
        .map_err(viz_err)?;
    chart
        .draw_series(
            points
                .iter()
                .map(|(x, r)| Circle::new((*x, *r), 3, BLUE.filled())),
        )
        .map_err(viz_err)?;
    root.present().map_err(viz_err)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("millwright_viz_{name}.svg"));
        p
    }

    #[test]
    fn roc_svg_writes_a_chart_and_reports_auc() {
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_score = vec![0.1, 0.35, 0.6, 0.9]; // perfectly separable
        let path = scratch("roc");
        let auc = roc_svg(&y_true, &y_score, &path, (400, 400)).unwrap();
        assert!((auc - 1.0).abs() < 1e-9, "auc = {auc}");
        let svg = std::fs::read_to_string(&path).unwrap();
        assert!(svg.contains("<svg") && svg.contains("</svg>"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn residuals_svg_writes_a_chart() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![1.1, 1.9, 3.2, 3.8, 5.1];
        let path = scratch("resid");
        residuals_svg(&y_true, &y_pred, &path, (400, 400)).unwrap();
        let svg = std::fs::read_to_string(&path).unwrap();
        assert!(svg.contains("<svg"));
        let _ = std::fs::remove_file(&path);
    }
}
