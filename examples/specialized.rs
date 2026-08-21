//! Phase 6: the long tail of real workloads — same contract, different shapes.
//!
//! Auto-ARIMA forecasting and out-of-core `partial_fit`. Run with:
//! `cargo run --example specialized --features "timeseries incremental"`

use millwright::prelude::*;

fn main() -> Result<()> {
    // --- time series: auto-ARIMA forecasting ---
    // A trend with a little seasonality-ish wobble.
    let series: Vec<f64> = (0..48)
        .map(|i| 100.0 + i as f64 * 1.5 + ((i % 6) as f64 - 2.5))
        .collect();
    let mut arima = AutoArima::new().max_p(3).max_q(3);
    arima.fit(&series)?;
    let forecast = arima.forecast(6)?;
    println!("last 3 observed : {:?}", &series[series.len() - 3..]);
    println!(
        "6-step forecast : {:?}",
        forecast
            .iter()
            .map(|v| (v * 10.0).round() / 10.0)
            .collect::<Vec<_>>()
    );

    // --- out-of-core: learn a line over streamed batches ---
    // y = 4*x - 1, never holding the whole dataset in memory at once.
    let mut model = IncrementalLinear::with_rate(0.05, 0.0);
    for epoch in 0..300 {
        let base = (epoch % 8) as f64 * 0.5;
        let rows: Vec<Vec<f64>> = (0..8).map(|i| vec![base + i as f64 * 0.1]).collect();
        let y: Vec<f64> = rows.iter().map(|r| 4.0 * r[0] - 1.0).collect();
        let batch = Dataset::new(Frame::from_rows(rows, vec!["x".into()])?, y)?;
        model.partial_fit(&batch)?; // one batch at a time
    }
    let probe = Frame::from_rows(vec![vec![3.0], vec![10.0]], vec!["x".into()])?;
    println!(
        "streamed model  : f(3)={:.2}, f(10)={:.2}  (true 11, 39)",
        model.predict(&probe)?[0],
        model.predict(&probe)?[1]
    );

    println!("ok — the long tail of real workloads.");
    Ok(())
}
