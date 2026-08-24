//! HTTP inference serving — a `/predict` endpoint over the tract runtime.
//!
//! [`Server::from_onnx`] loads an ONNX model and exposes it as an axum service:
//! `POST` the configured route with `{"rows": [[...]]}` and get back
//! `{"predictions": [...]}`. With a [`DriftMonitor`]
//! attached, every request feeds the monitor and `GET /metrics` reports live PSI
//! drift — a served model that watches its own request stream.
//!
//! ```no_run
//! use millwright::prelude::*;
//! # async fn run() -> millwright::Result<()> {
//! Server::from_onnx("churn.onnx")?
//!     .route("/predict")
//!     .serve("0.0.0.0:8080")
//!     .await
//! # }
//! ```

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::frame::Frame;
use crate::onnx::InferenceModel;

#[cfg(feature = "monitor")]
use crate::monitor::DriftMonitor;

#[derive(Deserialize)]
struct PredictRequest {
    rows: Vec<Vec<f64>>,
}

#[derive(Serialize)]
struct PredictResponse {
    predictions: Vec<f64>,
}

struct AppState {
    model: InferenceModel,
    limits: Limits,
    permits: Arc<tokio::sync::Semaphore>,
    timeout: Duration,
    #[cfg(feature = "monitor")]
    monitor: Option<Arc<DriftMonitor>>,
}

#[derive(Clone, Copy)]
struct Limits {
    rows: usize,
    columns: usize,
    body_bytes: usize,
}

/// An inference server bound to a model and a route.
pub struct Server {
    model: InferenceModel,
    route: String,
    limits: Limits,
    max_concurrency: usize,
    timeout: Duration,
    #[cfg(feature = "monitor")]
    monitor: Option<Arc<DriftMonitor>>,
}

impl Server {
    /// Load an ONNX model to serve.
    pub fn from_onnx(path: impl AsRef<std::path::Path>) -> Result<Server> {
        Ok(Server {
            model: InferenceModel::load(path)?,
            route: "/predict".to_string(),
            limits: Limits {
                rows: 10_000,
                columns: 10_000,
                body_bytes: 8 * 1024 * 1024,
            },
            max_concurrency: 64,
            timeout: Duration::from_secs(30),
            #[cfg(feature = "monitor")]
            monitor: None,
        })
    }

    /// Serve the tagged ONNX artifact for a model straight from a [`Registry`].
    ///
    /// [`Registry`]: crate::registry::Registry
    #[cfg(feature = "registry")]
    pub fn from_registry(
        registry: &crate::registry::Registry,
        name: &str,
        tag: &str,
    ) -> Result<Server> {
        Server::from_onnx(registry.onnx_path(name, tag)?)
    }

    /// Set the prediction route (default `/predict`).
    pub fn route(mut self, path: impl Into<String>) -> Self {
        self.route = path.into();
        self
    }

    /// Set request limits for rows, columns per row, and JSON body bytes.
    pub fn request_limits(mut self, rows: usize, columns: usize, body_bytes: usize) -> Self {
        self.limits = Limits {
            rows: rows.max(1),
            columns: columns.max(1),
            body_bytes: body_bytes.max(1),
        };
        self
    }

    /// Limit concurrent inference requests (default 64).
    pub fn max_concurrency(mut self, requests: usize) -> Self {
        self.max_concurrency = requests.max(1);
        self
    }

    /// Set the maximum time a request waits for inference (default 30 seconds).
    pub fn inference_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Attach a drift monitor; predictions feed it and `GET /metrics` reports it.
    #[cfg(feature = "monitor")]
    pub fn with_monitor(mut self, monitor: DriftMonitor) -> Self {
        self.monitor = Some(Arc::new(monitor));
        self
    }

    /// Build the axum [`Router`] (useful for testing without binding a port).
    pub fn router(self) -> Router {
        let route = self.route.clone();
        let state = Arc::new(AppState {
            model: self.model,
            limits: self.limits,
            permits: Arc::new(tokio::sync::Semaphore::new(self.max_concurrency)),
            timeout: self.timeout,
            #[cfg(feature = "monitor")]
            monitor: self.monitor,
        });
        let router = Router::new()
            .route(&route, post(predict_handler))
            .route("/healthz", get(|| async { StatusCode::OK }))
            .layer(DefaultBodyLimit::max(self.limits.body_bytes));
        #[cfg(feature = "monitor")]
        let router = router.route("/metrics", axum::routing::get(metrics_handler));
        router.with_state(state)
    }

    /// Bind `addr` and serve until the process ends.
    pub async fn serve(self, addr: &str) -> Result<()> {
        self.serve_with_shutdown(addr, std::future::pending()).await
    }

    /// Bind and serve until `shutdown` resolves, then drain active connections.
    pub async fn serve_with_shutdown<F>(self, addr: &str, shutdown: F) -> Result<()>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let router = self.router();
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| Error::Backend(format!("bind {addr}: {e}")))?;
        axum::serve(listener, router)
            .with_graceful_shutdown(shutdown)
            .await
            .map_err(|e| Error::Backend(format!("serve: {e}")))
    }
}

async fn predict_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PredictRequest>,
) -> std::result::Result<Json<PredictResponse>, (StatusCode, String)> {
    if req.rows.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "no rows provided".into()));
    }
    if req.rows.len() > state.limits.rows {
        return Err((StatusCode::PAYLOAD_TOO_LARGE, "too many rows".into()));
    }
    let ncols = req.rows[0].len();
    if ncols > state.limits.columns {
        return Err((StatusCode::PAYLOAD_TOO_LARGE, "too many columns".into()));
    }
    if ncols == 0 || req.rows.iter().any(|r| r.len() != ncols) {
        return Err((
            StatusCode::BAD_REQUEST,
            "rows must be non-empty and rectangular".into(),
        ));
    }
    let columns = (0..ncols).map(|i| format!("f{i}")).collect();
    let frame = Frame::from_rows(req.rows, columns)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let permit = state.permits.clone().acquire_owned().await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "server is shutting down".into(),
        )
    })?;
    let inference_state = Arc::clone(&state);
    let inference = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        inference_state.model.predict(&frame)
    });
    let predictions = tokio::time::timeout(state.timeout, inference)
        .await
        .map_err(|_| (StatusCode::GATEWAY_TIMEOUT, "inference timed out".into()))?
        .map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference worker failed".into(),
            )
        })?
        .map_err(|_| {
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                "model could not process this input".into(),
            )
        })?;

    #[cfg(feature = "monitor")]
    if let Some(monitor) = &state.monitor {
        monitor.observe(&predictions);
    }

    Ok(Json(PredictResponse { predictions }))
}

#[cfg(feature = "monitor")]
async fn metrics_handler(
    State(state): State<Arc<AppState>>,
) -> std::result::Result<Json<serde_json::Value>, (StatusCode, String)> {
    let Some(monitor) = &state.monitor else {
        return Err((StatusCode::NOT_FOUND, "no monitor attached".into()));
    };
    let status = monitor.report().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "monitor unavailable".into(),
        )
    })?;
    Ok(Json(serde_json::json!({
        "drifted": status.drifted,
        "psi": status.psi,
        "observed": status.observed,
    })))
}

#[cfg(all(test, feature = "smartcore-backend"))]
mod tests {
    use super::*;
    use crate::backends::smartcore::LinearRegression;
    use crate::frame::Dataset;
    use crate::onnx::ExportOnnx;
    use crate::traits::Estimator;
    use axum::body::Body;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn arbitrary_request_bytes_never_panic(bytes in prop::collection::vec(any::<u8>(), 0..2048)) {
            let _ = serde_json::from_slice::<PredictRequest>(&bytes);
        }
    }
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    // A per-test filename: tests run in parallel, so a shared path would race
    // (one test deleting the file another is still loading).
    fn linear_onnx(tag: &str) -> std::path::PathBuf {
        // y = 2*x1 + 3*x2 + 1
        let rows: Vec<Vec<f64>> = (0..15).map(|i| vec![i as f64, (i % 4) as f64]).collect();
        let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] + 3.0 * r[1] + 1.0).collect();
        let ds = Dataset::new(
            Frame::from_rows(rows, vec!["x1".into(), "x2".into()]).unwrap(),
            y,
        )
        .unwrap();
        let mut lr = LinearRegression::new();
        lr.fit(&ds).unwrap();
        let path = std::env::temp_dir().join(format!("mw_serve_{}_{tag}.onnx", std::process::id()));
        lr.export_onnx(&path).unwrap();
        path
    }

    #[tokio::test]
    async fn predict_endpoint_returns_predictions() {
        let path = linear_onnx("predict");
        let app = Server::from_onnx(&path).unwrap().router();

        let body =
            serde_json::to_vec(&serde_json::json!({ "rows": [[20.0, 1.0], [5.0, 2.0]] })).unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/predict")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let preds = parsed["predictions"].as_array().unwrap();
        // y = 2*20 + 3*1 + 1 = 44 ; 2*5 + 3*2 + 1 = 17
        assert!((preds[0].as_f64().unwrap() - 44.0).abs() < 1e-2);
        assert!((preds[1].as_f64().unwrap() - 17.0).abs() < 1e-2);
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn ragged_rows_are_rejected() {
        let path = linear_onnx("ragged");
        let app = Server::from_onnx(&path).unwrap().router();
        let body = serde_json::to_vec(&serde_json::json!({ "rows": [[1.0, 2.0], [3.0]] })).unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/predict")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn health_and_request_limits_are_enforced() {
        let path = linear_onnx("limits");
        let app = Server::from_onnx(&path)
            .unwrap()
            .request_limits(1, 2, 1024)
            .router();

        let health = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/healthz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);

        let body = serde_json::to_vec(&serde_json::json!({
            "rows": [[1.0, 2.0], [3.0, 4.0]]
        }))
        .unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/predict")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let _ = std::fs::remove_file(&path);
    }
}
