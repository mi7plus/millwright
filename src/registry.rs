//! A versioned model registry — the loop scikit-learn leaves as homework.
//!
//! [`Registry::local`] versions a trained model's ONNX artifact on disk,
//! content-addressed so identical models dedupe. Each version carries metadata
//! (metrics, a note) and the reference prediction distribution the drift monitor
//! watches against. Tags like `prod` are movable pointers; [`Registry::rollback`]
//! reverts a tag to the previous version.
//!
//! ```no_run
//! use millwright::prelude::*;
//! # fn main() -> millwright::Result<()> {
//! # let model: RandomForest = todo!();
//! let reg = Registry::local("./models");
//! let v = reg.register("churn", &model, Metadata::default())?;
//! reg.tag("churn", &v.id, "prod")?;
//! // later, revert prod to the previous version:
//! reg.rollback("churn", "prod")?;
//! # Ok(())
//! # }
//! ```

use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::onnx::ExportOnnx;

fn io_err(e: impl std::fmt::Display) -> Error {
    Error::Backend(format!("registry io: {e}"))
}

/// FNV-1a 64-bit content hash, hex-encoded — enough to dedupe identical
/// artifacts without pulling in a crypto dependency.
fn content_id(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// Metadata stored with a registered version.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Metadata {
    /// Held-out / CV metrics that travel with the version.
    pub metrics: Vec<(String, f64)>,
    /// The training prediction distribution the drift monitor watches against.
    pub reference: Vec<f64>,
    /// A free-form note (git commit, data hash, …).
    pub note: String,
}

/// A registered model version.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Version {
    /// The model name.
    pub name: String,
    /// The content-addressed version id.
    pub id: String,
    /// The version's metadata.
    pub metadata: Metadata,
}

/// A local, file-backed model registry rooted at a directory.
pub struct Registry {
    root: PathBuf,
}

impl Registry {
    /// Open (or create on first write) a registry under `path`.
    pub fn local(path: impl Into<PathBuf>) -> Self {
        Registry { root: path.into() }
    }

    fn name_dir(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }
    fn log_path(&self, name: &str) -> PathBuf {
        self.name_dir(name).join("log.json")
    }
    fn tag_path(&self, name: &str, tag: &str) -> PathBuf {
        self.name_dir(name).join("tags").join(tag)
    }

    /// Register a model under `name`: export its ONNX, content-address it, and
    /// store the artifact plus metadata. Re-registering an identical model
    /// returns the same id (dedupe).
    pub fn register(
        &self,
        name: &str,
        model: &impl ExportOnnx,
        metadata: Metadata,
    ) -> Result<Version> {
        let proto = model.to_onnx()?;
        let bytes = onnx_export_rs::to_bytes(&proto)
            .map_err(|e| Error::Backend(format!("ONNX serialize: {e}")))?;
        let id = content_id(&bytes);

        let dir = self.name_dir(name);
        fs::create_dir_all(&dir).map_err(io_err)?;
        fs::write(dir.join(format!("{id}.onnx")), &bytes).map_err(io_err)?;

        let version = Version {
            name: name.to_string(),
            id: id.clone(),
            metadata,
        };
        fs::write(
            dir.join(format!("{id}.json")),
            serde_json::to_vec_pretty(&version).map_err(io_err)?,
        )
        .map_err(io_err)?;

        let mut log = self.versions(name)?;
        if !log.contains(&id) {
            log.push(id.clone());
            fs::write(
                self.log_path(name),
                serde_json::to_vec(&log).map_err(io_err)?,
            )
            .map_err(io_err)?;
        }
        Ok(version)
    }

    /// Point a tag (e.g. `prod`) at a version id.
    pub fn tag(&self, name: &str, id: &str, tag: &str) -> Result<()> {
        let tag_dir = self.name_dir(name).join("tags");
        fs::create_dir_all(&tag_dir).map_err(io_err)?;
        fs::write(tag_dir.join(tag), id).map_err(io_err)?;
        Ok(())
    }

    /// Resolve a tag or id to a concrete version id.
    pub fn resolve(&self, name: &str, reference: &str) -> Result<String> {
        let tag_path = self.tag_path(name, reference);
        if tag_path.exists() {
            return Ok(fs::read_to_string(tag_path)
                .map_err(io_err)?
                .trim()
                .to_string());
        }
        if self
            .name_dir(name)
            .join(format!("{reference}.onnx"))
            .exists()
        {
            return Ok(reference.to_string());
        }
        Err(Error::Backend(format!(
            "no version or tag '{reference}' for model '{name}'"
        )))
    }

    /// Fetch a version's metadata by tag or id.
    pub fn get(&self, name: &str, reference: &str) -> Result<Version> {
        let id = self.resolve(name, reference)?;
        let json = fs::read(self.name_dir(name).join(format!("{id}.json"))).map_err(io_err)?;
        serde_json::from_slice(&json).map_err(io_err)
    }

    /// The on-disk path of a version's ONNX artifact (for serving).
    pub fn onnx_path(&self, name: &str, reference: &str) -> Result<PathBuf> {
        let id = self.resolve(name, reference)?;
        Ok(self.name_dir(name).join(format!("{id}.onnx")))
    }

    /// Every version id for `name`, oldest first.
    pub fn versions(&self, name: &str) -> Result<Vec<String>> {
        let p = self.log_path(name);
        if !p.exists() {
            return Ok(Vec::new());
        }
        serde_json::from_slice(&fs::read(p).map_err(io_err)?).map_err(io_err)
    }

    /// Revert a tag to the version registered immediately before its current
    /// target, returning the id it now points at.
    pub fn rollback(&self, name: &str, tag: &str) -> Result<String> {
        let current = self.resolve(name, tag)?;
        let log = self.versions(name)?;
        let idx = log
            .iter()
            .position(|v| v == &current)
            .ok_or_else(|| Error::Backend("tag target is not in the version log".into()))?;
        if idx == 0 {
            return Err(Error::Backend("no earlier version to roll back to".into()));
        }
        let prev = log[idx - 1].clone();
        self.tag(name, &prev, tag)?;
        Ok(prev)
    }
}

#[cfg(all(test, feature = "smartcore-backend"))]
mod tests {
    use super::*;
    use crate::backends::smartcore::RandomForest;
    use crate::frame::{Dataset, Frame};
    use crate::traits::Estimator;

    fn temp_root(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("mw_registry_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        p
    }

    fn fit_rf(seed_trees: u16) -> RandomForest {
        let mut rows = Vec::new();
        let mut y = Vec::new();
        for i in 0..12 {
            rows.push(vec![i as f64 * 0.1, 0.0]);
            y.push(0.0);
            rows.push(vec![9.0 + i as f64 * 0.1, 1.0]);
            y.push(1.0);
        }
        let ds = Dataset::new(
            Frame::from_rows(rows, vec!["a".into(), "b".into()]).unwrap(),
            y,
        )
        .unwrap();
        let mut rf = RandomForest::new().n_trees(seed_trees).max_depth(3);
        rf.fit(&ds).unwrap();
        rf
    }

    #[test]
    fn register_tag_rollback_roundtrip() {
        let root = temp_root("roundtrip");
        let reg = Registry::local(&root);

        let v1 = reg
            .register("churn", &fit_rf(10), Metadata::default())
            .unwrap();
        let v2 = reg
            .register(
                "churn",
                &fit_rf(20),
                Metadata {
                    metrics: vec![("f1".into(), 0.97)],
                    ..Default::default()
                },
            )
            .unwrap();
        assert_ne!(v1.id, v2.id);
        assert_eq!(reg.versions("churn").unwrap().len(), 2);

        reg.tag("churn", &v2.id, "prod").unwrap();
        assert_eq!(reg.resolve("churn", "prod").unwrap(), v2.id);
        assert_eq!(
            reg.get("churn", "prod").unwrap().metadata.metrics[0].1,
            0.97
        );
        assert!(reg.onnx_path("churn", "prod").unwrap().exists());

        // roll prod back to v1
        let reverted = reg.rollback("churn", "prod").unwrap();
        assert_eq!(reverted, v1.id);
        assert_eq!(reg.resolve("churn", "prod").unwrap(), v1.id);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn identical_models_dedupe() {
        let root = temp_root("dedupe");
        let reg = Registry::local(&root);
        let rf = fit_rf(10);
        let a = reg.register("m", &rf, Metadata::default()).unwrap();
        let b = reg.register("m", &rf, Metadata::default()).unwrap();
        assert_eq!(a.id, b.id);
        assert_eq!(reg.versions("m").unwrap().len(), 1);
        let _ = fs::remove_dir_all(&root);
    }
}
