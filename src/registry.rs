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

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::onnx::ExportOnnx;

fn io_err(e: impl std::fmt::Display) -> Error {
    Error::Backend(format!("registry io: {e}"))
}

const SCHEMA_VERSION: u32 = 1;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn validate_identifier(kind: &str, value: &str) -> Result<()> {
    let valid = !value.is_empty()
        && value != "."
        && value != ".."
        && value.len() <= 128
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'));
    if valid {
        Ok(())
    } else {
        Err(Error::Backend(format!(
            "invalid registry {kind} '{value}': use 1-128 ASCII letters, digits, '.', '-' or '_'"
        )))
    }
}

fn validate_id(id: &str) -> Result<()> {
    // Sixteen characters is the legacy v0 registry FNV id. It remains readable
    // so existing registries can be tagged and migrated; all new ids are SHA-256.
    if matches!(id.len(), 16 | 64) && id.bytes().all(|b| b.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(Error::Backend("invalid registry version id".into()))
    }
}

fn content_id(bytes: &[u8], metadata: &Metadata) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update(b"millwright-registry\0");
    hash.update(SCHEMA_VERSION.to_le_bytes());
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
    hash.update(serde_json::to_vec(metadata).map_err(io_err)?);
    Ok(format!("{:x}", hash.finalize()))
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::Backend("registry path has no parent".into()))?;
    fs::create_dir_all(parent).map_err(io_err)?;
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temp = parent.join(format!(".millwright-{}-{sequence}.tmp", std::process::id()));
    let mut file = File::create(&temp).map_err(io_err)?;
    file.write_all(bytes).map_err(io_err)?;
    file.sync_all().map_err(io_err)?;
    if path.exists() {
        fs::remove_file(path).map_err(io_err)?;
    }
    if let Err(error) = fs::rename(&temp, path) {
        let _ = fs::remove_file(&temp);
        return Err(io_err(error));
    }
    Ok(())
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

#[derive(Serialize, Deserialize)]
struct StoredVersion {
    #[serde(default = "current_schema_version")]
    schema_version: u32,
    #[serde(flatten)]
    version: Version,
}

fn current_schema_version() -> u32 {
    SCHEMA_VERSION
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

    fn name_dir(&self, name: &str) -> Result<PathBuf> {
        validate_identifier("model name", name)?;
        Ok(self.root.join(name))
    }
    fn log_path(&self, name: &str) -> Result<PathBuf> {
        Ok(self.name_dir(name)?.join("log.json"))
    }
    fn tag_path(&self, name: &str, tag: &str) -> Result<PathBuf> {
        validate_identifier("tag", tag)?;
        Ok(self.name_dir(name)?.join("tags").join(tag))
    }

    fn lock(&self) -> Result<File> {
        fs::create_dir_all(&self.root).map_err(io_err)?;
        let lock = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(self.root.join(".registry.lock"))
            .map_err(io_err)?;
        lock.lock_exclusive().map_err(io_err)?;
        Ok(lock)
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
        validate_identifier("model name", name)?;
        let id = content_id(&bytes, &metadata)?;
        let _lock = self.lock()?;

        let dir = self.name_dir(name)?;
        fs::create_dir_all(&dir).map_err(io_err)?;
        let artifact_path = dir.join(format!("{id}.onnx"));
        if !artifact_path.exists() {
            atomic_write(&artifact_path, &bytes)?;
        }

        let version = Version {
            name: name.to_string(),
            id: id.clone(),
            metadata,
        };
        let metadata_path = dir.join(format!("{id}.json"));
        if !metadata_path.exists() {
            atomic_write(
                &metadata_path,
                &serde_json::to_vec_pretty(&StoredVersion {
                    schema_version: SCHEMA_VERSION,
                    version: version.clone(),
                })
                .map_err(io_err)?,
            )?;
        }

        let mut log = self.versions(name)?;
        if !log.contains(&id) {
            log.push(id.clone());
            atomic_write(
                &self.log_path(name)?,
                &serde_json::to_vec(&log).map_err(io_err)?,
            )?;
        }
        Ok(version)
    }

    /// Point a tag (e.g. `prod`) at a version id.
    pub fn tag(&self, name: &str, id: &str, tag: &str) -> Result<()> {
        validate_id(id)?;
        validate_identifier("tag", tag)?;
        let _lock = self.lock()?;
        let name_dir = self.name_dir(name)?;
        if !name_dir.join(format!("{id}.onnx")).is_file()
            || !name_dir.join(format!("{id}.json")).is_file()
        {
            return Err(Error::Backend(format!(
                "cannot tag missing registry version '{id}'"
            )));
        }
        let tag_dir = name_dir.join("tags");
        fs::create_dir_all(&tag_dir).map_err(io_err)?;
        atomic_write(&tag_dir.join(tag), id.as_bytes())?;
        Ok(())
    }

    /// Resolve a tag or id to a concrete version id.
    pub fn resolve(&self, name: &str, reference: &str) -> Result<String> {
        validate_identifier("reference", reference)?;
        let tag_path = self.tag_path(name, reference)?;
        if tag_path.exists() {
            let id = fs::read_to_string(tag_path)
                .map_err(io_err)?
                .trim()
                .to_string();
            validate_id(&id)?;
            if !self.name_dir(name)?.join(format!("{id}.onnx")).is_file() {
                return Err(Error::Backend(
                    "registry tag points to a missing artifact".into(),
                ));
            }
            return Ok(id);
        }
        validate_id(reference)?;
        if self
            .name_dir(name)?
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
        let json = fs::read(self.name_dir(name)?.join(format!("{id}.json"))).map_err(io_err)?;
        let stored: StoredVersion = serde_json::from_slice(&json).map_err(io_err)?;
        if stored.schema_version > SCHEMA_VERSION {
            return Err(Error::Backend(format!(
                "registry schema {} is newer than supported schema {SCHEMA_VERSION}",
                stored.schema_version
            )));
        }
        Ok(stored.version)
    }

    /// The on-disk path of a version's ONNX artifact (for serving).
    pub fn onnx_path(&self, name: &str, reference: &str) -> Result<PathBuf> {
        let id = self.resolve(name, reference)?;
        Ok(self.name_dir(name)?.join(format!("{id}.onnx")))
    }

    /// Every version id for `name`, oldest first.
    pub fn versions(&self, name: &str) -> Result<Vec<String>> {
        let p = self.log_path(name)?;
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

    #[cfg(all(feature = "serve", feature = "monitor"))]
    #[test]
    fn serve_and_monitor_from_registry() {
        use crate::backends::smartcore::LinearRegression;
        // A linear model exports to an ONNX graph tract can actually run.
        let rows: Vec<Vec<f64>> = (0..12).map(|i| vec![i as f64, (i % 3) as f64]).collect();
        let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] + 1.0).collect();
        let ds = Dataset::new(
            Frame::from_rows(rows, vec!["x1".into(), "x2".into()]).unwrap(),
            y,
        )
        .unwrap();
        let mut lr = LinearRegression::new();
        lr.fit(&ds).unwrap();

        let root = temp_root("from_registry");
        let reg = Registry::local(&root);
        let v = reg
            .register(
                "demand",
                &lr,
                Metadata {
                    reference: vec![1.0, 2.0, 3.0, 2.0, 1.0, 3.0],
                    ..Default::default()
                },
            )
            .unwrap();
        reg.tag("demand", &v.id, "prod").unwrap();

        // serve the prod artifact straight from the registry...
        let _server = crate::serve::Server::from_registry(&reg, "demand", "prod").unwrap();
        // ...and build a PSI monitor from the version's stored reference.
        let _monitor = crate::monitor::DriftMonitor::from_registry(&v).unwrap();

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

    #[test]
    fn registry_rejects_path_traversal_and_missing_tag_targets() {
        let root = temp_root("validation");
        let reg = Registry::local(&root);
        let rf = fit_rf(10);
        assert!(reg.register("../escape", &rf, Metadata::default()).is_err());
        let version = reg.register("safe", &rf, Metadata::default()).unwrap();
        assert!(reg.tag("safe", &version.id, "../prod").is_err());
        assert!(reg.tag("safe", &"0".repeat(64), "prod").is_err());
        assert!(reg.resolve("safe", "../outside").is_err());
        assert!(!root.parent().unwrap().join("escape").exists());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn metadata_is_part_of_the_immutable_version_identity() {
        let root = temp_root("metadata_identity");
        let reg = Registry::local(&root);
        let rf = fit_rf(10);
        let first = reg.register("m", &rf, Metadata::default()).unwrap();
        let second = reg
            .register(
                "m",
                &rf,
                Metadata {
                    note: "new training context".into(),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_ne!(first.id, second.id);
        assert!(reg.get("m", &first.id).unwrap().metadata.note.is_empty());
        assert_eq!(
            reg.get("m", &second.id).unwrap().metadata.note,
            "new training context"
        );
        let _ = fs::remove_dir_all(&root);
    }
}
