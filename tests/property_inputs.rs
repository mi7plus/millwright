use millwright::frame::Frame;
use proptest::prelude::*;
use std::io::Write;

proptest! {
    #[test]
    fn frame_constructor_never_panics_on_ragged_external_rows(
        rows in prop::collection::vec(
            prop::collection::vec(any::<f64>(), 0..16),
            0..64,
        ),
    ) {
        let width = rows.first().map(Vec::len).unwrap_or(0);
        let columns = (0..width).map(|i| format!("f{i}")).collect();
        let result = Frame::from_rows(rows.clone(), columns);

        let rectangular = rows.iter().all(|row| row.len() == width);
        prop_assert_eq!(result.is_ok(), rectangular);
    }

    #[test]
    fn frame_constructor_rejects_column_count_mismatches(
        width in 0usize..16,
        height in 1usize..64,
        delta in 1usize..8,
    ) {
        let rows = vec![vec![0.0; width]; height];
        let columns = (0..width.saturating_add(delta))
            .map(|i| format!("f{i}"))
            .collect();
        prop_assert!(Frame::from_rows(rows, columns).is_err());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn arbitrary_csv_text_never_panics(text in any::<String>()) {
        let mut file = tempfile::NamedTempFile::new().expect("temporary CSV");
        file.write_all(text.as_bytes()).expect("write temporary CSV");
        let _ = Frame::from_csv(file.path());
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn arbitrary_onnx_bytes_never_panic(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
        use millwright::onnx::InferenceModel;

        let mut file = tempfile::NamedTempFile::new().expect("temporary ONNX");
        file.write_all(&bytes).expect("write temporary ONNX");
        let _ = InferenceModel::load(file.path());
    }

    #[cfg(feature = "registry")]
    #[test]
    fn registry_rejects_path_traversal_names(suffix in "[A-Za-z0-9_-]{1,32}") {
        use millwright::registry::Registry;

        let root = tempfile::tempdir().expect("temporary registry");
        let registry = Registry::local(root.path());
        let malicious = format!("../{suffix}");
        prop_assert!(registry.versions(&malicious).is_err());
        prop_assert!(!root.path().parent().unwrap().join(&suffix).exists());
    }
}
