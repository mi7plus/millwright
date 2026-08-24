use millwright::frame::Frame;
use proptest::prelude::*;

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
        height in 0usize..64,
        delta in 1usize..8,
    ) {
        let rows = vec![vec![0.0; width]; height];
        let columns = (0..width.saturating_add(delta))
            .map(|i| format!("f{i}"))
            .collect();
        prop_assert!(Frame::from_rows(rows, columns).is_err());
    }
}
