from pathlib import Path

import pytest

import millwright as mw


def test_version_and_frame_protocol(binary_frame):
    assert mw.version() == "2.2.3"
    assert binary_frame.shape == (4, 1)
    assert binary_frame.columns() == ["x"]
    assert len(binary_frame) == 4
    assert repr(binary_frame) == "Frame(4 rows x 1 cols)"


@pytest.mark.parametrize(
    "rows,columns",
    [
        ([[1.0], [2.0, 3.0]], None),
        ([[1.0]], ["a", "b"]),
    ],
)
def test_invalid_frame_shapes_raise_value_error(rows, columns):
    with pytest.raises(ValueError):
        mw.Frame.from_rows(rows, columns)


def test_table_profile_and_html_escaping(binary_frame, tmp_path: Path):
    table = mw.Table.from_frame(binary_frame)
    assert table.shape == binary_frame.shape
    assert table.to_frame().shape == binary_frame.shape

    report = tmp_path / "profile.html"
    mw.Profile.of(table).to_html(str(report))
    html = report.read_text(encoding="utf-8")
    assert "<html" in html.lower()
    assert report.stat().st_size > 0


def test_missing_csv_is_reported():
    with pytest.raises((ValueError, OSError)):
        mw.Table.from_csv("definitely-not-a-real-millwright-file.csv")
