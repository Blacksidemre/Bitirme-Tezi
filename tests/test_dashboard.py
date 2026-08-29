from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_loads_without_runtime_exception() -> None:
    project_root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(project_root / "dashboard" / "app.py"))

    app.run(timeout=30)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Piyasa Görünümü",
        "Model Laboratuvarı",
        "Fiyat Simülatörü",
        "Veri Kalitesi",
    ]
    assert len(app.metric) >= 8
    assert len(app.get("plotly_chart")) >= 8
