from __future__ import annotations

from pathlib import Path

import pytest


def test_streamlit_dashboard_smoke() -> None:
    """Render the dashboard once without triggering model inference/downloads."""
    pytest.importorskip("streamlit")
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("This Streamlit version does not provide AppTest.")
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    assert any("Computer Vision-Based Traffic Load" in str(element.value) for element in app.title)
