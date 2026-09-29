"""The dashboard must start cleanly on the bundled data, offline."""

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_without_errors():
    app = AppTest.from_file("../app.py", default_timeout=120)
    app.run()
    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert "Winner's Sharpe" in labels
    assert "Overfitting probability" in labels


def test_dashboard_reports_the_case_study():
    app = AppTest.from_file("../app.py", default_timeout=120)
    app.run()
    values = {metric.label: metric.value for metric in app.metric}
    assert values["Winner's Sharpe"] == "0.41"
    assert values["Overfitting probability"] == "89.8%"
