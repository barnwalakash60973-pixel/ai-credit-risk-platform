"""JDK auto-detection used on Windows (pure filesystem logic, so it is testable on any OS)."""
from src.python.transformers.spark_session import find_windows_jdk


def test_picks_newest_supported_jdk(tmp_path):
    for name in ["jdk-17.0.2-hotspot", "jdk-17.0.20.101-hotspot", "jdk-21.0.1", "jdk-8.0.0", "readme"]:
        (tmp_path / name).mkdir()
    assert find_windows_jdk([str(tmp_path)]).endswith("jdk-21.0.1")


def test_ignores_unsupported_versions_and_returns_none(tmp_path):
    (tmp_path / "jdk-8.0.352").mkdir()
    assert find_windows_jdk([str(tmp_path)]) is None


def test_returns_none_when_roots_missing():
    assert find_windows_jdk(["/definitely/not/here"]) is None
