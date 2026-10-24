"""Helper to create a local SparkSession (single-node, sufficient for 150K-250K row datasets)."""
import logging
import os
import sys

# --- Windows PySpark environment ------------------------------------------------------------
# PySpark's .cmd launch scripts break on paths containing spaces ("C:\Program Files\..."), so on
# Windows JAVA_HOME must be an 8.3 short path. The JDK location is auto-detected instead of being
# hardcoded to one machine's exact JDK version (which broke Spark on every other PC).
_JDK_ROOTS_WINDOWS = [
    r"C:\Program Files\Microsoft",
    r"C:\Program Files\Eclipse Adoptium",
    r"C:\Program Files\Java",
    r"C:\Program Files\Amazon Corretto",
    r"C:\Program Files\Zulu",
]


def find_windows_jdk(roots: list[str] | None = None) -> str | None:
    """Newest JDK 11/17/21 directory under the usual install roots (names like jdk-17.0.2-hotspot)."""
    import glob
    import re

    candidates = []
    for root in roots or _JDK_ROOTS_WINDOWS:
        for path in glob.glob(os.path.join(root, "*")):
            m = re.search(r"(?:jdk|jre)[-_]?(\d+)(?:\.(\d+))?(?:\.(\d+))?", os.path.basename(path), re.I)
            if m and int(m.group(1)) in (11, 17, 21) and os.path.isdir(path):
                candidates.append((tuple(int(g or 0) for g in m.groups()), path))
    return max(candidates)[1] if candidates else None


def _to_short_path(path: str) -> str:
    """8.3 short path (no spaces). Returns the input unchanged if it cannot be converted."""
    try:
        import ctypes

        buf = ctypes.create_unicode_buffer(260)
        n = ctypes.windll.kernel32.GetShortPathNameW(path, buf, 260)  # type: ignore[attr-defined]
        return buf.value if n else path
    except Exception:
        return path


if sys.platform == "win32":
    java_home = os.environ.get("JAVA_HOME") or find_windows_jdk()
    if java_home:
        os.environ["JAVA_HOME"] = _to_short_path(java_home) if " " in java_home else java_home
    else:
        logging.getLogger(__name__).warning(
            "JAVA_HOME is not set and no JDK 11/17/21 was found. Install one (e.g. Temurin 17) "
            "and set JAVA_HOME, otherwise Spark will not start."
        )
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def get_spark_session(app_name: str = "credit-risk-pipeline") -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "Asia/Singapore")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        # --- FIX UNTUK PYTEST HANGING DI WINDOWS ---
        .config("spark.ui.enabled", "false")                 # Matikan Web UI agar tidak mengunci port
        .config("spark.ui.showConsoleProgress", "false")     # Matikan progress bar yang menyumbat stdout pytest
        .config("spark.python.worker.reuse", "true")         # Reuse worker process
        .config("spark.driver.extraJavaOptions", "-Djava.net.preferIPv4Stack=true")
        .config("spark.executor.extraJavaOptions", "-Djava.net.preferIPv4Stack=true")
        .master("local[2]")                                  # Alokasikan 2 core agar stabil saat pytest
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")                  # Kurangi log noise saat pengujian
    logger.info(f"SparkSession '{app_name}' ready. Spark version: {spark.version}")
    return spark