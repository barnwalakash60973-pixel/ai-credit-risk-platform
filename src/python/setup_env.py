"""
One-command first-time setup + health check.

    python -m src.python.setup_env

 1. creates .env from .env.example if it does not exist
 2. generates PII_HASH_KEY and PII_ENCRYPTION_KEY if they are blank (bronze ingest crashes without them)
 3. tells you which database will be used (PostgreSQL / SQLite)
 4. checks Java (PySpark), the Kaggle credentials / dataset file, and Ollama (chatbot)
Safe to run repeatedly: it never overwrites a value you already set.
"""
import re
import secrets
import shutil
import subprocess
from pathlib import Path

from cryptography.fernet import Fernet

ROOT = Path(__file__).resolve().parents[2]
ENV, EXAMPLE = ROOT / ".env", ROOT / ".env.example"


def _read_env() -> dict[str, str]:
    out = {}
    for line in ENV.read_text().splitlines():
        m = re.match(r"^([A-Z0-9_]+)=(.*)$", line.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def _set(key: str, value: str) -> None:
    text = ENV.read_text()
    if re.search(rf"^{key}=", text, re.M):
        text = re.sub(rf"^{key}=.*$", f"{key}={value}", text, flags=re.M)
    else:
        text += f"\n{key}={value}\n"
    ENV.write_text(text)


def main() -> None:
    ok = lambda m: print(f"  [ OK ] {m}")
    warn = lambda m: print(f"  [FIX ] {m}")

    print("\n== .env ==")
    if not ENV.exists():
        shutil.copy(EXAMPLE, ENV)
        ok("created .env from .env.example")
    else:
        ok(".env already exists")
    env = _read_env()

    if not env.get("PII_HASH_KEY"):
        _set("PII_HASH_KEY", secrets.token_hex(32))
        ok("generated PII_HASH_KEY")
    if not env.get("PII_ENCRYPTION_KEY"):
        _set("PII_ENCRYPTION_KEY", Fernet.generate_key().decode())
        ok("generated PII_ENCRYPTION_KEY (keep it! losing it makes stored PII unreadable)")
    env = _read_env()

    print("\n== database ==")
    if env.get("POSTGRES_USER") and env.get("POSTGRES_PASSWORD") and env.get("DB_BACKEND", "auto") != "sqlite":
        warn("POSTGRES_USER/PASSWORD are set -> PostgreSQL will be used. Is `docker compose up -d` running? "
             "If not, comment those two lines out in .env to use SQLite instead.")
    else:
        ok("no PostgreSQL credentials -> using SQLite (data/sqlite/), no server needed")

    print("\n== dataset ==")
    csv = ROOT / "data" / "raw" / "cs-training.csv"
    if csv.exists():
        ok(f"found {csv.relative_to(ROOT)}")
    elif env.get("KAGGLE_USERNAME", "").startswith("your_") or not env.get("KAGGLE_USERNAME"):
        warn("data/raw/cs-training.csv is missing. Either put KAGGLE_USERNAME / KAGGLE_KEY in .env "
             "(and accept the rules at kaggle.com/c/GiveMeSomeCredit, otherwise you get 403), "
             "or download the file manually and save it as data/raw/cs-training.csv")
    else:
        warn("data/raw/cs-training.csv is missing -> run: python -m src.python.extractors.kaggle_downloader "
             "(you must first click 'Late Submission / Join' at kaggle.com/c/GiveMeSomeCredit, else 403)")

    print("\n== tools ==")
    java = shutil.which("java")
    if java:
        out = subprocess.run([java, "-version"], capture_output=True, text=True).stderr.splitlines()[0]
        m = re.search(r'version "(\d+)(?:\.(\d+))?', out)
        major = (int(m.group(2)) if m and m.group(1) == "1" else int(m.group(1))) if m else 0
        if major >= 17:
            ok(f"Java: {out}")
        else:
            warn(f"Java: {out} - too old. PySpark 4.x needs Java 17 or newer (install Temurin 17 or 21).")
    else:
        warn("Java not found - PySpark (silver layer) will fail. Install JDK 17 and re-open the terminal.")
    if shutil.which("ollama"):
        ok("Ollama installed (run `ollama pull llama3.2` and keep `ollama serve` running for the chatbot)")
    else:
        warn("Ollama not found - the chatbot and decision letters need it: https://ollama.com")
    print()


if __name__ == "__main__":
    main()
