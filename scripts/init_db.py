"""Initialize TimescaleDB schema from SQL file."""
import subprocess
import sys
from pathlib import Path

def main():
    from src.config import get_settings
    settings = get_settings()
    sql_file = Path(__file__).parent / "init_db.sql"

    if not sql_file.exists():
        print(f"ERROR: {sql_file} not found")
        sys.exit(1)

    print(f"Initializing database: {settings.db_url}")

    from urllib.parse import urlparse
    import os
    parsed = urlparse(settings.db_url)

    env = {**os.environ, "PGPASSWORD": parsed.password or "alphaforge"}

    cmd = [
        "psql",
        f"--host={parsed.hostname}",
        f"--port={parsed.port or 5432}",
        f"--username={parsed.username}",
        f"--dbname={parsed.path.lstrip('/')}",
        f"--file={sql_file}",
    ]

    result = subprocess.run(cmd, env=env)
    if result.returncode == 0:
        print("✅ Database initialized successfully")
    else:
        print("❌ Database initialization failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
