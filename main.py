# Run with: uvicorn main:app --reload
# Worker:   celery -A celery_app worker --loglevel=info --pool=solo

import os
import subprocess
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import UPLOAD_DIR
from database import Base, engine
from logging_config import setup_logging
from routes import router

setup_logging(json_format=os.getenv("LOG_JSON", "").lower() == "true")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --wait makes Docker block until the postgres healthcheck passes
    subprocess.run(["docker", "compose", "up", "-d", "--wait"], check=True)

    # Rewrite pg_hba.conf to use trust auth (no password check) and
    # force lc_messages=C so PostgreSQL sends ASCII error messages
    # (avoids cp1251 UnicodeDecodeError on Russian Windows psycopg2).
    subprocess.run(
        ["docker", "exec", "-u", "postgres", "bankdoc_postgres",
         "bash", "-c",
         "printf 'local all all trust\\nhost all all all trust\\n'"
         " > /var/lib/postgresql/data/pg_hba.conf"
         " && echo \"lc_messages = 'C'\" >> /var/lib/postgresql/data/postgresql.conf"
         " && pg_ctl reload -D /var/lib/postgresql/data -s"],
        check=False,
    )
    time.sleep(2)  # give pg_hba + lc_messages reload a moment

    # Retry once — give lc_messages reload a bit more time on first run
    for attempt in range(2):
        try:
            Base.metadata.create_all(bind=engine)
            break
        except UnicodeDecodeError:
            if attempt == 0:
                logger.warning("PostgreSQL UnicodeDecodeError on attempt 1, retrying in 3s...")
                time.sleep(3)
                continue
            # Second failure: show diagnostics
            import socket as _socket
            try:
                with _socket.create_connection(("127.0.0.1", 5432), timeout=3):
                    tcp_status = "OK — port 5432 is reachable"
            except Exception as _e:
                tcp_status = f"FAILED — {_e}"
            raise RuntimeError(
                "PostgreSQL connection failed after retry (Windows locale issue).\n"
                f"TCP port test: {tcp_status}\n"
                "Try: docker exec bankdoc_postgres psql -U postgres -c \"ALTER SYSTEM SET lc_messages='C'; SELECT pg_reload_conf();\"\n"
                "Or:  docker compose down -v && restart"
            )

    yield
    subprocess.run(["docker", "compose", "stop"], check=False)


app = FastAPI(title="BankDoc AI", version="2.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(router)

templates = Jinja2Templates(directory="templates")


@app.get("/")
def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/uploads/{filename}")
def serve_pdf(filename: str):
    from fastapi import HTTPException
    path = UPLOAD_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="application/pdf")


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "BankDoc AI"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
