import os
import sys

# Force English messages + UTF-8 locale so psycopg2 can decode server messages on Russian Windows
os.environ['LANGUAGE'] = 'en_US:en'
os.environ['PGCLIENTENCODING'] = 'UTF8'
os.environ['LC_ALL'] = 'C'
os.environ['LC_MESSAGES'] = 'C'
os.environ['PYTHONUTF8'] = '1'

# On Windows, also reset the C-runtime locale so libpq uses ASCII/UTF-8 error messages
if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.cdll.msvcrt.setlocale(0, b'C')  # LC_ALL = 0
    except Exception:
        pass

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv # type: ignore

load_dotenv()

_url = (
    os.getenv("DATABASE_URL", "postgresql://bankdoc:bankdoc123@127.0.0.1:5433/bankdoc")
    .strip()
    .replace("localhost", "127.0.0.1")  # avoid IPv6 resolution on Windows
)
# Disable SSL (Docker postgres has no SSL cert; SSL negotiation errors also show as Russian)
DATABASE_URL = _url + ("&" if "?" in _url else "?") + "sslmode=disable&client_encoding=utf8"

engine = create_engine(DATABASE_URL, connect_args={"client_encoding": "utf8", "options": "-c client_encoding=UTF8"})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
