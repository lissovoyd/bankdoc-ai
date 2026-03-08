"""API route aggregation."""

from fastapi import APIRouter

from routes.docs import router as docs_router
from routes.ask import router as ask_router

router = APIRouter()
router.include_router(docs_router)
router.include_router(ask_router)
