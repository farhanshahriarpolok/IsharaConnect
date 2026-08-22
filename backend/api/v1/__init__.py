"""API v1 routers package."""

from backend.api.v1.auth import router as auth_router
from backend.api.v1.certificates import router as certificates_router
from backend.api.v1.exams import router as exams_router
from backend.api.v1.nlp import router as nlp_router
from backend.api.v1.progress import router as progress_router
from backend.api.v1.rooms import router as rooms_router
from backend.api.v1.users import router as users_router

__all__ = [
    "auth_router",
    "certificates_router",
    "exams_router",
    "nlp_router",
    "progress_router",
    "rooms_router",
    "users_router",
]
