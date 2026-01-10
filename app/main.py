from fastapi import FastAPI
from app.api.routes.health import router



app = FastAPI(
    title="Smart Healthcare Diagnostic Platform",
    version="1.0.0"
)

app.include_router(router)

