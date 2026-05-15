from fastapi import FastAPI
from app.api.routes import router
from app.ui.routes import router as ui_router
from app.core.config import settings

# Initialize the FastAPI app
app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(router, prefix="/api/v1")
app.include_router(ui_router)

@app.get("/")
def health_check():
    return {"status": "System is online and ready for RAG."}