from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from routers import chat, emergency, contacts, location
from database import init_db
from utils.config import config

app = FastAPI(title="MADO Assistant API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(emergency.router, prefix="/api/v1", tags=["Emergency"])
app.include_router(contacts.router, prefix="/api/v1", tags=["Contacts"])
app.include_router(location.router, prefix="/api/v1", tags=["Location"])


@app.get("/")
async def root():
    return {"status": "MADO API running", "version": "2.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=config.DEBUG)