from fastapi import FastAPI
from app.api import router, register_events

app = FastAPI(title="Vision API", version="1.0.0")
register_events(app)
app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "Vision API - OCR & Blur Detection"}
