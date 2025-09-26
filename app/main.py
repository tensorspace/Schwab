# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from .ingest import load_articles
from .retriever import HybridRetriever
from .pipeline import run_query
from .monitor import make_app_monitoring

DF = load_articles("data/stock_news.json")
RETR = HybridRetriever(DF["text"].tolist())

app = FastAPI(title="News Chat")
make_app_monitoring(app)  # metrics, health

class ChatReq(BaseModel):
    message: str
    k: int = 3
    use_llm: bool = False

@app.post("/chat")
def chat(body: ChatReq):
    return {"answers": run_query(DF, RETR, body.message, k=body.k, use_llm=body.use_llm)}

@app.get("/healthz")
def health():
    return {"ok": True}