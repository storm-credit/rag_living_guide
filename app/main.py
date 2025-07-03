from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from core.config import Settings
from app.llm_router import LLMRouter
from app.pipeline import RAGPipeline
from core.llm.base import BaseLLM
from core.llm.mistral import MistralLLM
from core.llm.koalpaca import KoAlpacaLLM

settings = Settings()
app = FastAPI(title="RAG Living Guide API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터: 한국어 KoAlpaca, 영어 Mistral
llm_router = LLMRouter()
# 파이프라인: 기본적으로 Mistral 사용
pipeline = RAGPipeline(llm_router.en, settings)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/query")
def query_endpoint(q: str):
    try:
        answer = pipeline.run(q)
        return {"query": q, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
