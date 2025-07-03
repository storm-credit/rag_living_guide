# app/web_ui.py

import streamlit as st
from app.pipeline import RAGPipeline
from app.llm_router import LLMRouter
from core.config import Settings

# 페이지 설정
st.set_page_config(page_title="RAG Living Guide", layout="wide")
st.title("🏠 자취생 & 외국인을 위한 생활 정보 챗봇")

# 설정 및 라우터/파이프라인 초기화
settings = Settings()
router = LLMRouter()
# 다국어 라우터에서 언어 감지 후 LLM을 넘겨줍니다.
# pipeline은 RAGPipeline(llm, settings) 형태로 생성
pipeline = RAGPipeline(llm=router, settings=settings)

# 사용자 입력
query = st.text_input("무엇이 궁금하신가요?")

if query:
    with st.spinner("답변 생성 중..."):
        # pipeline.run()이 내부적으로 retriever → router.generate() 호출
        answer = pipeline.run(query)
    st.markdown("### 📘 답변")
    st.success(answer)
