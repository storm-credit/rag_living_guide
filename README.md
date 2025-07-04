🏠 RAG Living Guide
한국어+영어 다국어 학습 가능한 RAG 시스템
“자취생 & 외국인 생활 정보 챗봇”

프로젝트 구조
rag_living_guide_full/
├── app/                             # 실행 애플리케이션 (Streamlit + FastAPI)
│   ├── main.py                      # FastAPI 엔트리 + Streamlit 런처
│   ├── config.py                    # Pydantic 설정
│   ├── pipeline.py                  # RAGPipeline 클래스
│   ├── llm_router.py                # LLMRouter 클래스
│   └── web_ui.py                    # Streamlit UI
│
├── core/                            # 핵심 로직 모듈
│   ├── __init__.py
│   ├── config.py                    # Settings (Pydantic BaseSettings)
│   ├── logger.py                    # 로깅 설정
│   ├── embedder.py                  # Embedder (배치·캐시 최적화)
│   ├── retriever.py                 # Retriever (metadata‐aware)
│   └── vectorstore.py               # VectorStore (FAISS/Qdrant 추상화)
│   └── llm/                         # LLM 추상화
│       ├── base.py                  # BaseLLM 인터페이스
│       ├── mistral.py               # MistralLLM (subprocess 개선)
│       └── koalpaca.py              # KoAlpacaLLM (retry·API key 지원)
│
├── training/                        # 학습 파이프라인
│   ├── retriever/                   # Dense Retriever 학습
│   │   ├── dataset.py               # TripletDataset (메타 포함)
│   │   ├── train.py                 # SBERT+Triplet Loss + Typer CLI
│   │   └── eval.py                  # IR 평가 스크립트 + metrics
│   └── llm/                         # LLM 파인튜닝
│       ├── prepare_data.py          # QA→Alpaca 형식 변환
│       └── train_lora.py            # PEFT LoRA 학습 + Typer CLI
│
├── scripts/                         # 유틸리티 스크립트
│   ├── build_faiss.py              # FAISS 인덱스 생성
│   ├── setup_faiss_env.bat         # Windows용 FAISS-only 환경
│   └── ci_checks.sh                # CI용 lint/format/test
│
├── tests/                           # 단위 & 통합 테스트
│   ├── conftest.py                 # pytest fixtures (mock 등)
│   ├── test_retriever.py
│   ├── test_llm.py
│   └── test_pipeline.py            # RAGPipeline 통합 테스트
│
├── data/                            # 원본 & 전처리 문서
│   ├── raw/                         # MD/PDF 원본
│   └── processed/                   # TXT/JSONL 변환본
│
├── vectorstore/                     # FAISS/Qdrant 인덱스
├── trained_models/                  # 학습된 embedding & LLM
├── .env.example                     # 환경변수 템플릿
├── .gitignore                       # Git 제외 목록
├── requirements.txt                 # 핵심 패키지
├── requirements-faiss.txt           # FAISS 전용 최소 패키지
├── Dockerfile                       # 컨테이너화
└── README.md                        # 프로젝트 개요 + 사용법


⚙️ 설치
git clone <repo_url>
cd rag_living_guide_full
pip install -r requirements.txt

(옵션) FAISS 전용 환경
pip install -r requirements-faiss.txt

🔧 환경 변수
.env.example 복사 → .env 생성

KOALPACA_API_URL=http://localhost:8001/generate
KOALPACA_API_KEY=
EMBEDDING_MODEL=trained_models/embedding_model/
VECTORSTORE_PATH=vectorstore/index
LOG_LEVEL=INFO

📖 데이터 전처리 & 인덱스
data/raw/ 에 원본 문서 배치
data/processed/ 에 txt 변환본 배치
python scripts/build_faiss.py

🚀 실행
FastAPI
uvicorn app.main:app --reload

Streamlit
streamlit run app/web_ui.py

🧪 테스트
pytest

📈 학습
Retriever (Dense)
python training/retriever/train.py --help

LLM (LoRA)
python training/llm/prepare_data.py
python training/llm/train_lora.py --help

📜 라이선스
MIT License