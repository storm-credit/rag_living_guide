@echo off
REM 🧙🏾‍♂️ FAISS 전용 가상환경 생성 스크립트

echo Creating virtual environment in .venv_faiss...
python -m venv .venv_faiss

echo Activating environment...
call .venv_faiss\Scripts\activate

echo Installing faiss and numpy (>=1.25) only...
pip install --upgrade pip
pip install faiss-cpu numpy>=1.25.0

echo.
echo ✅ FAISS 전용 환경 구성 완료!
echo 🔄 사용할 땐 다음 명령으로 활성화:
echo call .venv_faiss\Scripts\activate
pause