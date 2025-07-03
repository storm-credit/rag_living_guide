from pydantic import BaseSettings, Field

class AppSettings(BaseSettings):
    APP_TITLE: str = Field("RAG Living Guide", env="APP_TITLE")
    API_HOST: str = Field("0.0.0.0", env="API_HOST")
    API_PORT: int = Field(8000, env="API_PORT")

    class Config:
        env_file = ".env"
