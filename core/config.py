from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    KOALPACA_API_URL: str = Field("http://localhost:8001/generate", env="KOALPACA_API_URL")
    KOALPACA_API_KEY: str | None = Field(None, env="KOALPACA_API_KEY")
    EMBEDDING_MODEL: str = Field("trained_models/embedding_model/", env="EMBEDDING_MODEL")
    VECTORSTORE_PATH: str = Field("vectorstore/index", env="VECTORSTORE_PATH")
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
