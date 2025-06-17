# from pydantic_settings import BaseSettings
# # from pydantic import HttpUrl
# class Settings(BaseSettings):
#     MONGO_URI: str
#     MONGO_DB: str
#     MONGO_COLLECTION: str
#     # OLLAMA_BASE_URL: HttpUrl
#     OLLAMA_MODEL: str

#     class Config:
#         env_file = ".env"

# settings = Settings()

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    MONGO_URI: str
    MONGO_DB: str
    MONGO_COLLECTION: str

    # tambahkan ini
    OLLAMA_BASE_URL: str
    OLLAMA_MODEL: str
    
    # Postgres config (TAMBAHKAN INI!)
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",  # agar field lain juga tidak bikin error
    )

settings = Settings()
