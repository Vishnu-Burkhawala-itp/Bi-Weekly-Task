from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Production Tech Docs Q&A API"
    LLAMA_CLOUD_API_KEY: str 
    COHERE_API_KEY: str
    OPENAI_API_KEY: str 
    
    class Config:
        env_file = ".env"

settings = Settings()