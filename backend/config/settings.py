"""
Configuration module for the Rainwater Harvesting Platform.
Environment-based configuration using Pydantic.
"""

from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings - loads from environment variables"""
    
    # ==== APPLICATION ====
    APP_NAME: str = "Rainwater Harvesting Platform"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = True
    
    # ==== DATABASE ====
    DATABASE_URL: str = "postgresql://rainwater:rainwater_dev_pass@localhost:5432/rainwater_dev"
    SQLALCHEMY_ECHO: bool = False
    SQLALCHEMY_POOL_SIZE: int = 20
    SQLALCHEMY_MAX_OVERFLOW: int = 40
    SQLALCHEMY_POOL_PRE_PING: bool = True
    
    # ==== REDIS ====
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_DB: int = 0
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    
    # ==== INFLUXDB ====
    INFLUXDB_URL: str = "http://localhost:8086"
    INFLUXDB_ORG: str = "rainwater"
    INFLUXDB_BUCKET: str = "rainwater"
    INFLUXDB_TOKEN: str = "your-token-here"
    
    # ==== API ====
    API_TITLE: str = "Rainwater Harvesting Platform API"
    API_DESCRIPTION: str = "Geospatial rainwater harvesting analysis"
    API_VERSION: str = "2.0.0"
    API_DOCS_URL: str = "/api/v1/docs"
    API_OPENAPI_URL: str = "/api/v1/openapi.json"
    
    # ==== CORS ====
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8501"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: list = ["*"]
    CORS_HEADERS: list = ["*"]
    
    # ==== AUTHENTICATION ====
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # ==== WEATHER API ====
    METEOSTAT_TIMEOUT: int = 30
    METEOSTAT_RETRIES: int = 3
    METEOSTAT_BACKOFF_FACTOR: float = 2.0
    
    # ==== SIMULATION ====
    SIMULATION_MAX_WORKERS: int = 4
    SIMULATION_MAX_QUEUE_SIZE: int = 1000
    SIMULATION_TIMEOUT_SECONDS: int = 300
    
    # ==== LOGGING ====
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # or "text"
    LOG_FILE: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()
