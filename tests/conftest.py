"""Pytest configuration and shared fixtures."""

import pytest
from unittest.mock import Mock, AsyncMock


@pytest.fixture
def mock_settings():
    """Mock application settings."""
    mock = Mock()
    mock.DATABASE_URL = "postgresql://test:test@localhost:5432/test"
    mock.REDIS_URL = "redis://localhost:6379"
    mock.DEBUG = True
    return mock


@pytest.fixture
def mock_logger():
    """Mock logger instance."""
    return Mock()


@pytest.fixture
def mock_db():
    """Mock database session."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def event_loop_policy():
    """Event loop policy for asyncio tests."""
    import asyncio
    if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.get_event_loop_policy()
