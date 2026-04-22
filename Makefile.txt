.PHONY: help install setup dev test lint format clean docker-build docker-up docker-down migrate

help:
	@echo "Making rainwater harvesting platform..."
	@echo ""
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make setup         - Complete development setup"
	@echo "  make dev           - Start development environment"
	@echo "  make test          - Run test suite"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make lint          - Run all linters (black, flake8, mypy, isort)"
	@echo "  make format        - Format code with black and isort"
	@echo "  make clean         - Remove build artifacts and caches"
	@echo "  make docker-build  - Build Docker images"
	@echo "  make docker-up     - Start Docker containers"
	@echo "  make docker-down   - Stop Docker containers"
	@echo "  make migrate       - Run database migrations"

install:
	pip install --upgrade pip setuptools wheel
	pip install -e ".[dev]"

setup: install
	pre-commit install
	pre-commit run --all-files || true

dev:
	docker-compose -f deployment/docker/docker-compose.yml up

test:
	pytest

test-cov:
	pytest --cov=backend --cov=core --cov=data --cov-report=html

lint:
	black --check backend core data
	isort --check-only backend core data
	flake8 backend core data
	mypy backend

format:
	black backend core data
	isort backend core data

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf build dist *.egg-info

docker-build:
	docker-compose -f deployment/docker/docker-compose.yml build

docker-up:
	docker-compose -f deployment/docker/docker-compose.yml up -d

docker-down:
	docker-compose -f deployment/docker/docker-compose.yml down

migrate:
	alembic upgrade head
