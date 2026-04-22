@echo off
REM Pre-commit hook setup script for local development (Windows)

echo Installing pre-commit hooks...
pre-commit install

echo Running initial checks...
pre-commit run --all-files

echo Done! Pre-commit hooks installed and verified.
pause
