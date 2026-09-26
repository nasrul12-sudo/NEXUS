.PHONY: help install install-dev test lint format typecheck clean run-supervisor

help:
	@echo "NEXUS development commands:"
	@echo "  make install        Install runtime dependencies"
	@echo "  make install-dev    Install dev dependencies"
	@echo "  make test           Run tests"
	@echo "  make lint           Run ruff"
	@echo "  make format         Format with ruff"
	@echo "  make typecheck      Run mypy"
	@echo "  make clean          Clean caches"
	@echo "  make run-supervisor Run process supervisor"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=nexus --cov-report=term-missing

lint:
	ruff check nexus/ tests/

format:
	ruff format nexus/ tests/
	ruff check --fix nexus/ tests/

typecheck:
	mypy nexus/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -f /tmp/nexus-events.sock

run-supervisor:
	python -m nexus.launcher.supervisor
