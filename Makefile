# StressCam — Makefile
# Comandos de desenvolvimento para backend Python e frontend Next.js
#
# Author: Matheus Siqueira <https://www.matheussiqueira.dev/>

PYTHON   := python
VENV     := .venv
PIP      := $(VENV)/Scripts/pip
PYTEST   := $(VENV)/Scripts/pytest
STREAMLIT:= $(VENV)/Scripts/streamlit
NPM      := npm

.PHONY: help setup install install-dev test test-cov lint run run-streamlit \
        frontend frontend-dev frontend-build clean clean-sessions clean-models

# ── Ajuda ────────────────────────────────────────────────────────────────

help: ## Exibe esta mensagem de ajuda
	@echo ""
	@echo "  StressCam — Comandos disponíveis"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────

setup: ## Cria virtualenv e instala todas as dependências
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Ambiente configurado. Ative com: $(VENV)\Scripts\activate (Windows)"

install: ## Instala dependências de produção
	$(PIP) install -r requirements.txt

install-dev: ## Instala dependências de desenvolvimento (pytest, etc.)
	$(PIP) install -r requirements.txt pytest pytest-cov

# ── Testes ─────────────────────────────────────────────────────────────────

test: ## Executa toda a suite de testes com pytest
	$(PYTEST) -v

test-cov: ## Executa testes com relatório de cobertura
	$(PYTEST) --cov=stresscam --cov-report=term-missing --cov-report=html

test-watch: ## Executa testes em modo watch (requer pytest-watch)
	$(VENV)/Scripts/ptw tests/

# ── Qualidade de código ────────────────────────────────────────────────────

lint: ## Verifica tipos com mypy e estilo com ruff (requer pip install ruff mypy)
	$(VENV)/Scripts/ruff check stresscam/ tests/ || true
	$(VENV)/Scripts/mypy stresscam/ --ignore-missing-imports || true

# ── Backend ────────────────────────────────────────────────────────────────

run: ## Inicia o pipeline principal (HUD OpenCV)
	$(PYTHON) -m stresscam.app

run-streamlit: ## Inicia o dashboard Streamlit
	$(STREAMLIT) run stresscam/streamlit_app.py

run-rf: ## Inicia com modelo Random Forest
	STRESSCAM_MODEL_TYPE=rf $(PYTHON) -m stresscam.app

# ── Frontend (Next.js) ────────────────────────────────────────────────────

frontend: ## Instala depend��ncias do frontend
	$(NPM) install

frontend-dev: ## Inicia servidor de desenvolvimento Next.js
	$(NPM) run dev

frontend-build: ## Compila o frontend para produção
	$(NPM) run build

# ── Limpeza ────────────────────────────────────────────────────────────────

clean: ## Remove artefatos gerados (cache, pyc, logs)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.log" -not -path "./.venv/*" -delete 2>/dev/null || true

clean-sessions: ## Remove gravações de sessões anteriores
	rm -rf sessions/

clean-models: ## Remove modelos treinados
	rm -rf models/
