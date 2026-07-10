# ABSA — task runner. On Windows without GNU Make, use the mirror script:  ./make.ps1 <target>
ifeq ($(OS),Windows_NT)
	PY  := .venv/Scripts/python.exe
else
	PY  := .venv/bin/python
endif
PIP := $(PY) -m pip

.DEFAULT_GOAL := help
.PHONY: help setup setup-gpu prepare-data train train-baseline train-transformer \
        evaluate serve demo export benchmark test lint format type check clean

help:  ## Show this help
	@echo "ABSA targets:"
	@echo "  setup            Create CPU venv + install (demo/dev machine)"
	@echo "  setup-gpu        Create venv + CUDA install (training desktop)"
	@echo "  prepare-data     Download/parse SemEval -> data/processed"
	@echo "  train            Train baseline + transformer (deberta-v3, best)"
	@echo "  train-demo       Fine-tune bert-base for the CPU demo (quantized)"
	@echo "  evaluate         Evaluate + write reports/figures"
	@echo "  serve            Run FastAPI service on :8000"
	@echo "  demo             Run Streamlit demo (CPU, offline)"
	@echo "  export           Export int8/ONNX CPU artifact -> artifacts/"
	@echo "  benchmark        Print CPU latency + artifact size"
	@echo "  test / lint / type / format / check   Quality gates"

setup:  ## CPU venv + install (demo/dev laptop)
	python -m venv .venv
	$(PIP) install --upgrade pip wheel
	$(PIP) install -r requirements-cpu.txt
	-$(PY) -m spacy download en_core_web_sm
	-$(PY) -m pre_commit install

setup-gpu:  ## CUDA venv + install (training desktop, RTX 4070)
	python -m venv .venv
	$(PIP) install --upgrade pip wheel
	$(PIP) install -r requirements-gpu.txt
	-$(PY) -m spacy download en_core_web_sm

prepare-data:  ## Prepare datasets
	$(PY) scripts/prepare_data.py

train: train-baseline train-transformer  ## Train both tracks

train-baseline:
	$(PY) scripts/train.py --track baseline

train-transformer:
	$(PY) scripts/train.py --track transformer

train-demo:  ## Fine-tune bert-base for the CPU demo (quantization-friendly)
	$(PY) scripts/train.py --track transformer --encoder bert-base-uncased --epochs 3 \
		--output-dir artifacts/transformer-bert --no-mlflow

evaluate:  ## Evaluate + build reports
	$(PY) scripts/evaluate.py

export:  ## Export CPU deployment artifact (int8 + optional ONNX)
	$(PY) scripts/export_model.py

benchmark:  ## CPU latency + artifact size
	$(PY) scripts/benchmark.py

serve:  ## FastAPI service
	$(PY) -m uvicorn absa.serving.app:app --host 0.0.0.0 --port 8000

demo:  ## Streamlit demo (CPU, offline)
	$(PY) -m streamlit run app/streamlit_app.py

test:  ## Run tests + coverage gate
	$(PY) -m pytest

lint:  ## ruff + black --check
	$(PY) -m ruff check .
	$(PY) -m black --check .

format:  ## Auto-fix lint + format
	$(PY) -m ruff check --fix .
	$(PY) -m black .

type:  ## mypy
	$(PY) -m mypy src/absa

check: lint type test  ## All gates

clean:  ## Remove caches/build artifacts
	$(PY) -c "import shutil,glob,os; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache','.mypy_cache','.ruff_cache','build','dist','htmlcov']]"
