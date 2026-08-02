PY ?= .venv/Scripts/python.exe

.PHONY: help install data features scorecard train reject calibrate explain fairness onnx api dashboard test lint all clean

help:
	@echo "make install    - install package + dev extras"
	@echo "make data       - generate the synthetic mobile-money book"
	@echo "make features   - WOE/IV binning + feature matrix"
	@echo "make scorecard  - fit the interpretable logistic challenger"
	@echo "make train      - LightGBM/XGBoost + Optuna, tracked in MLflow"
	@echo "make reject     - reject inference on the through-the-door population"
	@echo "make calibrate  - probability calibration + points scaling + profit curve"
	@echo "make fairness   - disparate impact / equal opportunity audit"
	@echo "make onnx       - export champion to ONNX and benchmark p99 latency"
	@echo "make api        - run the FastAPI scoring service"
	@echo "make dashboard  - run the Streamlit risk dashboard"
	@echo "make all        - full pipeline end to end"

install:
	$(PY) -m pip install -e ".[dev,dashboard]"

data:
	$(PY) -m mkopo.cli generate-data

features:
	$(PY) -m mkopo.cli build-features

scorecard:
	$(PY) -m mkopo.cli fit-scorecard

train:
	$(PY) -m mkopo.cli train

reject:
	$(PY) -m mkopo.cli reject-inference

calibrate:
	$(PY) -m mkopo.cli calibrate

fairness:
	$(PY) -m mkopo.cli fairness

onnx:
	$(PY) -m mkopo.cli export-onnx

api:
	$(PY) -m uvicorn mkopo.api.main:app --reload --port 8000

dashboard:
	$(PY) -m streamlit run dashboard/app.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

all: data features scorecard train reject calibrate fairness onnx

clean:
	rm -rf artifacts data/raw data/processed mlruns .pytest_cache .ruff_cache
