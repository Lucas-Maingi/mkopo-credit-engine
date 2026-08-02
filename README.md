<div align="center">

# Mkopo Score

**Behavioural credit scoring for mobile-money economies.**

Build a credit profile for someone who has no credit file — from how they move money.

</div>

---

## The problem

In Kenya, roughly 84% of adults use mobile money and a small minority have a traditional
credit file. Products like Fuliza and KCB M-Pesa solved this by scoring people on the
data that *does* exist: transaction behaviour. Mkopo Score is a complete, production-shaped
implementation of that idea — from synthetic data generation through to a calibrated,
explainable, drift-monitored scoring API.

It is built to be judged the way a real lending book is judged, not the way a Kaggle
notebook is judged:

| Most portfolio projects | Mkopo Score |
|---|---|
| Optimise AUC | Optimise expected portfolio profit at a chosen approval rate |
| Train on approved loans only | Correct the selection bias with **reject inference** |
| Output a probability | Output a **300–850 score** with PDO-scaled points and reason codes |
| "Model is explainable (SHAP)" | Adverse-action reason codes a customer could actually read |
| Accuracy only | **Fairness audit** against the four-fifths rule + equal opportunity |
| `model.pkl` | ONNX export with a **p99 latency SLO** |

## Status

🚧 Under active construction — built stage by stage, one commit per stage.

- [x] Project scaffold, configuration, lending economics
- [ ] Synthetic mobile-money data engine
- [ ] WOE/IV feature layer
- [ ] Logistic scorecard challenger
- [ ] LightGBM / XGBoost champion (Optuna + MLflow)
- [ ] Reject inference
- [ ] Calibration, points scaling, profit curve
- [ ] SHAP explainability + reason narratives
- [ ] Fairness audit
- [ ] ONNX export + latency benchmark
- [ ] FastAPI scoring service
- [ ] Streamlit dashboard + drift monitoring
- [ ] Test suite + CI + Docker + model card

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -e ".[dev,dashboard]"
make all          # full pipeline: data -> features -> models -> calibration -> ONNX
make api          # scoring service on :8000
make dashboard    # risk dashboard on :8501
```

## Repository layout

```
src/mkopo/
  config.py          scorecard scaling, lending economics, fairness thresholds
  data/              synthetic mobile-money book generator
  features/          WOE/IV binning and feature engineering
  models/            scorecard, GBMs, reject inference, calibration, ONNX export
  evaluation/        AUC/KS/Gini, PSI/CSI, expected-profit curve, fairness
  explain/           SHAP -> signed points -> plain-English reason codes
  api/               FastAPI scoring service
dashboard/           Streamlit risk dashboard
docs/                model card, model risk note, data dictionary
tests/               pytest suite
```

## A note on the data

All data in this repository is **synthetic**. No real customer data, from Safaricom or
anyone else, is used. The generator is causally structured — a latent repayment capacity
drives the observable behaviours — so the models learn genuine structure rather than a
leaked shortcut. See [`docs/data-dictionary.md`](docs/data-dictionary.md).

## Licence

MIT © Lucas Maingi
