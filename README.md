# Stochastic Credit Engine
### NovaCred Bank · Credit Default Risk via Stochastic Processes

> *"A borrower's financial life is not a snapshot — it's a trajectory through uncertainty."*

**Live demo:** [stochastic-credit-engine.onrender.com](https://stochastic-credit-engine.onrender.com) &nbsp;|&nbsp; **API docs:** `/docs`

---

## What is STPA?

STPA models credit default risk as a **living stochastic system** — not a static classification problem.  
Instead of predicting "will this person default: yes/no", STPA asks:

> *"What does this borrower's financial future look like across 10,000 possible timelines?"*

It then outputs:
- **PD Score (0–100)** — probability of default as a score
- **Survival Curve** — P(no default) at each future month
- **Expected Time to Default**
- **Stress-tested PD** — how the score changes under recession conditions

---

## The Three Engines

| Engine | Process | Purpose |
|--------|---------|---------|
| 🔵 Markov Chain | Non-Homogeneous CTMC | Tracks borrower's credit state zone |
| 🟢 OU Diffusion | Ornstein-Uhlenbeck SDE | Models slow drift of financial health |
| 🔴 Jump Process | Compound Poisson | Models sudden shocks (job loss, medical) |

All three engines feed into a **Monte Carlo fusion layer** that runs 10,000 path simulations per borrower.

---

## Architecture

```
series_2/2b/stpa/
│
├── core/
│   ├── markov_engine.py       # Markov state transition chains
│   ├── diffusion_engine.py    # Ornstein-Uhlenbeck process
│   ├── jump_engine.py         # Compound Poisson jump shocks
│   └── monte_carlo.py         # Fusion simulation layer
│
├── risk/
│   ├── oracle.py              # PD scoring + risk assessment output
│   ├── stress_tester.py       # Macro scenario stress testing
│   └── survival.py            # Survival curve analysis
│
├── data/
│   ├── loader.py              # UCI / Lending Club / Synthetic data
│   └── calibrator.py          # Estimate OU + Markov params from data
│
├── api/
│   ├── main.py                # FastAPI (port 8000)
│   └── schemas.py             # Pydantic request/response models
│
├── dashboard/
│   ├── app.py                 # Flask app (port 5000)
│   └── plots.py               # Plotly chart generators
│
├── database/
│   └── db.py                  # SQLite persistence layer
│
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Start the dashboard (new terminal)
python dashboard/app.py
```

---

## API Usage

```python
import requests

response = requests.post("http://localhost:8000/simulate", json={
    "borrower_id": "B001",
    "health_score": 55.0,
    "long_run_mean": 50.0,
    "reversion_speed": 0.3,
    "volatility": 10.0,
    "initial_state": "STRESSED",
    "risk_tier": "high_risk",
    "horizon_months": 24,
    "n_simulations": 5000
})

result = response.json()
print(f"PD Score: {result['pd_score']} / 100")
print(f"Risk Tier: {result['risk_tier']}")
print(f"Expected default in: {result['expected_ttd_months']:.1f} months")
```

---

## Stress Testing

Run all macro scenarios in one call:

```python
response = requests.post("http://localhost:8000/stress-test", json={
    "borrower_id": "B001",
    "health_score": 55.0,
    ...
})

# Returns PD scores under: Baseline, Mild Recession, Severe Recession,
# Rate Shock, Unemployment Spike, Combined Shock
```

---

## Mathematics

### OU Process
$$dX_t = \theta(\mu - X_t)dt + \sigma dW_t$$

### Jump Shocks  
$$J(t) = \sum_{i=1}^{N(t)} Y_i, \quad N(t) \sim \text{Poisson}(\lambda t)$$

### Markov Transitions
$$P(\text{state}_{t+1} = j \mid \text{state}_t = i) = P_{ij}$$

### Probability of Default
$$PD = \frac{1}{N}\sum_{n=1}^{N} \mathbf{1}[\min_t X_t^{(n)} < X_{\text{threshold}}]$$

---

## Dataset

Supports:
- **UCI Credit Card Default** — Taiwan dataset (30,000 borrowers)
- **Lending Club** — US peer-to-peer lending data
- **Synthetic** — Built-in generator for testing

---

*Built with NumPy, SciPy, FastAPI, Flask, Plotly, SQLite*
