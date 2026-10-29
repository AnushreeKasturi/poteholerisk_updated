# PotholeRisk — Bengaluru Dynamic Risk Engine

A full-stack web application that analyzes road and weather conditions to estimate pothole-related driving risk across Bengaluru.

🚀 Live Demo

https://potholerisk.onrender.com

## Features

* Real-time weather integration using OpenWeather API
* Secure backend API proxy using Node.js & Express
* Dynamic risk analysis engine
* Responsive frontend UI
* Environment variable protection using `.env`
* API key hidden from the browser and DevTools

---

## Tech Stack

* Frontend: HTML, CSS, JavaScript
* Backend: Node.js, Express.js
* API: OpenWeather API

---

## Security

The OpenWeather API key is stored securely on the server using environment variables.

The browser never directly accesses the API key.

### Architecture

```text
Browser  ──GET /api/weather──▶  Express Server  ──▶ OpenWeather API
         ◀── Weather JSON ───                 ◀── Weather JSON
```

* The frontend communicates only with the backend server
* `server.js` injects the API key securely
* The API key is never exposed in frontend code or network requests

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AnushreeKasturi/Potholerisk.git
cd Potholerisk
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Configure Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Add your OpenWeather API key:

```env
OPENWEATHER_API_KEY=your_api_key_here
```

---

## Running the Project

```bash
npm start
```

Open in browser:

```text
http://localhost:3000
```

---

## Research Code (paper results)

This repository also contains the code and results for the accompanying paper.

| Result | What it measures | Where |
|---|---|---|
| **73.1% accuracy** (weighted F1 72.4%, ROC-AUC 0.778) | Real predictive performance: XGBoost classifier predicting High-Risk (Fatal / Grievous) vs Lower-Risk accidents on Karnataka State Police Bengaluru records, 2016–2023, stratified 80:20 hold-out split | `models_final/` |
| 93.6% accuracy | Internal consistency only: how well a classifier recovers the risk formula's own LOW/MEDIUM/HIGH labels on simulated data. **Not** a measure of accident-prediction accuracy | `synthetic_validation/` |

### Reproducing the results

```bash
pip install -r requirements.txt
python models_final/train_final_model.py               # real-data model, metrics, confusion matrix, ROC curve
python synthetic_validation/synthetic_formula_validation.py   # synthetic consistency check
```

Outputs are written to `models_final/` (metrics, classification report with
confusion matrix, trained model) and `results/` (`confusion_matrix.png`,
`roc_curve.png`). Results are deterministic with the pinned versions in
`requirements.txt`.

### Data

* `dataset/processed/` — cleaned and feature-engineered accident and traffic data used by the models.
* `dataset/accident/` — raw Karnataka State Police accident reports (not committed to git; obtain from the original source).
* `dataset/traffic/` — Bengaluru traffic dataset.
* `synthetic_validation/synthetic_scenarios.csv` — the generated synthetic dataset (20,000 scenarios).

---

## Project Structure

```text
Potholerisk/
├── server.js                  # Express backend (weather API proxy)
├── public/index.html          # Frontend
├── package.json
├── requirements.txt           # Python dependencies for the research code
├── .env.example
├── dataset/                   # Raw and processed data
├── notebooks/                 # Exploration, preprocessing, feature engineering, modelling
├── models_final/              # Final XGBoost model + evaluation (paper's main result)
├── models_improved/           # Intermediate experiments leading to the final model
├── synthetic_validation/      # Formula internal-consistency check
└── results/                   # Figures (confusion matrix, ROC curve)
```

---

## Future Improvements

* Live pothole detection using computer vision
* Accident probability prediction
* Route safety scoring
* Interactive GIS mapping
* User reporting system

---

## Author

Anushree Kasturi
