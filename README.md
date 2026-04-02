# 🏥 Healthcare ML Pipeline Dashboard

**University of Sheffield · MSc Data Analytics · Dissertation Project 2024**

> An interactive end-to-end machine learning pipeline for predicting healthcare billing amounts — developed and evaluated by Dhiman Kumar, supervised by Dr Fatima Maikore.

[![Live Dashboard](https://img.shields.io/badge/Dashboard-Live-2dd4bf?style=flat-square)](https://your-username.github.io/healthcare-ml-pipeline)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-F7931E?style=flat-square)](https://scikit-learn.org)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat-square)](https://www.kaggle.com/datasets/awaiskaggler/insurance-csv)

---

## 📊 Dashboard Preview

Open `index.html` in any modern browser or visit the GitHub Pages deployment to explore:

- **Pipeline Overview** — full end-to-end workflow visualisation
- **Data Exploration** — insurance dataset analysis and distributions  
- **Model Sections** — Linear Regression, Decision Tree, Gradient Boost deep dives
- **Model Comparison** — side-by-side R², MSE, and radar chart comparisons
- **Feature Analysis** — importance scores and feature configuration impact
- **Code Reference** — complete Python implementation snippets

---

## 🎯 Key Results

| Model | Test R² | Test MSE | Train R² | Train MSE |
|-------|---------|----------|---------|-----------|
| Linear Regression | 0.8025 | 0.0346 | 0.7295 | 0.0419 |
| Decision Tree (tuned) | 0.8524 | 0.0258 | 0.8216 | 0.0277 |
| **Gradient Boost ★** | **0.8641** | **0.0238** | **0.8457** | **0.0239** |

**Gradient Boosting** achieved the best overall performance — lowest MSE (0.0238), highest R² (0.8641), and near-perfect generalisation (Train/Test MSE gap of only 0.0001).

---

## 🗂 Project Structure

```
healthcare-ml-pipeline/
├── index.html          # Interactive dashboard (open in browser)
├── README.md           # This file
└── pipeline.py         # Python ML pipeline (optional — see below)
```

---

## 🔬 Dataset

**Source:** [Kaggle — Insurance CSV](https://www.kaggle.com/datasets/awaiskaggler/insurance-csv)

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Patient age (18–64) |
| `sex` | Categorical | Male / Female → Label encoded |
| `bmi` | Numeric | Body Mass Index (15.96–53.13) |
| `children` | Numeric | Number of dependants |
| `smoker` | Categorical | Yes / No → Label encoded (dominant predictor) |
| `region` | Categorical | US region |
| `charges` | Target | Medical billing amount → Log₁₀ transformed |

**Records:** 1,338 (after removing 1 duplicate)  
**Split:** 80% training (~1,070) / 20% testing (~268)  
**Missing values:** None

---

## ⚙️ Pipeline Steps

```
Data Collection → Pre-processing → Feature Engineering → Model Training → Evaluation → Results
      ↓                ↓                  ↓                   ↓              ↓           ↓
  Kaggle CSV     Remove dupes        Log transform        LR / DT / GB    R² + MSE   GBoost wins
                 Label encode        Select features      Hyperparams     Scatter      0.8641 R²
                 Check nulls         Train/test split     Tuning          Bar plots
```

---

## 🧠 Models

### 1. Linear Regression (Baseline)
- **Test R²:** 0.8025 · **Test MSE:** 0.0346  
- Assumes linear relationships between features and log(charges)  
- Strong interpretable baseline; outperforms similar Kaggle benchmarks (R²=0.6951)  
- Limitation: misses non-linear interactions

### 2. Decision Tree Regressor (Tuned)
- **Test R²:** 0.8524 · **Test MSE:** 0.0258  
- Hyperparameters: `max_depth=30`, `min_samples_split=50`, `min_samples_leaf=10`  
- **Critical finding:** Without tuning → Train R²=0.9993 but Test R²=0.5696 (severe overfitting)  
- Tuning raised Test R² from 0.57 → **0.85**

### 3. Gradient Boosting Regressor ★ Best
- **Test R²:** 0.8641 · **Test MSE:** 0.0238  
- Sequential ensemble — each tree corrects previous errors  
- No manual hyperparameter tuning required  
- Train/Test MSE gap: only **0.0001** — near-perfect generalisation

---

## 📈 Feature Importance

| Feature | Estimated Importance | Role |
|---------|---------------------|------|
| **smoker** | ~72% | Dominant predictor — smokers pay 3–4× more |
| age | ~18% | Strong linear relationship with charges |
| bmi | ~8% | Moderate influence, especially for smokers |
| sex | ~2% | Minimal impact on billing |

**Key insight:** Removing `smoker` from features causes Test R² to drop from 0.86 → ~0.32 (−53%).

---

## 🚀 Running the Python Pipeline

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Download dataset from Kaggle
# Place insurance.csv in project root

# Run pipeline
python pipeline.py
```

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load & preprocess
medical = pd.read_csv('insurance.csv').drop_duplicates()
le_sex, le_smoker = LabelEncoder(), LabelEncoder()
medical['sex']     = le_sex.fit_transform(medical['sex'])
medical['smoker']  = le_smoker.fit_transform(medical['smoker'])
medical['charges'] = np.log10(medical['charges'])

X = medical[['age', 'sex', 'bmi', 'smoker']]
Y = medical['charges']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train & evaluate all three models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree':     DecisionTreeRegressor(max_depth=30, min_samples_split=50, min_samples_leaf=10, random_state=42),
    'Gradient Boost':    GradientBoostingRegressor(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, Y_train)
    test_pred  = model.predict(X_test)
    train_pred = model.predict(X_train)
    print(f"\n{name}")
    print(f"  Test  R²: {r2_score(Y_test, test_pred):.4f} | MSE: {mean_squared_error(Y_test, test_pred):.4f}")
    print(f"  Train R²: {r2_score(Y_train, train_pred):.4f} | MSE: {mean_squared_error(Y_train, train_pred):.4f}")
```

---

## 📚 References

1. Choi et al. (2022) — High-Cost Patient Prediction using NHS Korea data. [NCBI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9603723/)
2. BioMedical Engineering OnLine — ML for HCHN patient expenditure prediction. [Link](https://biomedical-engineering-online.biomedcentral.com/articles/10.1186/s12938-018-0568-3)
3. Langenberger et al. — ML for high-cost patient prediction via healthcare claims. [NCBI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9847900/)
4. Ke et al. (2017) — LightGBM: A Highly Efficient Gradient Boosting Decision Tree. [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

---

## 🎓 Academic Context

- **Degree:** MSc Data Analytics
- **Department:** Computer Science, University of Sheffield
- **Supervisor:** Dr Fatima Maikore
- **Module:** Dissertation Project (23-24)
- **Submitted:** September 10, 2024

---

## 📄 License

This project is submitted as an academic dissertation. Dataset © Kaggle / respective authors.

---

*Dashboard built with Chart.js · Python pipeline built with scikit-learn*
