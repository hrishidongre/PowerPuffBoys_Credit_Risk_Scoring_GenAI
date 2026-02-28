# Credit Risk Scoring Engine

## From Score-Based Lending to Behavioral Risk Intelligence

### Project Overview

This project implements an **AI-driven credit risk classification system** that predicts borrower risk across four priority tiers using behavioral and financial signals from internal bank records and external CIBIL bureau data.

- **Milestone 1:** End-to-end ML pipeline -- data merging, cleaning, EDA, and multi-class classification using Decision Trees, Random Forests, and Gradient Boosting. Deliberate exclusion of Credit Score to force the model to learn from behavioral features.
- **Milestone 2:** Real-time scoring interface via Streamlit where bank officers can select prospects, adjust trade line inputs, and receive instant risk predictions with probability breakdowns.

---

### Technology Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python |
| **ML Models** | Decision Tree, Random Forest, HistGradientBoosting (scikit-learn) |
| **Data Processing** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **UI Framework** | Streamlit |
| **Model Serialization** | joblib |
| **Notebook** | Jupyter |

---

### Project Structure

```
PowerPuffBoys_Credit_Risk_Scoring_GenAI/
|
|-- Credit Risk Prediction.ipynb    # Full ML pipeline: EDA, cleaning, 3 models, evaluation
|-- app.py                          # Streamlit app for real-time risk prediction
|-- requirements.txt                # Python dependencies
|
|-- Dataset/
|   |-- Internal_Bank_Dataset.xlsx  # 25 trade line features per prospect
|   |-- External_Cibil_Dataset.xlsx # 60+ CIBIL features (delinquency, enquiries, demographics)
|   |-- Unseen_CIBL_Data.csv        # Held-out prospect data for inference
|   |-- schema.md                   # Complete data dictionary for all features
|
|-- models/
|   |-- finalized_model.joblib      # Trained HistGradientBoosting model (serialized)
```

---

### Risk Classification

The target variable `Approved_Flag` is mapped to four risk tiers:

| Class | Label | Description |
| :--- | :--- | :--- |
| **P1** | Very Low Risk | Strong repayment history, long credit history, minimal delinquency |
| **P2** | Low Risk | Generally reliable borrower, minor flags in recent activity |
| **P3** | Medium Risk | Notable delinquency patterns, limited or unstable credit history |
| **P4** | High Risk | Significant missed payments, recent delinquencies, high enquiry volume |

---

### Datasets

Two datasets are merged on `PROSPECT_ID` via inner join. Full data dictionary available in [`Dataset/schema.md`](Dataset/schema.md).

#### Internal Bank Dataset (25 Features)

Describes borrower account activity:

| Category | Features |
| :--- | :--- |
| **Account Counts** | Total trade lines, active vs. closed, opened/closed in last 6M and 12M |
| **Account Percentages** | Percent active, percent closed, percent opened in recent periods |
| **Missed Payments** | Total missed payment count |
| **Loan Type Breakdown** | Auto, Credit Card, Consumer, Gold, Home, Personal Loan, Secured, Unsecured, Other |
| **Account Age** | Age of oldest and newest trade lines (months) |

#### External CIBIL Dataset (60+ Features)

Bureau-level behavioral and demographic data:

| Category | Features |
| :--- | :--- |
| **Delinquency** | Times delinquent, max delinquency level, days past due (30+, 60+), delinquency in 6/12 months |
| **Payment Classification** | Standard, substandard, doubtful, and loss payment counts (overall, 6M, 12M) |
| **Enquiry Activity** | Total enquiries, CC and PL enquiries across 3M, 6M, 12M windows |
| **Demographics** | Age, gender, marital status, education, net monthly income, employment tenure |
| **Flags and Exposure** | CC/PL/HL/GL flags, unsecured exposure percentage, utilization metrics |

---

### Data Preprocessing Pipeline

| Step | Detail |
| :--- | :--- |
| **Sentinel Replacement** | `-99999` values converted to `NaN` (dataset convention for missing data) |
| **Column Removal** | `CC_utilization` and `PL_utilization` dropped (80%+ missing values) |
| **Delinquency Imputation** | 6 delinquency columns filled with `0` (null = no delinquency occurred) |
| **Numeric Imputation** | Remaining numeric columns filled with column median |
| **Duplicate Removal** | Duplicate rows dropped |
| **Target Encoding** | `Approved_Flag` mapped: P1=0, P2=1, P3=2, P4=3 |
| **One-Hot Encoding** | Applied to `MARITALSTATUS`, `EDUCATION`, `GENDER`, `last_prod_enq2`, `first_prod_enq2` |
| **Credit Score Removal** | Deliberately dropped -- see reasoning below |

---

### Key Design Decision: Dropping Credit Score

The initial Decision Tree with `Credit_Score` achieved **99.5% accuracy**, but feature importance revealed `Credit_Score` accounted for **99.95%** of the model's decisions:

```
Credit_Score                 0.999482
enq_L3m                      0.000164
time_since_recent_payment    0.000154
Age_Oldest_TL                0.000134
EDUCATION_UNDER GRADUATE     0.000066
```

The model was simply replicating CIBIL's existing output. Since the goal is to provide **explainable insight beyond credit score**, it was removed. After removal, the model learned from genuinely behavioral signals:

```
Age_Oldest_TL                   0.2346   (length of credit history)
enq_L3m                         0.2199   (recent borrowing intent)
time_since_recent_deliquency    0.1108   (recent repayment behavior)
num_std_12mts                   0.0854   (standard payments in last year)
time_since_recent_enq           0.0800   (recency of enquiry activity)
num_std                         0.0619   (overall standard payments)
pct_PL_enq_L6m_of_ever          0.0450   (personal loan enquiry trend)
Age_Newest_TL                   0.0223   (most recent account age)
max_deliq_12mts                 0.0125   (worst delinquency in last year)
time_since_first_deliquency     0.0085   (how long ago first missed payment)
```

---

### Model Development

Three models trained on 80/20 stratified split, all **without** Credit Score.

#### Model 1: Decision Tree (Baseline)

| Parameter | Value |
| :--- | :--- |
| **max_depth** | 5 |
| **criterion** | gini |
| **Accuracy** | 78% |
| **P3 Recall** | 0.31 |

Purpose: Establish baseline and extract interpretable decision path. Struggled heavily with P3 (Medium Risk).

#### Model 2: Random Forest (GridSearchCV)

| Parameter | Value |
| :--- | :--- |
| **n_estimators** | 300 |
| **max_depth** | None |
| **min_samples_leaf** | 3 |
| **Best CV Macro F1** | 0.698 |
| **P3 Recall** | 0.41 |

Improved P1 and P4, but P3 recall remained poor due to P2 class dominance in training data.

#### Model 3: HistGradientBoosting -- Cost-Sensitive (Final Model)

Custom sample weights applied to penalize P3 misclassification:

```python
class_weight = {0: 1.0, 1: 1.0, 2: 3.0, 3: 1.5}
```

| Parameter | Value |
| :--- | :--- |
| **max_depth** | 8 |
| **learning_rate** | 0.05 |
| **max_iter** | 300 |
| **P3 Recall** | 0.65 (up from 0.41) |

---

### Final Model Performance

Test set evaluation (10,268 samples):

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **P1** (Very Low Risk) | 0.81 | 0.79 | 0.80 | 1,161 |
| **P2** (Low Risk) | 0.90 | 0.83 | 0.86 | 6,440 |
| **P3** (Medium Risk) | 0.43 | 0.65 | 0.52 | 1,491 |
| **P4** (High Risk) | 0.86 | 0.65 | 0.74 | 1,176 |
| **Macro Avg** | 0.75 | 0.73 | 0.73 | 10,268 |
| **Weighted Avg** | 0.81 | 0.78 | -- | 10,268 |

> **Note:** P3 precision (0.43) is intentionally lower because the cost-sensitive weighting flags borderline P2 cases as P3. Being cautious in lending is preferred over missing medium-risk applicants.

#### Model Comparison

| Model | Accuracy | Macro F1 | P3 Recall | Selected |
| :--- | :--- | :--- | :--- | :--- |
| Decision Tree | 0.78 | 0.68 | 0.31 | No |
| Random Forest | 0.78 | 0.70 | 0.41 | No |
| **HistGradientBoosting** | **0.78** | **0.73** | **0.65** | **Yes** |

---

### Streamlit Application

The web app (`app.py`) serves as the end-user interface for bank officers.

#### Core Features

| Feature | Description |
| :--- | :--- |
| **Prospect Selection** | Dropdown to select any prospect from unseen CIBIL data, or randomize |
| **Trade Line Controls** | 25 adjustable inputs -- sliders for percentages, number inputs for counts |
| **Risk Prediction** | Color-coded risk banner (green/blue/orange/red) with probability scores for all 4 classes |
| **Reset Defaults** | One-click reset of all trade line inputs to sensible defaults |

#### Prediction Workflow

1. Select a prospect (or randomize) -- the app displays their full CIBIL record.
2. Adjust any of the 25 bureau trade line features (optional).
3. Click **"Predict Risk"** -- the app merges prospect data with trade line inputs, applies the same preprocessing as the notebook (sentinel replacement, delinquency fill, median imputation, one-hot encoding, feature alignment), runs the trained model, and displays the result.

---

### Getting Started

#### Prerequisites

- Python 3.9+

#### Installation

```bash
git clone https://github.com/adithyanst/PowerPuffBoys_Credit_Risk_Scoring_GenAI.git
cd PowerPuffBoys_Credit_Risk_Scoring_GenAI

python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

#### Run the App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

#### Reproduce the Pipeline

```bash
jupyter notebook "Credit Risk Prediction.ipynb"
```

Run all cells sequentially to reproduce data loading, merging, cleaning, EDA, model training, evaluation, and export.

---

### Deliverables

| Deliverable | Location |
| :--- | :--- |
| **ML Pipeline** | `Credit Risk Prediction.ipynb` |
| **Trained Model** | `models/finalized_model.joblib` |
| **Web Application** | `app.py` |
| **Data Dictionary** | `Dataset/schema.md` |
| **Datasets** | `Dataset/` |

---

### Team

**PowerPuffBoys**

---

### License

This project was developed for academic and research purposes.