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
