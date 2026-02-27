Customer Churn Predictor


OVERVIEW:

A complete end-to-end machine learning classification project that predicts whether a telecom customer will cancel their subscription (churn). The system gives a churn probability score and explains WHY a customer is at risk — enabling retention teams to intervene before customers leave.

Business Impact: If a telecom company has 10,000 customers and 2,600 are about to churn, this model identifies ~1,650 of them in advance,enabling targeted retention offers.

WHERE I GOT THE DATA:

Dataset: IBM Telco Customer Churn Dataset

Source:  Kaggle

Size:    7,043 rows × 21 features

Target:  Churn (Yes = customer left, No = customer stayed)

Features include:

- Demographics    — gender, age (senior citizen), partner, dependents
  
- Account Info    — tenure (months), contract type, payment method, paperless billing
  
- Services        — phone, internet, online security, online backup, device protection, tech support, streaming TV/movies
  
- Charges         — monthly charges, total charges

WHAT I DID — STEP BY STEP:

PHASE 1: Exploratory Data Analysis (EDA)

- Discovered 26.5% churn rate — imbalanced dataset(74% Stay vs 26% Leave)
  
- Found TotalCharges stored as string dtype — converted to numeric, found 11 missing values for new customers
  
- Filled missing TotalCharges with 0 (new customers have no total charges yet)
  
- Key findings from visualizations:
  
  → Month-to-month customers churn 3x more than two-year contract customers
  
  → New customers (tenure < 6 months) are at highest risk
  
  → Fiber optic internet customers churn more than DSL
  
  → Higher monthly charges correlate with higher churn
  
  → Senior citizens have a higher churn rate

PHASE 2: Feature Engineering & Encoding

- Dropped customerID (no predictive value)
  
- Binary encoded 11 Yes/No columns (Yes-1, No-0)
  
- One-hot encoded 4 multi-value columns:
  
  → Contract (3 values)
  
  → InternetService (3 values)
  
  → MultipleLines (3 values)
  
  → PaymentMethod (4 values)
  
- Created 4 engineered features:
  
  → charges_per_tenure  = MonthlyCharges / (tenure + 1)(cost burden relative to loyalty)
  
  → total_services      = sum of all add-on services(engagement score — more services = less churn)
  
  → is_new_customer     = 1 if tenure < 6 months(new customers at highest early churn risk)
  
  → high_charges        = 1 if charges > $64.76 average(above-average bill flag)
  
- Applied Stratified 80/20 train/test split to preserve the 26.5% churn ratio in both sets
  
- Applied SMOTE (Synthetic Minority Oversampling):
  
  Before: {Stay: 4,139, Churn: 1,495} — imbalanced
  
  After:  {Stay: 4,139, Churn: 4,139} — balanced
  
  SMOTE applied ONLY to training data — never test data
  
- Fitted StandardScaler on training data only

PHASE 3: Model Training & Evaluation

- Trained and compared 6 classification models:
  
  Model                | Accuracy | Recall | ROC-AUC
  
  Logistic Regression  | 77.7%    | 63.6%  | 0.832 
  
  Gradient Boosting    | 76.8%    | 65.8%  | 0.830
  
  XGBoost              | 76.8%    | 60.7%  | 0.822
  
  Random Forest        | 77.0%    | 58.6%  | 0.821
  
  LightGBM             | 76.9%    | 59.6%  | 0.815
  
  Decision Tree        | 73.5%    | 63.1%  | 0.753

SURPRISE FINDING: Logistic Regression beat XGBoost! This tells us the churn decision boundary is largely linear — simpler models win when relationships are straightforward. This also makes the model more interpretable and faster to deploy.

Confusion Matrix (best model):

  True Negatives  (correctly said Stay):   857
  
  False Positives (wrongly said Churn):    178
  
  False Negatives (missed churners!):      136
  
  True Positives  (correctly caught):      238

WHY RECALL MATTERS HERE:

Missing a churner (False Negative) = company loses a customer worth thousands in lifetime revenue.

False alarm (False Positive) = company sends an unnecessary retention offer, much cheaper cost.Therefore we optimised for Recall over Precision.

PHASE 4: Web Application

- Built with Streamlit + Plotly
  
- Sidebar with all customer profile inputs
  
- Real-time churn risk score (0-100%)
  
- Colour-coded gauge (red=high risk, green=low risk)
  
- Smart business insights explaining WHY customer is at risk
  
- Key churn drivers chart showing what increases/decreases risk
  
- Deployed to Streamlit Cloud

RESULTS

Best Model:    Logistic Regression

ROC-AUC:       0.832 (83.2% chance of correctly ranking a churner above a non-churner)

Recall:        63.6% (catches 63.6% of actual churners)

F1 Score:      0.603

Training Size: 8,278 (after SMOTE balancing)

Test Size:     1,409

TECH STACK

Language:   Python 

Libraries:  scikit-learn, imbalanced-learn (SMOTE), XGBoost, LightGBM, pandas, numpy, plotly, streamlit

Deployment: Streamlit Cloud

LIVE DEMO: https://customer-churn-predictor-tu9qpmbzlbwsxgecbutz8e.streamlit.app/

