

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&display=swap');
    .main { background-color: #0f1117; }
    .block-container { padding-top: 3rem; }
    .title-text {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f87171, #fb923c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle-text {
        color: #64748b; font-size: 1rem;
        margin-top: 0.2rem; margin-bottom: 2rem;
    }
    .metric-card {
        background: #1e2330; border: 1px solid #2d3748;
        border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
    }
    .metric-label { color: #64748b; font-size: 0.8rem;
        letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.4rem; }
    .metric-value { color: #e2e8f0; font-size: 1.6rem; font-weight: 700; }
    .risk-high {
        background: linear-gradient(135deg, #2e1a1a, #3d1f1f);
        border: 2px solid #f87171; border-radius: 16px;
        padding: 2rem; text-align: center; margin: 1.5rem 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #1a2e1a, #1a3a1a);
        border: 2px solid #4ade80; border-radius: 16px;
        padding: 2rem; text-align: center; margin: 1.5rem 0;
    }
    .risk-label { font-size: 0.85rem; letter-spacing: 2px;
        text-transform: uppercase; margin-bottom: 0.5rem; }
    .risk-value { color: white; font-size: 3rem; font-weight: 800;
        font-family: 'Syne', sans-serif; }
    .risk-sub { color: #64748b; font-size: 0.9rem; margin-top: 0.5rem; }
    .insight-box {
        background: #13161e; border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem; margin: 0.5rem 0;
        color: #94a3b8; font-size: 0.9rem;
    }
    .insight-red  { border-left: 3px solid #f87171; }
    .insight-green{ border-left: 3px solid #4ade80; }
    .insight-amber{ border-left: 3px solid #fb923c; }
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_artifacts():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    with open('model_info.pkl', 'rb') as f:
        info = pickle.load(f)
    return model, scaler, features, info

model, scaler, feature_names, model_info = load_artifacts()


st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="title-text">📡 Customer Churn Predictor</p>',

            unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">ML-powered churn risk analysis · '
            'Logistic Regression · ROC-AUC: 0.832 · Telco Dataset</p>',
            unsafe_allow_html=True)


c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-label">Model</div>'
                '<div class="metric-value">Logistic Reg.</div></div>',
                unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-label">ROC-AUC</div>'
                '<div class="metric-value">0.832</div></div>',
                unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-label">Recall</div>'
                '<div class="metric-value">63.6%</div></div>',
                unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="metric-label">Customers</div>'
                '<div class="metric-value">7,043</div></div>',
                unsafe_allow_html=True)

st.markdown("---")



st.sidebar.markdown("## 👤 Customer Profile")
st.sidebar.markdown("Fill in the customer details:")

st.sidebar.markdown("### 📋 Account Info")
tenure         = st.sidebar.slider("Tenure (months)", 0, 72, 12)
contract       = st.sidebar.selectbox("Contract Type",
                    ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Payment Method",
                    ["Electronic check", "Mailed check",
                     "Bank transfer (automatic)", "Credit card (automatic)"])
paperless      = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

st.sidebar.markdown("### 💰 Charges")
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 119.0, 65.0, 0.5)

st.sidebar.markdown("### 🌐 Services")
internet       = st.sidebar.selectbox("Internet Service",
                    ["Fiber optic", "DSL", "No"])
phone          = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines",
                    ["Yes", "No", "No phone service"])
online_sec     = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup  = st.sidebar.selectbox("Online Backup",   ["Yes", "No", "No internet service"])
device_prot    = st.sidebar.selectbox("Device Protection",["Yes", "No", "No internet service"])
tech_support   = st.sidebar.selectbox("Tech Support",    ["Yes", "No", "No internet service"])
streaming_tv   = st.sidebar.selectbox("Streaming TV",    ["Yes", "No", "No internet service"])
streaming_mov  = st.sidebar.selectbox("Streaming Movies",["Yes", "No", "No internet service"])

st.sidebar.markdown("### 👥 Demographics")
gender     = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior     = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner    = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])



def encode(val): return 1 if val in ['Yes', 'Male'] else 0
def encode_svc(val): return 1 if val == 'Yes' else 0

total_charges = monthly_charges * tenure


charges_per_tenure = monthly_charges / (tenure + 1)
total_services     = sum([
    encode_svc(phone), encode_svc(online_sec),
    encode_svc(online_backup), encode_svc(device_prot),
    encode_svc(tech_support), encode_svc(streaming_tv),
    encode_svc(streaming_mov)
])
is_new_customer = 1 if tenure < 6 else 0
high_charges    = 1 if monthly_charges > 64.76 else 0


input_dict = {
    'gender':            encode(gender),
    'SeniorCitizen':     encode(senior),
    'Partner':           encode(partner),
    'Dependents':        encode(dependents),
    'tenure':            tenure,
    'PhoneService':      encode_svc(phone),
    'OnlineSecurity':    encode_svc(online_sec),
    'OnlineBackup':      encode_svc(online_backup),
    'DeviceProtection':  encode_svc(device_prot),
    'TechSupport':       encode_svc(tech_support),
    'StreamingTV':       encode_svc(streaming_tv),
    'StreamingMovies':   encode_svc(streaming_mov),
    'PaperlessBilling':  encode(paperless),
    'MonthlyCharges':    monthly_charges,
    'TotalCharges':      total_charges,
    # MultipleLines
    'MultipleLines_No phone service': 1 if multiple_lines == 'No phone service' else 0,
    'MultipleLines_Yes':              1 if multiple_lines == 'Yes' else 0,
    # InternetService
    'InternetService_Fiber optic':    1 if internet == 'Fiber optic' else 0,
    'InternetService_No':             1 if internet == 'No' else 0,
    # Contract
    'Contract_One year':              1 if contract == 'One year' else 0,
    'Contract_Two year':              1 if contract == 'Two year' else 0,
    # PaymentMethod
    'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
    'PaymentMethod_Electronic check':        1 if payment_method == 'Electronic check' else 0,
    'PaymentMethod_Mailed check':            1 if payment_method == 'Mailed check' else 0,
    # Engineered
    'charges_per_tenure': charges_per_tenure,
    'total_services':     total_services,
    'is_new_customer':    is_new_customer,
    'high_charges':       high_charges,
}

input_df     = pd.DataFrame([input_dict])[feature_names]
input_scaled = scaler.transform(input_df)


churn_prob  = model.predict_proba(input_scaled)[0][1]
churn_pred  = model.predict(input_scaled)[0]
risk_pct    = churn_prob * 100
risk_label  = "HIGH RISK 🔴" if churn_prob > 0.6 else \
              "MEDIUM RISK 🟡" if churn_prob > 0.4 else "LOW RISK 🟢"



left, right = st.columns([1, 1.2], gap="large")

with left:
    
    box_class = "risk-high" if churn_prob > 0.5 else "risk-low"
    label_color = "#f87171" if churn_prob > 0.5 else "#4ade80"
    st.markdown(f"""
    <div class="{box_class}">
        <div class="risk-label" style="color:{label_color}">Churn Risk Score</div>
        <div class="risk-value">{risk_pct:.1f}%</div>
        <div class="risk-sub">{risk_label}</div>
        <div class="risk-sub" style="margin-top:0.3rem">
            {"⚠️ This customer is likely to leave soon" if churn_prob > 0.5
             else "✅ This customer is likely to stay"}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Risk insights
    st.markdown("### 💡 Risk Factors")

    if contract == "Month-to-month":
        st.markdown('<div class="insight-box insight-red">🔴 <b>Month-to-month contract</b> — highest churn risk. Consider offering a yearly plan discount.</div>', unsafe_allow_html=True)
    elif contract == "Two year":
        st.markdown('<div class="insight-box insight-green">🟢 <b>Two-year contract</b> — very low churn risk.</div>', unsafe_allow_html=True)

    if is_new_customer:
        st.markdown('<div class="insight-box insight-red">🔴 <b>New customer</b> (under 6 months) — high early churn risk. Prioritize onboarding.</div>', unsafe_allow_html=True)

    if internet == "Fiber optic" and high_charges:
        st.markdown('<div class="insight-box insight-amber">🟡 <b>Fiber optic + high charges</b> — this combination has elevated churn in our data.</div>', unsafe_allow_html=True)

    if total_services >= 5:
        st.markdown('<div class="insight-box insight-green">🟢 <b>High service engagement</b> — customers with many services churn less.</div>', unsafe_allow_html=True)
    elif total_services <= 2:
        st.markdown('<div class="insight-box insight-amber">🟡 <b>Low service engagement</b> — consider upselling add-ons to increase retention.</div>', unsafe_allow_html=True)

    if payment_method == "Electronic check":
        st.markdown('<div class="insight-box insight-amber">🟡 <b>Electronic check payments</b> — associated with higher churn in this dataset.</div>', unsafe_allow_html=True)

    # Customer summary
    st.markdown("### 📋 Customer Summary")
    summary = pd.DataFrame({
        'Feature': ['Tenure', 'Contract', 'Monthly Charges',
                    'Total Services ✨', 'Is New Customer ✨',
                    'Charges/Tenure ✨'],
        'Value': [f"{tenure} months", contract, f"${monthly_charges:.2f}",
                  f"{total_services}/7", "Yes" if is_new_customer else "No",
                  f"${charges_per_tenure:.2f}"]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)


with right:
    # Gauge chart
    gauge_color = "#f87171" if churn_prob > 0.6 else \
                  "#fb923c" if churn_prob > 0.4 else "#4ade80"

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        number={'suffix': '%', 'font': {'size': 36,
                'color': gauge_color}},
        gauge={
            'axis': {'range': [0, 100], 'ticksuffix': '%'},
            'bar':  {'color': gauge_color},
            'bgcolor': "#1e2330",
            'bordercolor': "#2d3748",
            'steps': [
                {'range': [0, 40],   'color': '#1a2e1a'},
                {'range': [40, 60],  'color': '#2e2a1a'},
                {'range': [60, 100], 'color': '#2e1a1a'},
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 3},
                'thickness': 0.8, 'value': 50
            }
        },
        title={'text': "Churn Probability",
               'font': {'color': '#94a3b8', 'size': 14}}
    ))
    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e2e8f0'},
        height=300, margin=dict(t=60, b=10)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Feature importance chart
    st.markdown("### 📊 Key Churn Drivers (Model Insights)")
    drivers = pd.DataFrame({
        'Factor': ['Month-to-month contract', 'Fiber optic internet',
                   'High monthly charges', 'New customer (<6mo)',
                   'Electronic check payment', 'No tech support',
                   'Long tenure', 'Two-year contract',
                   'Many services', 'Has dependents'],
        'Impact': [0.35, 0.28, 0.22, 0.19, 0.15,
                   0.12, -0.30, -0.28, -0.18, -0.12],
    }).sort_values('Impact')

    fig_drivers = go.Figure(go.Bar(
        x=drivers['Impact'],
        y=drivers['Factor'],
        orientation='h',
        marker_color=['#f87171' if x > 0 else '#4ade80'
                      for x in drivers['Impact']],
        text=[f"+{v:.0%}" if v > 0 else f"{v:.0%}"
              for v in drivers['Impact']],
        textposition='outside'
    ))
    fig_drivers.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e2e8f0'},
        height=380,
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis={'gridcolor': '#1e2330',
               'title': '← Reduces Churn | Increases Churn →'},
        yaxis={'gridcolor': '#1e2330'}
    )
    st.plotly_chart(fig_drivers, use_container_width=True)
    st.caption("🔴 Red = increases churn risk  🟢 Green = reduces churn risk")


# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#334155; font-size:0.8rem; padding:1rem 0'>
    Built with ❤️ · Logistic Regression + Streamlit · Telco Churn Dataset ·
    ROC-AUC=0.832 · <a href='https://github.com/Akhiliny99/customer-churn-predictor' style='color:#f87171'>View on GitHub</a>
</div>

""", unsafe_allow_html=True)
