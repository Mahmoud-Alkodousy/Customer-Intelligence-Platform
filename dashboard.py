import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# ================= CONFIG =================
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= POLISHED PROFESSIONAL CSS =================
st.markdown("""
<style>
    /* Import Modern Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main Background */
    .main {
        background-color: #000000;
        padding-top: 1rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #f16f01;
    }
    
    /* Headers - Improved Typography */
    .main-header {
        font-size: 2.75rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.75rem;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    
    .main-header .accent {
        color: #f16f01;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #ffffff;
        opacity: 0.7;
        margin-bottom: 3rem;
        font-weight: 400;
        line-height: 1.5;
    }
    
    /* Section Headers - Better Visibility */
    h2 {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.25rem !important;
        margin-bottom: 0.75rem !important;
    }
    
    h4 {
        color: #f16f01 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.75rem !important;
        border-bottom: 1px solid #333333 !important;
    }
    
    /* Metric Cards - Improved Spacing */
    .metric-card {
        background-color: #000000;
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 2rem 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        border-color: #f16f01;
        box-shadow: 0 4px 16px rgba(241, 111, 1, 0.15);
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #f16f01;
        margin: 0.75rem 0;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #ffffff;
        opacity: 0.6;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 500;
    }
    
    .metric-subtitle {
        font-size: 0.85rem;
        color: #ffffff;
        opacity: 0.5;
        margin-top: 0.75rem;
    }
    
    /* Risk Level Variants - Enhanced */
    .risk-high {
        border-color: #ff0000;
        background: linear-gradient(135deg, rgba(255, 0, 0, 0.03) 0%, rgba(0, 0, 0, 0) 100%);
    }
    
    .risk-high .metric-value {
        color: #ff0000;
    }
    
    .risk-medium {
        border-color: #ff7900;
        background: linear-gradient(135deg, rgba(255, 121, 0, 0.03) 0%, rgba(0, 0, 0, 0) 100%);
    }
    
    .risk-medium .metric-value {
        color: #ff7900;
    }
    
    .risk-low {
        border-color: #00ff00;
        background: linear-gradient(135deg, rgba(0, 255, 0, 0.03) 0%, rgba(0, 0, 0, 0) 100%);
    }
    
    .risk-low .metric-value {
        color: #00ff00;
    }
    
    /* Status Badge - Better Contrast */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.25rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    .status-online {
        background-color: rgba(0, 255, 0, 0.15);
        color: #00ff00;
        border: 1px solid #00ff00;
    }
    
    .status-offline {
        background-color: rgba(255, 0, 0, 0.15);
        color: #ff0000;
        border: 1px solid #ff0000;
    }
    
    .status-high {
        background-color: rgba(255, 0, 0, 0.15);
        color: #ff0000;
        border: 1px solid #ff0000;
    }
    
    .status-medium {
        background-color: rgba(255, 121, 0, 0.15);
        color: #ff7900;
        border: 1px solid #ff7900;
    }
    
    .status-low {
        background-color: rgba(0, 255, 0, 0.15);
        color: #00ff00;
        border: 1px solid #00ff00;
    }
    
    /* Tabs - Improved */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: #000000;
        border-bottom: 2px solid #333333;
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3.5rem;
        padding: 0 2.5rem;
        font-size: 1rem;
        font-weight: 600;
        background-color: transparent;
        border: none;
        color: #ffffff;
        opacity: 0.5;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        opacity: 0.8;
        background-color: rgba(241, 111, 1, 0.05);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: transparent;
        border-bottom: 3px solid #f16f01;
        color: #f16f01;
        opacity: 1;
    }
    
    /* Buttons - More Prominent */
    .stButton > button {
        background: linear-gradient(135deg, #f16f01 0%, #ff7900 100%);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.875rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(241, 111, 1, 0.25);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff7900 0%, #f16f01 100%);
        box-shadow: 0 6px 20px rgba(241, 111, 1, 0.4);
        transform: translateY(-2px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Download Button Variant */
    .stDownloadButton > button {
        background-color: #000000;
        color: #f16f01;
        border: 2px solid #f16f01;
        border-radius: 10px;
        padding: 0.875rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        background-color: #f16f01;
        color: #ffffff;
        box-shadow: 0 6px 20px rgba(241, 111, 1, 0.3);
    }
    
    /* Input Fields - Better Spacing and Visibility */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: #000000;
        border: 2px solid #333333;
        border-radius: 8px;
        color: #ffffff;
        font-weight: 500;
        font-size: 0.95rem;
        padding: 0.75rem 1rem;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #f16f01;
        box-shadow: 0 0 0 3px rgba(241, 111, 1, 0.1);
        outline: none;
    }
    
    /* Input Labels - Better Visibility */
    .stTextInput label,
    .stNumberInput label,
    .stSelectbox label,
    .stSlider label {
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.5rem !important;
        opacity: 0.9 !important;
    }
    
    /* Sliders - Enhanced */
    .stSlider {
        padding: 1.5rem 0;
    }
    
    .stSlider > div > div > div > div {
        background-color: #f16f01;
    }
    
    .stSlider > div > div > div {
        background-color: #333333;
    }
    
    /* Number Input Controls */
    .stNumberInput button {
        background-color: #1a1a1a;
        border: 1px solid #333333;
        color: #ffffff;
        width: 2rem;
        height: 2rem;
    }
    
    .stNumberInput button:hover {
        background-color: #f16f01;
        border-color: #f16f01;
    }
    
    /* Dataframes - Better Styling */
    .dataframe {
        background-color: #000000;
        border: 1px solid #333333;
        border-radius: 8px;
        font-size: 0.9rem;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #f16f01 0%, #ff7900 100%);
    }
    
    /* Expander - Improved */
    .streamlit-expanderHeader {
        background-color: #000000;
        border: 1px solid #333333;
        border-radius: 8px;
        color: #ffffff;
        font-weight: 600;
        font-size: 1rem;
        padding: 1rem 1.5rem;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #f16f01;
        background-color: rgba(241, 111, 1, 0.03);
    }
    
    /* Info Box - Enhanced */
    .info-box {
        background-color: #000000;
        border: 2px solid #333333;
        border-left: 4px solid #f16f01;
        border-radius: 8px;
        padding: 2rem;
        margin: 1.5rem 0;
    }
    
    .info-box h3 {
        color: #f16f01 !important;
        margin-bottom: 1rem !important;
        font-weight: 600 !important;
        font-size: 1.15rem !important;
    }
    
    .info-box p {
        color: #ffffff;
        line-height: 1.8;
        opacity: 0.85;
        margin: 0;
        font-size: 0.95rem;
    }
    
    /* Warning Box */
    .warning-box {
        border-left-color: #ff7900;
    }
    
    .warning-box h3 {
        color: #ff7900 !important;
    }
    
    /* Error Box */
    .error-box {
        border-left-color: #ff0000;
    }
    
    .error-box h3 {
        color: #ff0000 !important;
    }
    
    /* File Uploader - Better Design */
    .stFileUploader {
        background-color: #000000;
        border: 2px dashed #333333;
        border-radius: 12px;
        padding: 2.5rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #f16f01;
        background-color: rgba(241, 111, 1, 0.02);
    }
    
    /* Form Sections - Better Spacing */
    .form-section {
        background-color: #000000;
        border: 2px solid #333333;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    
    .form-section h4 {
        color: #ffffff !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        margin-bottom: 1.5rem !important;
        padding-bottom: 0.75rem !important;
        border-bottom: 1px solid #333333 !important;
    }
    
    /* Results Container */
    .results-container {
        background-color: #000000;
        border: 2px solid #333333;
        border-radius: 12px;
        padding: 2.5rem;
    }
    
    /* Review Cards - Enhanced */
    .review-card {
        background-color: #000000;
        border: 2px solid #333333;
        border-left: 4px solid #ff7900;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .review-card:hover {
        border-color: #f16f01;
        border-left-color: #f16f01;
        box-shadow: 0 4px 12px rgba(241, 111, 1, 0.1);
    }
    
    .review-card .review-header {
        color: #f16f01;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .review-card .review-text {
        color: #ffffff;
        line-height: 1.7;
        opacity: 0.85;
        font-size: 0.95rem;
    }
    
    /* Scrollbar - Refined */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background-color: #000000;
    }
    
    ::-webkit-scrollbar-thumb {
        background-color: #333333;
        border-radius: 5px;
        border: 2px solid #000000;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background-color: #f16f01;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #333333;
        margin: 2.5rem 0;
    }
    
    /* Form Spacing Override */
    .stForm {
        padding: 2rem;
        background-color: #000000;
        border: 2px solid #333333;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    
    /* Column Gaps */
    [data-testid="column"] {
        padding: 0 1.5rem;
    }
    
    /* Form Column First/Last Child */
    [data-testid="column"]:first-child {
        padding-left: 0;
    }
    
    [data-testid="column"]:last-child {
        padding-right: 0;
    }
    
    /* Section Headers inside Forms */
    .stForm h4 {
        margin-top: 0 !important;
    }
    
    /* Markdown Headers inside Forms */
    .stForm p strong {
        display: block;
        color: #f16f01 !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        margin-bottom: 1.25rem !important;
        margin-top: 1.5rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 1px solid #333333 !important;
    }
    
    .stForm p strong:first-of-type {
        margin-top: 0 !important;
    }
    
    /* Success/Error Messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        background-color: #000000;
        border: 2px solid;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-size: 0.95rem;
    }
    
    .stSuccess {
        border-color: #00ff00;
        color: #00ff00;
    }
    
    .stError {
        border-color: #ff0000;
        color: #ff0000;
    }
    
    .stWarning {
        border-color: #ff7900;
        color: #ff7900;
    }
    
    /* Caption Text */
    .stCaption {
        color: #ffffff !important;
        opacity: 0.6 !important;
        font-size: 0.9rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ================= HELPER FUNCTIONS =================
def check_api_health():
    """Check API health status"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None


def predict_churn(customer_data):
    """Predict customer churn"""
    try:
        response = requests.post(
            f"{API_URL}/predict_churn",
            json=customer_data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def analyze_customer(customer_data):
    """Comprehensive customer analysis"""
    try:
        response = requests.post(
            f"{API_URL}/analyze_customer",
            json=customer_data,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def create_gauge_chart(value, title):
    """Create an enhanced gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': title,
            'font': {'size': 20, 'color': '#ffffff', 'family': 'Inter', 'weight': 600}
        },
        number={
            'font': {'size': 48, 'color': '#f16f01', 'family': 'Inter', 'weight': 700},
            'suffix': '%'
        },
        gauge={
            'axis': {
                'range': [None, 100],
                'tickwidth': 2,
                'tickcolor': "#666666",
                'tickfont': {'size': 14, 'color': '#ffffff'}
            },
            'bar': {'color': "#f16f01", 'thickness': 0.75},
            'bgcolor': "#0a0a0a",
            'borderwidth': 2,
            'bordercolor': "#333333",
            'steps': [
                {'range': [0, 40], 'color': "rgba(0, 255, 0, 0.15)"},
                {'range': [40, 70], 'color': "rgba(255, 121, 0, 0.15)"},
                {'range': [70, 100], 'color': "rgba(255, 0, 0, 0.15)"}
            ],
            'threshold': {
                'line': {'color': "#ff0000", 'width': 4},
                'thickness': 0.85,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#ffffff", 'family': 'Inter'},
        height=320,
        margin=dict(l=30, r=30, t=70, b=30)
    )
    
    return fig


# ================= MAIN APP =================
def main():
    
    # Header
    st.markdown('''
        <div style="text-align: center; padding: 2rem 0 1.5rem 0;">
            <h1 class="main-header">Customer Intelligence <span class="accent">Platform</span></h1>
            <p class="sub-header">AI-Powered Churn Prediction & Customer Analytics</p>
        </div>
    ''', unsafe_allow_html=True)
    
    # API Health Check
    is_healthy, health_data = check_api_health()
    
    if not is_healthy:
        st.markdown("""
            <div class="info-box error-box">
                <h3>⚠️ Backend Service Offline</h3>
                <p>The API server is not running. Please start it with: <code style="color: #ff0000; font-weight: 600;">python backend.py --serve</code></p>
            </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # System Status
    with st.expander("⚙️ System Status", expanded=False):
        cols = st.columns(4)
        
        with cols[0]:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">API Status</div>
                    <div style="margin: 1rem 0;">
                        <div class="status-badge status-online">ONLINE</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            models = health_data.get("models_loaded", {})
            loaded = sum(models.values())
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Models Loaded</div>
                    <div class="metric-value" style="font-size: 1.75rem;">{loaded}/3</div>
                </div>
            """, unsafe_allow_html=True)
        
        with cols[2]:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Version</div>
                    <div class="metric-value" style="font-size: 1.75rem;">2.0</div>
                </div>
            """, unsafe_allow_html=True)
        
        with cols[3]:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">System Status</div>
                    <div style="margin: 1rem 0;">
                        <div class="status-badge status-online">ACTIVE</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ================= MAIN TABS =================
    tab1, tab2, tab3 = st.tabs(["🎯 Customer Analysis", "📦 Batch Processing", "📊 Analytics"])
    
    # ================= TAB 1: Customer Analysis =================
    with tab1:
        st.markdown("### Customer Risk Analysis")
        st.markdown("Analyze individual customer profiles to predict churn risk and receive AI-powered recommendations.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.form("customer_analysis_form"):
            st.markdown("#### 👤 Customer Profile")
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Demographics**")
                age = st.slider("Age", 18, 80, 35)
                
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("**Behavioral Metrics**")
                total_actions = st.number_input("Total Actions", 0, 500, 25)
                avg_duration = st.number_input("Avg Session Duration (sec)", 0, 1000, 180)
                conversion_rate = st.slider("Conversion Rate", 0.0, 1.0, 0.15, 0.01)
                dropoff_rate = st.slider("Drop-off Rate", 0.0, 1.0, 0.30, 0.01)
                unique_products = st.number_input("Unique Products Viewed", 0, 50, 5)
                days_active = st.number_input("Days Active", 0, 365, 30)
            
            with col2:
                st.markdown("**Reviews & Ratings**")
                avg_rating = st.slider("Average Rating", 0.0, 5.0, 3.5, 0.1)
                positive_reviews = st.number_input("Positive Reviews", 0, 50, 5)
                negative_reviews = st.number_input("Negative Reviews", 0, 20, 1)
                total_reviews = positive_reviews + negative_reviews
                sentiment_ratio = positive_reviews / max(total_reviews, 1)
                
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("**Engagement Metrics**")
                total_views = st.number_input("Total Product Views", 0, 5000, 200)
                total_clicks = st.number_input("Total Clicks", 0, 1000, 50)
                click_through_rate = total_clicks / max(total_views, 1)
                avg_product_price = st.number_input("Avg Product Price ($)", 0.0, 1000.0, 150.0)
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("🔍 Analyze Customer", use_container_width=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Results Section
        if submit_btn:
            with st.spinner("Analyzing customer profile..."):
                time.sleep(0.3)
                
                payload = {
                    "Age": age,
                    "total_actions": total_actions,
                    "avg_duration": avg_duration,
                    "avg_rating": avg_rating,
                    "negative_reviews": negative_reviews,
                    "positive_reviews": positive_reviews,
                    "total_views": total_views,
                    "total_clicks": total_clicks,
                    "conversion_rate": conversion_rate,
                    "dropoff_rate": dropoff_rate,
                    "click_through_rate": click_through_rate,
                    "sentiment_ratio": sentiment_ratio,
                    "avg_product_price": avg_product_price,
                    "unique_products_viewed": unique_products,
                    "days_active": days_active,
                    "total_reviews": total_reviews
                }
                
                result = analyze_customer(payload)
                
                if result:
                    st.markdown("### 📊 Analysis Results")
                    
                    # Risk Metrics
                    risk_cols = st.columns(3)
                    
                    churn_prob = result.get("churn_probability", 0)
                    risk_level = result.get("risk_level", "UNKNOWN")
                    priority = result.get("priority", "NORMAL")
                    
                    with risk_cols[0]:
                        risk_class = "risk-high" if risk_level == "HIGH" else "risk-medium" if risk_level == "MEDIUM" else "risk-low"
                        st.markdown(f"""
                            <div class="metric-card {risk_class}">
                                <div class="metric-label">Churn Risk</div>
                                <div class="metric-value">{churn_prob*100:.0f}%</div>
                                <div class="status-badge status-{risk_level.lower()}" style="margin-top: 1rem;">{risk_level} RISK</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with risk_cols[1]:
                        priority_emoji = "🔴" if priority == "URGENT" else "🟠" if priority == "IMPORTANT" else "🟢"
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Action Priority</div>
                                <div class="metric-value" style="font-size: 3rem; margin: 1rem 0;">{priority_emoji}</div>
                                <div class="metric-subtitle">{priority}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with risk_cols[2]:
                        engagement = result.get("customer_summary", {}).get("engagement_score", 0)
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Engagement Score</div>
                                <div class="metric-value">{int(engagement)}</div>
                                <div class="metric-subtitle">Out of 100</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Gauge Chart
                    st.markdown("#### Risk Visualization")
                    gauge = create_gauge_chart(churn_prob, "Churn Probability")
                    st.plotly_chart(gauge, use_container_width=True)
                    
                    # AI Insights
                    st.markdown("#### 🤖 AI-Powered Insights")
                    insights = result.get("ai_insights", "No insights available")
                    st.markdown(f"""
                        <div class="info-box">
                            <p style="white-space: pre-wrap;">{insights}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Similar Reviews
                    reviews = result.get("similar_reviews", [])
                    if reviews and len(reviews) > 0:
                        st.markdown("#### 💬 Similar Customer Feedback")
                        
                        # Sample diverse reviews
                        sample_reviews = [
                            "Poor customer service experience. Representatives were unhelpful and did not resolve my issue. Very disappointed with the support quality.",
                            "Product quality has declined significantly. Not satisfied with recent purchases and considering switching to competitors.",
                            "Experiencing frequent technical issues with the platform. Support tickets take too long to resolve. Frustrating user experience overall."
                        ]
                        
                        for idx in range(min(3, len(reviews))):
                            review_text = reviews[idx] if idx < len(reviews) else sample_reviews[idx]
                            st.markdown(f"""
                                <div class="review-card">
                                    <div class="review-header">Customer Feedback #{idx + 1}</div>
                                    <div class="review-text">{review_text}</div>
                                </div>
                            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="info-box">
                    <h3>👆 Fill in Customer Details</h3>
                    <p>Enter customer information in the form above, then click "Analyze Customer" to generate a comprehensive risk assessment and receive AI-powered recommendations.</p>
                </div>
            """, unsafe_allow_html=True)
    
    # ================= TAB 2: Batch Processing =================
    with tab2:
        st.markdown("### Batch Customer Processing")
        st.markdown("Upload a CSV file containing customer data to analyze multiple customers simultaneously.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload Customer Data (CSV)",
            type=['csv'],
            help="Upload a CSV file with customer features for batch analysis"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.markdown("#### 📄 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                st.caption(f"Dataset contains {len(df):,} customers with {len(df.columns)} features")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if st.button("🚀 Process All Customers", use_container_width=True):
                    st.markdown("#### ⚙️ Processing Pipeline")
                    
                    results = []
                    progress = st.progress(0)
                    status = st.empty()
                    
                    for idx, row in df.iterrows():
                        payload = {
                            k: (None if pd.isna(v) or np.isinf(v) else float(v))
                            for k, v in row.to_dict().items()
                        }
                        
                        response = requests.post(f"{API_URL}/predict_churn", json=payload)
                        
                        if response.status_code == 200:
                            data = response.json()
                            results.append({
                                "churn_probability": data.get("churn_probability", 0),
                                "risk_level": data.get("risk_level", "UNKNOWN"),
                                "prediction": data.get("prediction", "N/A")
                            })
                        else:
                            results.append({
                                "churn_probability": None,
                                "risk_level": "ERROR",
                                "prediction": "ERROR"
                            })
                        
                        progress.progress((idx + 1) / len(df))
                        status.text(f"Processing customer {idx + 1} of {len(df)}...")
                    
                    results_df = pd.DataFrame(results)
                    final_df = pd.concat([df, results_df], axis=1)
                    
                    status.success("✅ Processing complete!")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Summary
                    st.markdown("#### 📊 Results Summary")
                    
                    cols = st.columns(4)
                    
                    with cols[0]:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Total Customers</div>
                                <div class="metric-value">{len(final_df):,}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with cols[1]:
                        high = (final_df["risk_level"] == "HIGH").sum()
                        st.markdown(f"""
                            <div class="metric-card risk-high">
                                <div class="metric-label">High Risk</div>
                                <div class="metric-value">{high:,}</div>
                                <div class="metric-subtitle">{high/len(final_df)*100:.1f}% of total</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with cols[2]:
                        medium = (final_df["risk_level"] == "MEDIUM").sum()
                        st.markdown(f"""
                            <div class="metric-card risk-medium">
                                <div class="metric-label">Medium Risk</div>
                                <div class="metric-value">{medium:,}</div>
                                <div class="metric-subtitle">{medium/len(final_df)*100:.1f}% of total</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with cols[3]:
                        low = (final_df["risk_level"] == "LOW").sum()
                        st.markdown(f"""
                            <div class="metric-card risk-low">
                                <div class="metric-label">Low Risk</div>
                                <div class="metric-value">{low:,}</div>
                                <div class="metric-subtitle">{low/len(final_df)*100:.1f}% of total</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    
                    # Charts
                    st.markdown("#### 📈 Visual Analytics")
                    
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        risk_counts = final_df["risk_level"].value_counts()
                        fig = go.Figure(data=[go.Pie(
                            labels=risk_counts.index,
                            values=risk_counts.values,
                            hole=0.5,
                            marker=dict(
                                colors=['#ff0000', '#ff7900', '#00ff00'],
                                line=dict(color='#000000', width=2)
                            ),
                            textfont=dict(size=14, color='#ffffff', family='Inter', weight=600)
                        )])
                        fig.update_layout(
                            title={
                                'text': "Risk Distribution",
                                'font': {'size': 18, 'color': '#ffffff', 'family': 'Inter'}
                            },
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#ffffff', family='Inter'),
                            height=380,
                            showlegend=True,
                            legend=dict(
                                font=dict(size=12),
                                bgcolor='rgba(0,0,0,0.5)',
                                bordercolor='#333333',
                                borderwidth=1
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with chart_col2:
                        fig = go.Figure(data=[go.Histogram(
                            x=final_df["churn_probability"],
                            nbinsx=20,
                            marker=dict(
                                color='#f16f01',
                                line=dict(color='#000000', width=1)
                            )
                        )])
                        fig.update_layout(
                            title={
                                'text': "Churn Probability Distribution",
                                'font': {'size': 18, 'color': '#ffffff', 'family': 'Inter'}
                            },
                            xaxis_title="Probability",
                            yaxis_title="Count",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(10,10,10,1)',
                            font=dict(color='#ffffff', family='Inter'),
                            height=380,
                            xaxis=dict(gridcolor='#333333'),
                            yaxis=dict(gridcolor='#333333')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Full Results
                    st.markdown("#### 📋 Complete Results")
                    st.dataframe(final_df, use_container_width=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Download
                    csv = final_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"churn_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
        else:
            st.markdown("""
                <div class="info-box">
                    <h3>📁 Upload Your Data</h3>
                    <p>Upload a CSV file containing customer data to perform batch churn prediction analysis. The system will process each customer and provide comprehensive risk assessments.</p>
                </div>
            """, unsafe_allow_html=True)
    
    # ================= TAB 3: Analytics =================
    with tab3:
        st.markdown("### Analytics Dashboard")
        st.markdown("Comprehensive analytics and insights based on processed customer data.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        try:
            data = pd.read_csv("artifacts/processed_data.csv")
            
            st.success(f"✅ Successfully loaded {len(data):,} customer records")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # KPIs
            st.markdown("#### 📊 Key Performance Indicators")
            
            cols = st.columns(4)
            
            with cols[0]:
                churn = data["churn_label"].mean() * 100
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Overall Churn Rate</div>
                        <div class="metric-value">{churn:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with cols[1]:
                rating = data["avg_rating"].mean()
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Average Rating</div>
                        <div class="metric-value">{rating:.2f}</div>
                        <div class="metric-subtitle">Out of 5.0</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with cols[2]:
                conv = data["conversion_rate"].mean() * 100
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Conversion Rate</div>
                        <div class="metric-value">{conv:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with cols[3]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Customers</div>
                        <div class="metric-value">{len(data):,}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # Charts
            st.markdown("#### 📈 Visual Analytics")
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                data["age_group"] = pd.cut(
                    data["Age"],
                    bins=[0, 25, 35, 45, 55, 100],
                    labels=["18-25", "26-35", "36-45", "46-55", "55+"]
                )
                
                churn_age = data.groupby("age_group")["churn_label"].mean() * 100
                
                fig = go.Figure(data=[go.Bar(
                    x=churn_age.index.astype(str),
                    y=churn_age.values,
                    marker=dict(
                        color=churn_age.values,
                        colorscale=[[0, '#00ff00'], [0.5, '#ff7900'], [1, '#ff0000']],
                        line=dict(color='#000000', width=1)
                    ),
                    text=[f"{v:.1f}%" for v in churn_age.values],
                    textposition='outside',
                    textfont=dict(size=13, color='#ffffff', family='Inter')
                )])
                
                fig.update_layout(
                    title={
                        'text': "Churn Rate by Age Group",
                        'font': {'size': 18, 'color': '#ffffff', 'family': 'Inter'}
                    },
                    xaxis_title="Age Group",
                    yaxis_title="Churn Rate (%)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(10,10,10,1)',
                    font=dict(color='#ffffff', family='Inter'),
                    height=420,
                    xaxis=dict(gridcolor='#333333'),
                    yaxis=dict(gridcolor='#333333')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with chart_col2:
                if "customer_segment" in data.columns:
                    seg = data["customer_segment"].value_counts()
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=[f"Segment {i}" for i in seg.index],
                        values=seg.values,
                        hole=0.5,
                        marker=dict(
                            colors=['#f16f01', '#ff7900', '#ff9933', '#ffb366'],
                            line=dict(color='#000000', width=2)
                        ),
                        textfont=dict(size=14, color='#ffffff', family='Inter', weight=600)
                    )])
                    
                    fig.update_layout(
                        title={
                            'text': "Customer Segmentation",
                            'font': {'size': 18, 'color': '#ffffff', 'family': 'Inter'}
                        },
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#ffffff', family='Inter'),
                        height=420,
                        showlegend=True,
                        legend=dict(
                            font=dict(size=12),
                            bgcolor='rgba(0,0,0,0.5)',
                            bordercolor='#333333',
                            borderwidth=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Correlation
            st.markdown("#### 🔥 Feature Correlation Analysis")
            st.markdown("Correlation coefficients between customer features and churn probability")
            
            features = [
                "avg_rating", "negative_reviews", "conversion_rate",
                "dropoff_rate", "total_actions", "sentiment_ratio"
            ]
            
            corr = data[features + ["churn_label"]].corr()["churn_label"].drop("churn_label").sort_values()
            
            fig = go.Figure(data=[go.Bar(
                x=corr.values,
                y=corr.index,
                orientation='h',
                marker=dict(
                    color=corr.values,
                    colorscale=[[0, '#00ff00'], [0.5, '#ff7900'], [1, '#ff0000']],
                    line=dict(color='#000000', width=1)
                ),
                text=[f"{v:.3f}" for v in corr.values],
                textposition='outside',
                textfont=dict(size=13, color='#ffffff', family='Inter')
            )])
            
            fig.update_layout(
                title={
                    'text': "Feature Correlation with Churn",
                    'font': {'size': 18, 'color': '#ffffff', 'family': 'Inter'}
                },
                xaxis_title="Correlation Coefficient",
                yaxis_title="",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(10,10,10,1)',
                font=dict(color='#ffffff', family='Inter'),
                height=520,
                xaxis=dict(gridcolor='#333333'),
                yaxis=dict(gridcolor='#333333')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except FileNotFoundError:
            st.markdown("""
                <div class="info-box warning-box">
                    <h3>⚠️ Analytics Data Not Available</h3>
                    <p>Processed customer data not found. Please run the training pipeline first:</p>
                    <p><code style="color: #ff7900; font-weight: 600; font-size: 1rem;">python backend.py --train</code></p>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error loading analytics: {str(e)}")


if __name__ == "__main__":
    main()