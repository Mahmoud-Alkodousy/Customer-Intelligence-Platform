import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# ================= CONFIG =================
# Get API URL from environment variable or use default
API_URL = os.environ.get('API_URL', 'http://localhost:8000')

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= POLISHED PROFESSIONAL CSS =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main {
        background-color: #000000;
        padding-top: 1rem;
    }
    
    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #f16f01;
    }
    
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
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #f16f01 0%, #ff7900 100%);
        color: #ffffff;
        border: none;
        padding: 0.85rem 1.75rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff7900 0%, #f16f01 100%);
        box-shadow: 0 6px 20px rgba(241, 111, 1, 0.4);
        transform: translateY(-2px);
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(241, 111, 1, 0.05) 0%, rgba(0, 0, 0, 0) 100%);
        border: 1px solid #f16f01;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
    }
    
    .info-box h3 {
        color: #f16f01 !important;
        margin-bottom: 1rem !important;
    }
    
    .info-box p {
        color: #ffffff;
        opacity: 0.8;
        line-height: 1.6;
        margin-bottom: 0.75rem;
    }
    
    .warning-box {
        border-color: #ff7900;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: #000000;
        border-bottom: 2px solid #333333;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3.5rem;
        padding: 0 2.5rem;
        font-size: 1rem;
        font-weight: 600;
        background-color: transparent;
        color: #ffffff;
        opacity: 0.5;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        opacity: 0.8;
        background-color: rgba(241, 111, 1, 0.05);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid #f16f01;
        color: #f16f01;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)


# ================= API HELPER FUNCTIONS =================
def check_api_status():
    """Check if API is reachable"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, None


def predict_churn_api(customer_data):
    """Call API for churn prediction"""
    try:
        response = requests.post(
            f"{API_URL}/predict_churn",
            json=customer_data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None


def analyze_customer_api(customer_data):
    """Call API for full customer analysis"""
    try:
        response = requests.post(
            f"{API_URL}/analyze_customer",
            json=customer_data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None


# ================= MAIN APP =================
def main():
    # Header
    st.markdown("""
        <h1 class="main-header">
            AI Customer Intelligence <span class="accent">Platform</span>
        </h1>
        <p class="sub-header">
            Advanced machine learning analytics for customer behavior prediction and retention optimization
        </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ System Controls")
        
        # API Status
        st.markdown("#### API Status")
        api_online, health_data = check_api_status()
        
        if api_online:
            st.markdown('<div class="status-badge status-online">✓ API Online</div>', unsafe_allow_html=True)
            
            if health_data and 'models_loaded' in health_data:
                models = health_data['models_loaded']
                st.markdown(f"""
                <div style="margin-top: 1rem; font-size: 0.85rem; color: #ffffff; opacity: 0.7;">
                    Churn Model: {'✓' if models.get('churn_model') else '✗'}<br>
                    Embedder: {'✓' if models.get('embedder') else '✗'}<br>
                    Vector Store: {'✓' if models.get('vector_store') else '✗'}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-offline">✗ API Offline</div>', unsafe_allow_html=True)
            st.warning(f"Cannot connect to API at: {API_URL}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        
        # API URL Config
        st.markdown("#### ⚙️ Configuration")
        st.text_input(
            "API URL",
            value=API_URL,
            disabled=True,
            help="Set API_URL environment variable to change"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("#### 📊 About")
        st.markdown("""
        This dashboard connects to the AI Customer Intelligence API to:
        - Predict customer churn probability
        - Provide AI-powered insights
        - Analyze customer behavior
        - Recommend retention strategies
        """)
    
    # Main Tabs
    tab1, tab2 = st.tabs(["🎯 Single Prediction", "📊 Batch Analysis"])
    
    # ================= TAB 1: Single Prediction =================
    with tab1:
        st.markdown("### Single Customer Analysis")
        st.markdown("Enter customer data to get real-time churn prediction and AI-powered recommendations.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if not api_online:
            st.error("⚠️ API is offline. Please check the connection and try again.")
        else:
            # Input Form
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Customer Profile")
                age = st.number_input("Age", min_value=18, max_value=100, value=30)
                total_actions = st.number_input("Total Actions", min_value=0, value=10)
                avg_duration = st.number_input("Avg Duration (min)", min_value=0.0, value=5.0, step=0.5)
                unique_products = st.number_input("Unique Products Viewed", min_value=0, value=3)
                days_active = st.number_input("Days Active", min_value=0, value=5)
            
            with col2:
                st.markdown("#### Engagement Metrics")
                total_views = st.number_input("Total Views", min_value=0, value=20)
                total_clicks = st.number_input("Total Clicks", min_value=0, value=10)
                conversion_rate = st.slider("Conversion Rate", 0.0, 1.0, 0.2, 0.01)
                dropoff_rate = st.slider("Drop-off Rate", 0.0, 1.0, 0.3, 0.01)
                ctr = st.slider("Click-Through Rate", 0.0, 1.0, 0.5, 0.01)
            
            with col3:
                st.markdown("#### Satisfaction Metrics")
                avg_rating = st.slider("Average Rating", 1.0, 5.0, 3.5, 0.1)
                total_reviews = st.number_input("Total Reviews", min_value=0, value=3)
                negative_reviews = st.number_input("Negative Reviews", min_value=0, value=1)
                positive_reviews = st.number_input("Positive Reviews", min_value=0, value=2)
                sentiment_ratio = st.slider("Sentiment Ratio", 0.0, 1.0, 0.67, 0.01)
                avg_product_price = st.number_input("Avg Product Price ($)", min_value=0.0, value=50.0, step=5.0)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Analysis Type Selection
            analysis_type = st.radio(
                "Analysis Type",
                ["Quick Prediction", "Full Analysis with AI Insights"],
                horizontal=True
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🔍 Analyze Customer", use_container_width=True):
                customer_data = {
                    'Age': age,
                    'total_actions': total_actions,
                    'avg_duration': avg_duration,
                    'avg_rating': avg_rating,
                    'negative_reviews': negative_reviews,
                    'positive_reviews': positive_reviews,
                    'total_views': total_views,
                    'total_clicks': total_clicks,
                    'conversion_rate': conversion_rate,
                    'dropoff_rate': dropoff_rate,
                    'click_through_rate': ctr,
                    'sentiment_ratio': sentiment_ratio,
                    'avg_product_price': avg_product_price,
                    'unique_products_viewed': unique_products,
                    'days_active': days_active,
                    'total_reviews': total_reviews
                }
                
                with st.spinner("Analyzing customer profile..."):
                    if analysis_type == "Quick Prediction":
                        result = predict_churn_api(customer_data)
                    else:
                        result = analyze_customer_api(customer_data)
                    
                    if result:
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Display Results
                        st.markdown("#### 📊 Analysis Results")
                        
                        result_col1, result_col2, result_col3 = st.columns(3)
                        
                        with result_col1:
                            churn_prob = result['churn_probability']
                            risk_level = result['risk_level']
                            risk_class = f"risk-{risk_level.lower()}"
                            st.markdown(f"""
                                <div class="metric-card {risk_class}">
                                    <div class="metric-label">Churn Probability</div>
                                    <div class="metric-value">{churn_prob:.1%}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with result_col2:
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">Risk Level</div>
                                    <div class="metric-value">{risk_level}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with result_col3:
                            prediction = result['prediction']
                            pred_emoji = "🔴" if "Risk" in prediction else "✅"
                            st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">Prediction</div>
                                    <div class="metric-value" style="font-size: 1.5rem;">{prediction} {pred_emoji}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # AI Insights (if full analysis)
                        if analysis_type == "Full Analysis with AI Insights" and 'ai_insights' in result:
                            st.markdown("<br><br>", unsafe_allow_html=True)
                            st.markdown("#### 🤖 AI-Powered Insights")
                            
                            insights = result['ai_insights']
                            
                            # Summary
                            st.markdown(f"**Summary:** {insights.get('summary', 'N/A')}")
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            insight_col1, insight_col2 = st.columns(2)
                            
                            with insight_col1:
                                st.markdown("**🔍 Root Causes:**")
                                for cause in insights.get('root_causes', []):
                                    st.markdown(f"- {cause}")
                            
                            with insight_col2:
                                st.markdown("**💡 Recommendations:**")
                                for rec in insights.get('recommendations', []):
                                    st.markdown(f"- {rec}")
                            
                            # Similar Reviews
                            if 'similar_reviews' in result:
                                st.markdown("<br>", unsafe_allow_html=True)
                                st.markdown("**📝 Similar Customer Feedback:**")
                                for i, review in enumerate(result['similar_reviews'][:3], 1):
                                    st.markdown(f"{i}. ⭐ {review['rating']:.1f} - {review['text']}")
    
    # ================= TAB 2: Batch Analysis =================
    with tab2:
        st.markdown("### Batch Customer Analysis")
        st.markdown("Upload a CSV file with customer data for bulk churn prediction.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if not api_online:
            st.error("⚠️ API is offline. Please check the connection and try again.")
        else:
            uploaded_file = st.file_uploader(
                "Upload CSV file with customer data",
                type=['csv'],
                help="File should contain columns matching the API input schema"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    st.success(f"✅ Loaded {len(df)} customers")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    if st.button("🚀 Run Batch Analysis", use_container_width=True):
                        with st.spinner("Processing customers..."):
                            results = []
                            
                            progress_bar = st.progress(0)
                            
                            for idx, row in df.iterrows():
                                customer_data = row.to_dict()
                                result = predict_churn_api(customer_data)
                                
                                if result:
                                    results.append({
                                        'CustomerID': row.get('CustomerID', idx),
                                        'Churn_Probability': result['churn_probability'],
                                        'Risk_Level': result['risk_level'],
                                        'Prediction': result['prediction']
                                    })
                                
                                progress_bar.progress((idx + 1) / len(df))
                            
                            final_df = pd.DataFrame(results)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Summary Stats
                            st.markdown("#### 📊 Batch Analysis Summary")
                            
                            sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
                            
                            with sum_col1:
                                high_risk = (final_df['Risk_Level'] == 'HIGH').sum()
                                st.markdown(f"""
                                    <div class="metric-card risk-high">
                                        <div class="metric-label">High Risk</div>
                                        <div class="metric-value">{high_risk}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with sum_col2:
                                medium_risk = (final_df['Risk_Level'] == 'MEDIUM').sum()
                                st.markdown(f"""
                                    <div class="metric-card risk-medium">
                                        <div class="metric-label">Medium Risk</div>
                                        <div class="metric-value">{medium_risk}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with sum_col3:
                                low_risk = (final_df['Risk_Level'] == 'LOW').sum()
                                st.markdown(f"""
                                    <div class="metric-card risk-low">
                                        <div class="metric-label">Low Risk</div>
                                        <div class="metric-value">{low_risk}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with sum_col4:
                                avg_prob = final_df['Churn_Probability'].mean()
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-label">Avg Churn Prob</div>
                                        <div class="metric-value">{avg_prob:.1%}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("<br><br>", unsafe_allow_html=True)
                            
                            # Distribution Chart
                            st.markdown("#### 📊 Churn Probability Distribution")
                            
                            fig = go.Figure(data=[go.Histogram(
                                x=final_df['Churn_Probability'],
                                nbinsx=30,
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
                        <p>Upload a CSV file containing customer data to perform batch churn prediction analysis.</p>
                        <p>Required columns: Age, total_actions, avg_duration, avg_rating, negative_reviews, positive_reviews, total_views, total_clicks, conversion_rate, dropoff_rate, click_through_rate, sentiment_ratio, avg_product_price, unique_products_viewed, days_active, total_reviews</p>
                    </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
