import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime

# ================= CONFIG =================
st.set_page_config(
    page_title="AI Customer Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "https://mahmoud-alkodousy--customer-intelligence-api-fastapi-app.modal.run"

# ================= CUSTOM CSS =================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .risk-medium {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #333;
    }
    .risk-low {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
    }
    .insight-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        color: #333333;
    }
    .insight-box h4 {
        color: #1f1f1f;
        margin-bottom: 0.5rem;
    }
    .insight-box p {
        color: #333333;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ================= HELPER FUNCTIONS =================
def check_api_health():
    """Check if API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") == "healthy", data.get("models_loaded", {})
        return False, {}
    except Exception as e:
        st.error(f"❌ Cannot connect to API: {str(e)}")
        return False, {}

def predict_churn(customer_data):
    """Get churn prediction from API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict_churn",
            json=customer_data,
            timeout=30
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            error_detail = response.json().get("detail", "Unknown error")
            return False, {"error": error_detail}
            
    except Exception as e:
        return False, {"error": str(e)}

def analyze_customer(customer_data):
    """Get full customer analysis with AI insights"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze_customer",
            json=customer_data,
            timeout=60
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            error_detail = response.json().get("detail", "Unknown error")
            return False, {"error": error_detail}
            
    except Exception as e:
        return False, {"error": str(e)}

# ================= MAIN APP =================
def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 AI Customer Intelligence Platform</h1>', unsafe_allow_html=True)
    st.markdown("### Predict customer churn and get actionable AI insights")
    
    # Check API Health
    with st.sidebar:
        st.header("🔌 System Status")
        
        is_healthy, models_loaded = check_api_health()
        
        if is_healthy:
            st.success("✅ API is online")
            
            st.write("**Models Loaded:**")
            for model, status in models_loaded.items():
                icon = "✅" if status else "❌"
                st.write(f"{icon} {model}")
        else:
            st.error("❌ API is offline")
            st.info("Please ensure the Modal API is deployed and running.")
            st.stop()
        
        st.divider()
        
        st.header("📊 About")
        st.info("""
        This platform uses advanced ML models to:
        - Predict customer churn probability
        - Identify risk factors
        - Generate AI-powered insights
        - Recommend retention actions
        """)
    
    # Main Content
    tabs = st.tabs(["🔮 Churn Prediction", "🧠 AI Analysis", "📈 Insights Dashboard"])
    
    # =================== TAB 1: CHURN PREDICTION ===================
    with tabs[0]:
        st.header("Customer Churn Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("👤 Demographics")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
        
        with col2:
            st.subheader("📊 Engagement Metrics")
            total_actions = st.number_input("Total Actions", min_value=0, value=50)
            avg_duration = st.number_input("Avg Duration (min)", min_value=0.0, value=5.2, step=0.1)
            total_views = st.number_input("Total Views", min_value=0, value=100)
            total_clicks = st.number_input("Total Clicks", min_value=0, value=25)
        
        with col3:
            st.subheader("⭐ Satisfaction")
            avg_rating = st.slider("Average Rating", min_value=1.0, max_value=5.0, value=3.5, step=0.1)
            negative_reviews = st.number_input("Negative Reviews", min_value=0, value=1)
            positive_reviews = st.number_input("Positive Reviews", min_value=0, value=3)
            total_reviews = st.number_input("Total Reviews", min_value=0, value=4)
        
        st.divider()
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("💰 Product Metrics")
            avg_product_price = st.number_input("Avg Product Price ($)", min_value=0.0, value=50.0, step=1.0)
            unique_products_viewed = st.number_input("Unique Products Viewed", min_value=0, value=10)
        
        with col_b:
            st.subheader("📅 Activity")
            days_active = st.number_input("Days Active", min_value=0, value=30)
            conversion_rate = st.slider("Conversion Rate", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
            dropoff_rate = st.slider("Drop-off Rate", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        
        # Calculate derived metrics
        click_through_rate = total_clicks / total_views if total_views > 0 else 0
        sentiment_ratio = positive_reviews / total_reviews if total_reviews > 0 else 0
        
        # Prepare data
        customer_data = {
            "Age": float(age),
            "total_actions": float(total_actions),
            "avg_duration": float(avg_duration),
            "avg_rating": float(avg_rating),
            "negative_reviews": float(negative_reviews),
            "positive_reviews": float(positive_reviews),
            "total_views": float(total_views),
            "total_clicks": float(total_clicks),
            "conversion_rate": float(conversion_rate),
            "dropoff_rate": float(dropoff_rate),
            "click_through_rate": float(click_through_rate),
            "sentiment_ratio": float(sentiment_ratio),
            "avg_product_price": float(avg_product_price),
            "unique_products_viewed": float(unique_products_viewed),
            "days_active": float(days_active),
            "total_reviews": float(total_reviews)
        }
        
        st.divider()
        
        if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
            with st.spinner("Analyzing customer data..."):
                success, result = predict_churn(customer_data)
                
                if success:
                    churn_prob = result.get('churn_probability', 0)
                    risk_level = result.get('risk_level', 'UNKNOWN')
                    prediction = result.get('prediction', 'Unknown')
                    
                    # Display Results
                    st.success("✅ Prediction Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Churn Probability</h3>
                            <h1>{churn_prob:.1%}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        risk_class = f"risk-{risk_level.lower()}"
                        st.markdown(f"""
                        <div class="metric-card {risk_class}">
                            <h3>Risk Level</h3>
                            <h1>{risk_level}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Status</h3>
                            <h1>{prediction}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Gauge Chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=churn_prob * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Churn Risk Score"},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 40], 'color': "lightgreen"},
                                {'range': [40, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
    
    # =================== TAB 2: AI ANALYSIS ===================
    with tabs[1]:
        st.header("🧠 Advanced AI Customer Analysis")
        st.info("Get deep insights powered by AI, including similar customer patterns and actionable recommendations")
        
        if st.button("🚀 Run Full AI Analysis", type="primary", use_container_width=True):
            with st.spinner("🤖 Running comprehensive AI analysis... This may take a minute..."):
                success, result = analyze_customer(customer_data)
                
                if success:
                    # Extract data
                    churn_prob = result.get('churn_probability', 0)
                    risk_level = result.get('risk_level', 'UNKNOWN')
                    priority = result.get('priority', 'NORMAL')
                    similar_reviews = result.get('similar_reviews', [])
                    ai_insights = result.get('ai_insights', {})
                    customer_summary = result.get('customer_summary', {})
                    
                    st.success("✅ Analysis Complete!")
                    
                    # Risk Overview
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Churn Risk", f"{churn_prob:.1%}", 
                                 delta=f"{churn_prob - 0.5:.1%}" if churn_prob > 0.5 else None,
                                 delta_color="inverse")
                    
                    with col2:
                        st.metric("Risk Level", risk_level)
                    
                    with col3:
                        st.metric("Priority", priority)
                    
                    with col4:
                        engagement = customer_summary.get('engagement_score', 0)
                        st.metric("Engagement Score", f"{engagement:.0f}")
                    
                    st.divider()
                    
                    # AI Insights
                    if ai_insights:
                        st.subheader("🎯 AI-Generated Insights")
                        
                        summary = ai_insights.get('summary', 'No summary available')
                        st.markdown(f"""
                        <div class="insight-box">
                            <h4 style="color: #1f1f1f;">📝 Executive Summary</h4>
                            <p style="color: #333333; line-height: 1.6;">{summary}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col_left, col_right = st.columns(2)
                        
                        with col_left:
                            st.markdown("### 🔍 Root Causes")
                            root_causes = ai_insights.get('root_causes', [])
                            for i, cause in enumerate(root_causes, 1):
                                st.warning(f"**{i}.** {cause}")
                        
                        with col_right:
                            st.markdown("### 💡 Recommendations")
                            recommendations = ai_insights.get('recommendations', [])
                            for i, rec in enumerate(recommendations, 1):
                                st.success(f"**{i}.** {rec}")
                    
                    st.divider()
                    
                    # Similar Customer Reviews
                    if similar_reviews:
                        st.subheader("👥 Similar Customer Feedback")
                        st.caption("Reviews from customers with similar patterns")
                        
                        for i, review in enumerate(similar_reviews, 1):
                            rating = review.get('rating', 0)
                            text = review.get('text', 'No text available')
                            
                            stars = "⭐" * int(rating)
                            
                            st.markdown(f"""
                            <div class="insight-box">
                                <p style="color: #1f1f1f;"><strong>Review {i}</strong> {stars} ({rating}/5)</p>
                                <p style="font-style: italic; color: #555555;">"{text}..."</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                else:
                    st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                    st.info("💡 Make sure the API models are fully loaded. Check the sidebar status.")
    
    # =================== TAB 3: INSIGHTS DASHBOARD ===================
    with tabs[2]:
        st.header("📈 Customer Insights Dashboard")
        
        # Create sample data for visualization
        metrics_df = pd.DataFrame({
            'Metric': ['Engagement', 'Satisfaction', 'Conversion', 'Retention'],
            'Score': [
                min(total_actions / 100, 1.0),
                avg_rating / 5.0,
                conversion_rate,
                1.0 - dropoff_rate
            ]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig1 = px.bar(
                metrics_df,
                x='Metric',
                y='Score',
                title='Customer Health Metrics',
                color='Score',
                color_continuous_scale='RdYlGn'
            )
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Radar chart
            fig2 = go.Figure()
            fig2.add_trace(go.Scatterpolar(
                r=metrics_df['Score'],
                theta=metrics_df['Metric'],
                fill='toself',
                name='Customer Profile'
            ))
            fig2.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title='Customer Profile Overview'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Activity Timeline
        st.subheader("📅 Activity Overview")
        
        activity_data = pd.DataFrame({
            'Category': ['Views', 'Clicks', 'Actions', 'Reviews'],
            'Count': [total_views, total_clicks, total_actions, total_reviews]
        })
        
        fig3 = px.pie(
            activity_data,
            values='Count',
            names='Category',
            title='Activity Distribution'
        )
        st.plotly_chart(fig3, use_container_width=True)

if __name__ == "__main__":
    main()
