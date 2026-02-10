import os
import json
import pickle
from typing import Dict, Any, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, classification_report

from sentence_transformers import SentenceTransformer
import chromadb
import requests

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import time


# ================= CONFIG =================
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "openai/gpt-4o-mini"

ARTIFACTS_DIR = "./artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ================= DATA LOADING =================
@st.cache_data
def load_data():
    """تحميل جميع ملفات البيانات"""
    try:
        customers = pd.read_csv("data/customers.csv")
        journey = pd.read_csv("data/customer_journey.csv")
        reviews = pd.read_csv("data/customer_reviews_with_analysis.csv")
        engagement = pd.read_csv("data/engagement_data.csv")
        geography = pd.read_csv("data/geography.csv")
        products = pd.read_csv("data/products.csv")
        
        return customers, journey, reviews, engagement, geography, products
    except Exception as e:
        st.error(f"❌ Error loading data files: {str(e)}")
        return None, None, None, None, None, None


# ================= ENHANCED FEATURE ENGINEERING =================
def build_features(customers, journey, reviews, engagement, geography, products):
    """بناء features متقدمة للتحليل"""
    
    # -------- Journey Features --------
    journey_agg = journey.groupby("CustomerID").agg(
        total_actions=("Action", "count"),
        avg_duration=("Duration", "mean"),
        homepage_visits=("Stage", lambda x: (x == "Homepage").sum()),
        product_page_visits=("Stage", lambda x: (x == "ProductPage").sum()),
        checkout_attempts=("Stage", lambda x: (x == "Checkout").sum()),
        dropoff_count=("Action", lambda x: (x == "Drop-off").sum()),
        unique_products_viewed=("ProductID", "nunique"),
        days_active=("VisitDate", lambda x: pd.to_datetime(x).nunique())
    ).reset_index()
    
    # Conversion metrics
    journey_agg["conversion_rate"] = (
        journey_agg["checkout_attempts"] / journey_agg["total_actions"].replace(0, 1)
    )
    journey_agg["dropoff_rate"] = (
        journey_agg["dropoff_count"] / journey_agg["total_actions"].replace(0, 1)
    )
    
    # -------- Review Features --------
    review_agg = reviews.groupby("CustomerID").agg(
        total_reviews=("ReviewID", "count"),
        avg_rating=("Rating", "mean"),
        min_rating=("Rating", "min"),
        max_rating=("Rating", "max"),
        rating_std=("Rating", "std"),
        negative_reviews=("review_type", lambda x: (x == 0).sum()),
        positive_reviews=("review_type", lambda x: (x == 1).sum()),
        quality_complaints=("problem_type", lambda x: (x == "Product Quality/Performance").sum()),
        price_complaints=("problem_type", lambda x: (x == "Price/Value").sum()),
    ).reset_index()
    
    review_agg["sentiment_ratio"] = (
        review_agg["positive_reviews"] / review_agg["total_reviews"].replace(0, 1)
    )
    
    # -------- Engagement Features --------
    # Split Views-Clicks properly
    engagement[["Views", "Clicks"]] = engagement["ViewsClicksCombined"].astype(str).str.split("-", expand=True).astype(float)
    
    # Map ProductID to CustomerID via journey
    product_customer_map = journey[["ProductID", "CustomerID"]].drop_duplicates()
    engagement_with_customer = engagement.merge(product_customer_map, on="ProductID", how="left")
    
    engagement_agg = engagement_with_customer.groupby("CustomerID").agg(
        total_views=("Views", "sum"),
        total_clicks=("Clicks", "sum"),
        total_likes=("Likes", "sum"),
        engagement_events=("EngagementID", "count"),
        blog_engagement=("ContentType", lambda x: x.str.lower().str.contains("blog", na=False).sum()),
        video_engagement=("ContentType", lambda x: x.str.lower().str.contains("video", na=False).sum()),
        social_engagement=("ContentType", lambda x: x.str.lower().str.contains("social", na=False).sum()),
        unique_campaigns=("CampaignID", "nunique")
    ).reset_index()
    
    engagement_agg["click_through_rate"] = (
        engagement_agg["total_clicks"] / engagement_agg["total_views"].replace(0, 1)
    )
    engagement_agg["engagement_per_event"] = (
        engagement_agg["total_likes"] / engagement_agg["engagement_events"].replace(0, 1)
    )
    
    # -------- Product Value Features --------
    journey_with_price = journey.merge(products[["ProductID", "Price"]], on="ProductID", how="left")
    product_agg = journey_with_price.groupby("CustomerID").agg(
        avg_product_price=("Price", "mean"),
        total_product_value=("Price", "sum"),
        max_product_price=("Price", "max")
    ).reset_index()
    
    # -------- Merge all features --------
    df = customers.copy()
    df = df.merge(geography, on="GeographyID", how="left")
    df = df.merge(journey_agg, on="CustomerID", how="left")
    df = df.merge(review_agg, on="CustomerID", how="left")
    df = df.merge(engagement_agg, on="CustomerID", how="left")
    df = df.merge(product_agg, on="CustomerID", how="left")
    
    # Fill missing values
    df = df.fillna(0)
    
    # -------- Churn Label (Enhanced) --------
    df["churn_label"] = (
        ((df["negative_reviews"] >= 2) | 
         (df["dropoff_rate"] > 0.5) | 
         (df["avg_rating"] < 2.5) |
         ((df["sentiment_ratio"] < 0.3) & (df["total_reviews"] > 2)))
    ).astype(int)
    
    return df


# ================= CUSTOMER SEGMENTATION =================
def perform_segmentation(df, n_clusters=4):
    """تقسيم العملاء إلى شرائح"""
    
    segment_features = [
        "Age", "total_actions", "avg_rating", "total_views", 
        "total_clicks", "avg_product_price", "conversion_rate"
    ]
    
    X_segment = df[segment_features].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_segment)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["customer_segment"] = kmeans.fit_predict(X_scaled)
    
    # Save artifacts
    with open(os.path.join(ARTIFACTS_DIR, "segmentation_model.pkl"), "wb") as f:
        pickle.dump((kmeans, scaler, segment_features), f)
    
    return df


# ================= TRAIN CHURN MODEL =================
def train_churn_model(df):
    """تدريب نموذج التنبؤ بالـ Churn"""
    
    feature_cols = [
        "Age", "total_actions", "avg_duration", "avg_rating", 
        "negative_reviews", "positive_reviews", "total_views", 
        "total_clicks", "conversion_rate", "dropoff_rate",
        "click_through_rate", "sentiment_ratio", "avg_product_price",
        "unique_products_viewed", "days_active", "total_reviews"
    ]
    
    X = df[feature_cols].fillna(0)
    y = df["churn_label"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = GradientBoostingClassifier(
        n_estimators=200, 
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    test_preds = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_preds)
    
    # Save model
    with open(os.path.join(ARTIFACTS_DIR, "churn_model.pkl"), "wb") as f:
        pickle.dump((model, feature_cols), f)
    
    return model, test_auc, feature_cols


# ================= VECTOR STORE =================
def build_vector_store(reviews):
    """بناء Vector Store للـ Reviews"""
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Save embedder
    with open(os.path.join(ARTIFACTS_DIR, "embedder.pkl"), "wb") as f:
        pickle.dump(embedder, f)
    
    # Create ChromaDB collection
    chroma_dir = os.path.join(ARTIFACTS_DIR, "chroma_db")
    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    
    # Delete old collection if exists
    try:
        chroma_client.delete_collection("reviews")
    except:
        pass
    
    collection = chroma_client.create_collection("reviews")
    
    # Embed and store reviews
    texts = reviews["ReviewText"].fillna("").astype(str).tolist()
    embeddings = embedder.encode(texts, show_progress_bar=True)
    
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=[str(i) for i in range(len(texts))],
        metadatas=[
            {
                "rating": float(reviews.iloc[i]["Rating"]),
                "customer_id": str(reviews.iloc[i]["CustomerID"])
            }
            for i in range(len(reviews))
        ]
    )
    
    return embedder, collection


# ================= AI INSIGHTS =================
def retrieve_similar_reviews(query, embedder, collection, k=5):
    """البحث عن reviews مشابهة"""
    query_embedding = embedder.encode([query])[0]
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )
    
    similar = []
    for i, doc in enumerate(results['documents'][0]):
        similar.append({
            "text": doc[:200],
            "rating": results['metadatas'][0][i]['rating']
        })
    
    return similar


def generate_ai_insights(customer_data, churn_prob, similar_reviews):
    """توليد AI Insights باستخدام LLM"""
    
    if not OPENROUTER_API_KEY:
        return {
            "summary": "API Key not configured",
            "root_causes": [],
            "recommendations": []
        }
    
    prompt = f"""
You are an AI customer intelligence analyst. Analyze this customer profile and provide actionable insights.

**Customer Profile:**
- Age: {customer_data.get('Age', 'N/A')}
- Engagement Score: {customer_data.get('total_actions', 0)}
- Average Rating: {customer_data.get('avg_rating', 0):.2f}
- Negative Reviews: {customer_data.get('negative_reviews', 0)}
- Conversion Rate: {customer_data.get('conversion_rate', 0):.2%}
- Churn Probability: {churn_prob:.2%}

**Similar Customer Reviews:**
{chr(10).join([f"- Rating {r['rating']}: {r['text']}" for r in similar_reviews[:3]])}

Provide:
1. Brief summary (2-3 sentences)
2. Top 3 root causes for potential churn
3. Top 3 specific action recommendations

Format as JSON:
{{
  "summary": "...",
  "root_causes": ["...", "...", "..."],
  "recommendations": ["...", "...", "..."]
}}
"""
    
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            
            # Extract JSON from markdown if needed
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            return json.loads(content)
        else:
            return {
                "summary": "Error generating insights",
                "root_causes": [],
                "recommendations": []
            }
    except Exception as e:
        return {
            "summary": f"Error: {str(e)}",
            "root_causes": [],
            "recommendations": []
        }


# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    """تحميل النماذج المدربة"""
    models = {
        'churn_model': None,
        'feature_cols': None,
        'embedder': None,
        'collection': None
    }
    
    try:
        # Load churn model
        if os.path.exists(os.path.join(ARTIFACTS_DIR, "churn_model.pkl")):
            with open(os.path.join(ARTIFACTS_DIR, "churn_model.pkl"), "rb") as f:
                models['churn_model'], models['feature_cols'] = pickle.load(f)
        
        # Load embedder
        if os.path.exists(os.path.join(ARTIFACTS_DIR, "embedder.pkl")):
            with open(os.path.join(ARTIFACTS_DIR, "embedder.pkl"), "rb") as f:
                models['embedder'] = pickle.load(f)
        
        # Load ChromaDB
        chroma_dir = os.path.join(ARTIFACTS_DIR, "chroma_db")
        if os.path.exists(chroma_dir):
            chroma_client = chromadb.PersistentClient(path=chroma_dir)
            try:
                models['collection'] = chroma_client.get_collection("reviews")
            except:
                pass
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
    
    return models


# ================= PREDICTION FUNCTIONS =================
def predict_churn(customer_data, churn_model, feature_cols):
    """التنبؤ بـ Churn للعميل"""
    X = np.array([[customer_data.get(col, 0) for col in feature_cols]])
    churn_prob = churn_model.predict_proba(X)[0][1]
    
    if churn_prob > 0.7:
        risk_level = "HIGH"
    elif churn_prob > 0.4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return churn_prob, risk_level


# ================= STREAMLIT UI =================
st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CSS =================
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
        
        # System Status
        st.markdown("#### System Status")
        models = load_models()
        
        if models['churn_model'] is not None:
            st.markdown('<div class="status-badge status-low">✓ Models Loaded</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-high">✗ Models Not Found</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Training Section
        st.markdown("#### 🚀 Model Training")
        
        if st.button("🔄 Train/Retrain Models"):
            with st.spinner("Training models... This may take a few minutes..."):
                try:
                    # Load data
                    customers, journey, reviews, engagement, geography, products = load_data()
                    
                    if customers is not None:
                        # Build features
                        df = build_features(customers, journey, reviews, engagement, geography, products)
                        
                        # Save processed data
                        df.to_csv(os.path.join(ARTIFACTS_DIR, "processed_data.csv"), index=False)
                        
                        # Train churn model
                        model, auc, features = train_churn_model(df)
                        
                        # Perform segmentation
                        df = perform_segmentation(df)
                        
                        # Build vector store
                        embedder, collection = build_vector_store(reviews)
                        
                        st.success(f"✅ Training Complete! ROC-AUC: {auc:.4f}")
                        st.rerun()
                    else:
                        st.error("❌ Failed to load data files")
                except Exception as e:
                    st.error(f"❌ Training Error: {str(e)}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### 📊 About")
        st.markdown("""
        This platform uses advanced ML models to:
        - Predict customer churn probability
        - Segment customers intelligently
        - Provide AI-powered insights
        - Recommend retention strategies
        """)
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📊 Batch Analysis", "📈 Analytics"])
    
    # ================= TAB 1: Single Prediction =================
    with tab1:
        st.markdown("### Single Customer Analysis")
        st.markdown("Enter customer data to get real-time churn prediction and AI-powered recommendations.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Check if models are loaded
        if models['churn_model'] is None:
            st.warning("⚠️ Models not loaded. Please train models first using the sidebar.")
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
            
            if st.button("🔍 Analyze Customer", use_container_width=True):
                with st.spinner("Analyzing customer profile..."):
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
                    
                    # Predict churn
                    churn_prob, risk_level = predict_churn(
                        customer_data, 
                        models['churn_model'], 
                        models['feature_cols']
                    )
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Display Results
                    st.markdown("#### 📊 Analysis Results")
                    
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
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
                        prediction = "At Risk 🔴" if churn_prob > 0.5 else "Retained ✅"
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Prediction</div>
                                <div class="metric-value" style="font-size: 1.5rem;">{prediction}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # AI Insights
                    if models['embedder'] is not None and models['collection'] is not None:
                        st.markdown("<br><br>", unsafe_allow_html=True)
                        st.markdown("#### 🤖 AI-Powered Insights")
                        
                        with st.spinner("Generating AI insights..."):
                            # Retrieve similar reviews
                            query = f"Customer with rating {avg_rating}, {negative_reviews} complaints, {total_actions} actions"
                            similar_reviews = retrieve_similar_reviews(
                                query, 
                                models['embedder'], 
                                models['collection'], 
                                k=5
                            )
                            
                            # Generate insights
                            insights = generate_ai_insights(customer_data, churn_prob, similar_reviews)
                            
                            # Display insights
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
    
    # ================= TAB 2: Batch Analysis =================
    with tab2:
        st.markdown("### Batch Customer Analysis")
        st.markdown("Upload a CSV file with customer data for bulk churn prediction.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Check if models are loaded
        if models['churn_model'] is None:
            st.warning("⚠️ Models not loaded. Please train models first using the sidebar.")
        else:
            uploaded_file = st.file_uploader(
                "Upload CSV file with customer data",
                type=['csv'],
                help="File should contain columns matching the feature names used in training"
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
                                churn_prob, risk_level = predict_churn(
                                    customer_data,
                                    models['churn_model'],
                                    models['feature_cols']
                                )
                                
                                results.append({
                                    'CustomerID': row.get('CustomerID', idx),
                                    'Churn_Probability': churn_prob,
                                    'Risk_Level': risk_level,
                                    'Prediction': 'At Risk' if churn_prob > 0.5 else 'Retained'
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
                    <p>Processed customer data not found. Please train models first using the sidebar button.</p>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error loading analytics: {str(e)}")


if __name__ == "__main__":
    main()
