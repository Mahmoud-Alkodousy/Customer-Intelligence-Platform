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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from sentence_transformers import SentenceTransformer
import chromadb
import requests


# ================= CONFIG =================
OPENROUTER_API_KEY = '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'
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "openai/gpt-4o-mini"

ARTIFACTS_DIR = "./artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ================= DATA LOADING =================
def load_data():
    """تحميل جميع ملفات البيانات"""
    print("📂 Loading data files...")
    
    customers = pd.read_csv("1770716474197_customers.csv")
    journey = pd.read_csv("1770716474197_customer_journey.csv")
    reviews = pd.read_csv("1770716474197_customer_reviews_with_analysis3.csv")
    engagement = pd.read_csv("1770716474197_engagement_data.csv")
    geography = pd.read_csv("1770716474198_geography.csv")
    products = pd.read_csv("1770716474198_products.csv")
    
    print(f"✅ Loaded: {len(customers)} customers, {len(journey)} journey events, {len(reviews)} reviews")
    
    return customers, journey, reviews, engagement, geography, products


# ================= ENHANCED FEATURE ENGINEERING =================
def build_features(customers, journey, reviews, engagement, geography, products):
    """بناء features متقدمة للتحليل"""
    print("\n🔧 Building enhanced features...")
    
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
    
    # -------- Engagement Features (FIXED) --------
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
    # Calculate average product price per customer
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
    # Multiple criteria for churn
    df["churn_label"] = (
        ((df["negative_reviews"] >= 2) | 
         (df["dropoff_rate"] > 0.5) | 
         (df["avg_rating"] < 2.5) |
         ((df["sentiment_ratio"] < 0.3) & (df["total_reviews"] > 2)))
    ).astype(int)
    
    print(f"✅ Features built: {df.shape[1]} columns")
    print(f"📊 Churn distribution: {df['churn_label'].value_counts().to_dict()}")
    
    return df


# ================= CUSTOMER SEGMENTATION =================
def perform_segmentation(df, n_clusters=4):
    """تقسيم العملاء إلى شرائح"""
    print("\n🎯 Performing customer segmentation...")
    
    segment_features = [
        "Age", "total_actions", "avg_rating", "total_views", 
        "total_clicks", "avg_product_price", "conversion_rate"
    ]
    
    X_segment = df[segment_features].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_segment)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["customer_segment"] = kmeans.fit_predict(X_scaled)
    
    # Segment analysis
    segment_analysis = df.groupby("customer_segment").agg({
        "CustomerID": "count",
        "Age": "mean",
        "avg_rating": "mean",
        "total_actions": "mean",
        "conversion_rate": "mean",
        "churn_label": "mean"
    }).round(2)
    
    print("📊 Segment Analysis:")
    print(segment_analysis)
    
    # Save artifacts
    with open(os.path.join(ARTIFACTS_DIR, "segmentation_model.pkl"), "wb") as f:
        pickle.dump((kmeans, scaler, segment_features), f)
    
    return df, segment_analysis


# ================= TRAIN CHURN MODEL =================
def train_churn_model(df):
    """تدريب نموذج التنبؤ بالـ Churn"""
    print("\n🤖 Training churn prediction model...")
    
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
    train_preds = model.predict_proba(X_train)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]
    
    train_auc = roc_auc_score(y_train, train_preds)
    test_auc = roc_auc_score(y_test, test_preds)
    
    print(f"✅ Model trained successfully!")
    print(f"📈 Train ROC-AUC: {train_auc:.4f}")
    print(f"📈 Test ROC-AUC: {test_auc:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print("\n🎯 Top 10 Important Features:")
    print(feature_importance.head(10))
    
    # Save model
    with open(os.path.join(ARTIFACTS_DIR, "churn_model.pkl"), "wb") as f:
        pickle.dump((model, feature_cols), f)
    
    return model, test_auc, feature_cols


# ================= RAG SYSTEM =================
def build_vector_store(reviews):
    """بناء قاعدة بيانات متجهات للمراجعات"""
    print("\n🔍 Building vector store for reviews...")
    
    # Prepare review texts with metadata
    reviews["full_text"] = (
        reviews["ReviewText"].fillna("") + 
        f" [Rating: " + reviews["Rating"].astype(str) + 
        ", Type: " + reviews["problem_type"].fillna("none") + "]"
    )
    
    texts = reviews["full_text"].tolist()
    
    # Generate embeddings
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts, show_progress_bar=True)
    
    # Create ChromaDB collection with persistent storage
    chroma_dir = os.path.join(ARTIFACTS_DIR, "chroma_db")
    client = chromadb.PersistentClient(path=chroma_dir)
    
    # Delete if exists
    try:
        client.delete_collection("reviews")
    except:
        pass
    
    collection = client.create_collection("reviews")
    
    # Add to collection
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        metadata = {
            "rating": str(reviews.iloc[i]["Rating"]),
            "review_type": str(reviews.iloc[i]["review_type"]),
            "problem_type": str(reviews.iloc[i]["problem_type"])
        }
        
        collection.add(
            ids=[str(i)],
            documents=[text],
            embeddings=[emb.tolist()],
            metadatas=[metadata]
        )
    
    print(f"✅ Vector store built: {len(texts)} reviews indexed")
    
    # Save embedder
    with open(os.path.join(ARTIFACTS_DIR, "embedder.pkl"), "wb") as f:
        pickle.dump(embedder, f)
    
    return embedder, collection


# ================= LLM INTEGRATION =================
def call_llm(prompt: str, temperature: float = 0.7) -> str:
    """استدعاء نموذج اللغة عبر OpenRouter"""
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system", 
                "content": "You are an expert AI customer intelligence analyst. Provide actionable insights based on customer data."
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 800
    }
    
    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ LLM Error: {str(e)}"


def retrieve_similar_reviews(query: str, embedder, collection, k: int = 5):
    """استرجاع المراجعات المشابهة"""
    if collection is None or embedder is None:
        return []
    
    try:
        q_emb = embedder.encode([query])[0].tolist()
        results = collection.query(query_embeddings=[q_emb], n_results=k)
        return results["documents"][0] if results["documents"] else []
    except Exception as e:
        print(f"Error retrieving reviews: {e}")
        return []


def generate_ai_insights(customer_data: Dict, churn_prob: float, similar_reviews: List[str]):
    """توليد تحليل AI شامل"""
    
    prompt = f"""
Analyze this customer profile and provide strategic recommendations:

CUSTOMER DATA:
- Age: {customer_data.get('Age', 'N/A')}
- Total Actions: {customer_data.get('total_actions', 0)}
- Average Rating: {customer_data.get('avg_rating', 0):.2f}
- Negative Reviews: {customer_data.get('negative_reviews', 0)}
- Conversion Rate: {customer_data.get('conversion_rate', 0):.2%}
- Churn Probability: {churn_prob:.2%}

SIMILAR CUSTOMER COMPLAINTS:
{chr(10).join([f"- {review[:150]}" for review in similar_reviews[:3]])}

Provide a concise analysis with:
1. **Risk Assessment**: Brief churn risk level and key factors
2. **Retention Strategy**: 2-3 specific actions to prevent churn
3. **Personalized Message**: Draft a short, empathetic message to this customer
4. **Next Best Action**: Immediate next step for customer success team

Keep it actionable and under 200 words.
"""
    
    return call_llm(prompt)


# ================= FASTAPI APPLICATION =================
app = FastAPI(
    title="AI Customer Intelligence Platform",
    description="Advanced customer analytics with ML and LLM",
    version="2.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerInput(BaseModel):
    Age: float
    total_actions: float
    avg_duration: float
    avg_rating: float
    negative_reviews: float
    positive_reviews: Optional[float] = 0
    total_views: float
    total_clicks: float
    conversion_rate: Optional[float] = 0
    dropoff_rate: Optional[float] = 0
    click_through_rate: Optional[float] = 0
    sentiment_ratio: Optional[float] = 0
    avg_product_price: Optional[float] = 0
    unique_products_viewed: Optional[float] = 0
    days_active: Optional[float] = 0
    total_reviews: Optional[float] = 0


# Global artifacts
churn_model = None
feature_cols = None
embedder = None
collection = None
segment_model = None


@app.on_event("startup")
async def load_artifacts():
    """تحميل النماذج المدربة"""
    global churn_model, feature_cols, embedder, collection, segment_model
    
    try:
        # Load churn model
        if os.path.exists(os.path.join(ARTIFACTS_DIR, "churn_model.pkl")):
            with open(os.path.join(ARTIFACTS_DIR, "churn_model.pkl"), "rb") as f:
                churn_model, feature_cols = pickle.load(f)
            print("✅ Churn model loaded")
        
        # Load embedder
        if os.path.exists(os.path.join(ARTIFACTS_DIR, "embedder.pkl")):
            with open(os.path.join(ARTIFACTS_DIR, "embedder.pkl"), "rb") as f:
                embedder = pickle.load(f)
            print("✅ Embedder loaded")
        
        # Load ChromaDB with persistent storage
        chroma_dir = os.path.join(ARTIFACTS_DIR, "chroma_db")
        if os.path.exists(chroma_dir):
            chroma_client = chromadb.PersistentClient(path=chroma_dir)
            try:
                collection = chroma_client.get_collection("reviews")
                print("✅ Vector store loaded")
            except:
                print("⚠️ Vector store not found - run training first")
        else:
            print("⚠️ Vector store directory not found - run training first")
            
    except Exception as e:
        print(f"⚠️ Error loading artifacts: {e}")


@app.get("/")
def root():
    return {
        "message": "AI Customer Intelligence Platform API",
        "version": "2.0",
        "status": "running",
        "endpoints": ["/predict_churn", "/analyze_customer", "/health"]
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "churn_model": churn_model is not None,
            "embedder": embedder is not None,
            "vector_store": collection is not None
        }
    }


@app.post("/predict_churn")
def predict_churn(data: CustomerInput):
    """التنبؤ باحتمالية الـ Churn فقط"""
    
    if churn_model is None or feature_cols is None:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    
    try:
        # Prepare features
        input_dict = data.dict()
        X = np.array([[input_dict.get(col, 0) for col in feature_cols]])
        
        # Predict
        churn_prob = churn_model.predict_proba(X)[0][1]
        
        # Risk level
        if churn_prob > 0.7:
            risk_level = "HIGH"
        elif churn_prob > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "churn_probability": float(churn_prob),
            "risk_level": risk_level,
            "prediction": "At Risk" if churn_prob > 0.5 else "Retained"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/analyze_customer")
def analyze_customer(data: CustomerInput):
    """تحليل شامل للعميل مع AI Insights"""
    
    if any(x is None for x in [churn_model, feature_cols, embedder, collection]):
        raise HTTPException(status_code=503, detail="System not fully initialized")
    
    try:
        # Predict churn
        input_dict = data.dict()
        X = np.array([[input_dict.get(col, 0) for col in feature_cols]])
        churn_prob = churn_model.predict_proba(X)[0][1]
        
        # Retrieve similar reviews
        query = f"Customer with rating {data.avg_rating}, {data.negative_reviews} complaints, {data.total_actions} actions"
        similar_reviews = retrieve_similar_reviews(query, embedder, collection, k=5)
        
        # Generate AI insights
        ai_insights = generate_ai_insights(input_dict, churn_prob, similar_reviews)
        
        # Risk level
        if churn_prob > 0.7:
            risk_level = "HIGH"
            priority = "URGENT"
        elif churn_prob > 0.4:
            risk_level = "MEDIUM"
            priority = "IMPORTANT"
        else:
            risk_level = "LOW"
            priority = "NORMAL"
        
        return {
            "churn_probability": float(churn_prob),
            "risk_level": risk_level,
            "priority": priority,
            "similar_reviews": similar_reviews[:3],
            "ai_insights": ai_insights,
            "customer_summary": {
                "age": data.Age,
                "engagement_score": data.total_actions,
                "satisfaction_score": data.avg_rating,
                "conversion_rate": data.conversion_rate
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


# ================= TRAINING PIPELINE =================
def train_all_models():
    """تدريب جميع النماذج"""
    print("="*60)
    print("🚀 AI CUSTOMER INTELLIGENCE PLATFORM - TRAINING PIPELINE")
    print("="*60)
    
    # Load data
    customers, journey, reviews, engagement, geography, products = load_data()
    
    # Build features
    df = build_features(customers, journey, reviews, engagement, geography, products)
    
    # Save processed dataset
    df.to_csv(os.path.join(ARTIFACTS_DIR, "processed_data.csv"), index=False)
    print(f"\n💾 Processed data saved: {ARTIFACTS_DIR}/processed_data.csv")
    
    # Train churn model
    model, auc, features = train_churn_model(df)
    
    # Perform segmentation
    df, segment_analysis = perform_segmentation(df)
    segment_analysis.to_csv(os.path.join(ARTIFACTS_DIR, "segment_analysis.csv"))
    
    # Build vector store
    embedder, collection = build_vector_store(reviews)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"📊 Churn Model ROC-AUC: {auc:.4f}")
    print(f"🎯 Customer Segments: {df['customer_segment'].nunique()}")
    print(f"🔍 Reviews Indexed: {len(reviews)}")
    print("\n🚀 Ready to serve API requests!")
    
    return df


# ================= MAIN =================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Customer Intelligence Platform")
    parser.add_argument("--train", action="store_true", help="Train all models")
    parser.add_argument("--serve", action="store_true", help="Start API server")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    args = parser.parse_args()
    
    if args.train:
        train_all_models()
    
    if args.serve:
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    
    if not args.train and not args.serve:
        print("Usage: python backend.py --train | --serve | --train --serve")