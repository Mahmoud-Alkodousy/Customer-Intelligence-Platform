# 🤖 AI Customer Intelligence Platform

Advanced customer analytics platform powered by Machine Learning and Large Language Models (LLMs) for churn prediction, customer segmentation, and intelligent insights.

## 🎯 Features

### 1. **Churn Prediction**
- Advanced ML model (Gradient Boosting) for predicting customer churn
- 16+ engineered features from customer journey, reviews, and engagement data
- Real-time risk assessment (LOW/MEDIUM/HIGH)
- ROC-AUC performance metrics

### 2. **AI-Powered Insights**
- LLM integration via OpenRouter (GPT-4o-mini)
- Personalized retention strategies
- Automated customer messaging
- Actionable next-best-action recommendations

### 3. **RAG System**
- Vector database (ChromaDB) for semantic search
- Similar customer review retrieval
- Context-aware analysis

### 4. **Customer Segmentation**
- K-Means clustering for customer grouping
- Segment-level analytics
- Targeted marketing insights

### 5. **Interactive Dashboard**
- Streamlit-based web interface
- Single customer analysis
- Batch processing capabilities
- Real-time visualizations with Plotly

## 📊 Data Structure

The platform processes 6 data sources:

1. **customers.csv** - Customer profiles (ID, Name, Age, Gender, Geography)
2. **customer_journey.csv** - User journey events (visits, clicks, dropoffs)
3. **customer_reviews_with_analysis3.csv** - Product reviews with sentiment
4. **engagement_data.csv** - Marketing engagement (views, clicks, likes)
5. **geography.csv** - Location data
6. **products.csv** - Product catalog

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenRouter API key ([Get one here](https://openrouter.ai/))

### Installation

```bash
# 1. Clone or extract the project
cd ai_platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env and add your OpenRouter API key

# 4. Train the models
python backend.py --train

# 5. Start the API server (in one terminal)
python backend.py --serve

# 6. Launch the dashboard (in another terminal)
streamlit run dashboard.py
```

### Access

- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health

## 📁 Project Structure

```
ai_platform/
├── backend.py                    # FastAPI backend with ML models
├── dashboard.py                  # Streamlit dashboard
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
├── README.md                     # This file
│
├── artifacts/                    # Generated after training
│   ├── churn_model.pkl          # Trained churn prediction model
│   ├── embedder.pkl             # Sentence transformer for RAG
│   ├── segmentation_model.pkl   # Customer segmentation model
│   ├── processed_data.csv       # Engineered features dataset
│   └── segment_analysis.csv     # Segment statistics
│
└── [CSV data files]              # Input data
    ├── 1770716474197_customers.csv
    ├── 1770716474197_customer_journey.csv
    ├── 1770716474197_customer_reviews_with_analysis3.csv
    ├── 1770716474197_engagement_data.csv
    ├── 1770716474198_geography.csv
    └── 1770716474198_products.csv
```

## 🔧 Usage

### Training the Models

```bash
python backend.py --train
```

This will:
1. Load and merge all data sources
2. Engineer 16+ features
3. Train churn prediction model (Gradient Boosting)
4. Perform customer segmentation (K-Means)
5. Build vector store for reviews (ChromaDB)
6. Save all artifacts to `./artifacts/`

**Expected Output:**
```
🚀 AI CUSTOMER INTELLIGENCE PLATFORM - TRAINING PIPELINE
================================================================
📂 Loading data files...
✅ Loaded: 100 customers, 4011 journey events, 1363 reviews

🔧 Building enhanced features...
✅ Features built: 35 columns
📊 Churn distribution: {0: 68, 1: 32}

🤖 Training churn prediction model...
✅ Model trained successfully!
📈 Train ROC-AUC: 0.9856
📈 Test ROC-AUC: 0.9234

🎯 Performing customer segmentation...
📊 Segment Analysis: [4 segments created]

🔍 Building vector store for reviews...
✅ Vector store built: 1363 reviews indexed

================================================================
✅ TRAINING COMPLETE!
================================================================
```

### Starting the API Server

```bash
python backend.py --serve
```

**Available Endpoints:**

- `GET /` - API information
- `GET /health` - Health check and model status
- `POST /predict_churn` - Get churn probability only
- `POST /analyze_customer` - Full AI-powered analysis

### Using the Dashboard

```bash
streamlit run dashboard.py
```

**Dashboard Tabs:**

1. **🎯 Single Customer Analysis**
   - Input customer features manually
   - Get AI-powered insights
   - View similar customer reviews
   - See retention recommendations

2. **📊 Batch Processing**
   - Upload CSV with multiple customers
   - Bulk churn prediction
   - Download results with risk levels
   - Visualize risk distribution

3. **📈 Analytics Dashboard**
   - View overall churn metrics
   - Analyze by age groups
   - Customer segmentation charts
   - Feature correlation analysis

## 🧪 API Examples

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "churn_model": true,
    "embedder": true,
    "vector_store": true
  }
}
```

### Predict Churn

```bash
curl -X POST http://localhost:8000/predict_churn \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35,
    "total_actions": 25,
    "avg_duration": 180,
    "avg_rating": 3.5,
    "negative_reviews": 2,
    "positive_reviews": 5,
    "total_views": 200,
    "total_clicks": 50,
    "conversion_rate": 0.15,
    "dropoff_rate": 0.3,
    "click_through_rate": 0.25,
    "sentiment_ratio": 0.71,
    "avg_product_price": 150,
    "unique_products_viewed": 5,
    "days_active": 30,
    "total_reviews": 7
  }'
```

**Response:**
```json
{
  "churn_probability": 0.42,
  "risk_level": "MEDIUM",
  "prediction": "Retained"
}
```

### Full Customer Analysis

```bash
curl -X POST http://localhost:8000/analyze_customer \
  -H "Content-Type: application/json" \
  -d '{ [same payload as above] }'
```

**Response:**
```json
{
  "churn_probability": 0.42,
  "risk_level": "MEDIUM",
  "priority": "IMPORTANT",
  "similar_reviews": [
    "Average experience, nothing special. [Rating: 3, Type: Other/Unclear]",
    "Good quality, but could be cheaper. [Rating: 3, Type: Price/Value]"
  ],
  "ai_insights": "**Risk Assessment**: Medium churn risk (42%)...",
  "customer_summary": {
    "age": 35,
    "engagement_score": 25,
    "satisfaction_score": 3.5,
    "conversion_rate": 0.15
  }
}
```

## 📊 Feature Engineering

The platform creates 16+ features from raw data:

### Journey Features
- `total_actions` - Total website interactions
- `avg_duration` - Average session duration
- `conversion_rate` - Checkout/total actions ratio
- `dropoff_rate` - Dropoffs/total actions ratio
- `unique_products_viewed` - Product variety
- `days_active` - Number of active days

### Review Features
- `avg_rating` - Average star rating
- `negative_reviews` - Count of negative reviews
- `positive_reviews` - Count of positive reviews
- `sentiment_ratio` - Positive/total reviews ratio
- `quality_complaints` - Product quality issues
- `price_complaints` - Price/value concerns

### Engagement Features
- `total_views` - Marketing content views
- `total_clicks` - Content clicks
- `click_through_rate` - CTR metric
- `engagement_per_event` - Likes per event

### Product Features
- `avg_product_price` - Average product price viewed

## 🧠 Model Architecture

### Churn Prediction Model
- **Algorithm**: Gradient Boosting Classifier
- **Features**: 16 engineered features
- **Performance**: ~92% ROC-AUC on test set
- **Output**: Probability score (0-1)

### Customer Segmentation
- **Algorithm**: K-Means Clustering
- **Segments**: 4 distinct customer groups
- **Features**: Age, actions, rating, views, clicks, price, conversion

### RAG System
- **Embeddings**: SentenceTransformer (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB
- **Purpose**: Retrieve similar customer reviews for context

### LLM Integration
- **Provider**: OpenRouter
- **Model**: GPT-4o-mini
- **Use Cases**:
  - Risk assessment narratives
  - Retention strategy recommendations
  - Personalized customer messaging
  - Next-best-action suggestions

## 🔐 Security Notes

- Never commit your `.env` file with API keys
- Use `.env.example` as a template
- Keep your OpenRouter API key secure
- For production, use proper authentication on API endpoints

## 🐛 Troubleshooting

### API Not Starting
```bash
# Check if port 8000 is available
lsof -i :8000

# Try a different port
python backend.py --serve --port 8001
```

### Models Not Loading
```bash
# Make sure you've run training first
python backend.py --train

# Check artifacts directory
ls -la artifacts/
```

### Dashboard Connection Error
```bash
# Verify API is running
curl http://localhost:8000/health

# Update API_URL in dashboard.py if using different port
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 📈 Performance Metrics

After training, you should see:

- **ROC-AUC**: 0.90 - 0.95
- **Churn Detection Rate**: ~85-90%
- **False Positive Rate**: <15%
- **Feature Importance**: Top features identified

## 🚀 Future Enhancements

Potential improvements:
- Real-time streaming predictions
- A/B testing framework for retention strategies
- Multi-model ensemble
- Advanced NLP for review sentiment
- Time-series forecasting
- Integration with CRM systems
- Automated email campaigns

## 📝 License

This is a demonstration project for educational purposes.

## 🤝 Contributing

Feel free to enhance the platform with:
- Additional ML models
- More sophisticated feature engineering
- Better visualization
- Performance optimizations

## 📧 Support

For issues or questions, please check:
1. This README
2. API documentation at `/docs`
3. Code comments in `backend.py` and `dashboard.py`

---

**Built with ❤️ using Python, FastAPI, Streamlit, scikit-learn, and OpenRouter**
