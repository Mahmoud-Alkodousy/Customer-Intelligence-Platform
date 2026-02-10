# 🚀 Quick Start Guide

Get the AI Customer Intelligence Platform up and running in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- fastapi, uvicorn (API)
- streamlit, plotly (Dashboard)
- scikit-learn (ML models)
- sentence-transformers, chromadb (RAG)
- pandas, numpy (Data processing)

## Step 2: Configure API Key

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenRouter API key
# Get one free at: https://openrouter.ai/
```

Your `.env` should look like:
```
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
OPENROUTER_URL=https://openrouter.ai/api/v1/chat/completions
LLM_MODEL=openai/gpt-4o-mini
```

## Step 3: Train Models

```bash
python backend.py --train
```

This will:
- ✅ Load 6 CSV data files (100 customers, 4k+ events)
- ✅ Engineer 16+ features
- ✅ Train churn prediction model (ROC-AUC ~0.92)
- ✅ Create customer segments (K-Means)
- ✅ Build vector store for reviews (ChromaDB)
- ✅ Save artifacts to `./artifacts/`

**Expected time:** 1-3 minutes

## Step 4: Start API Server

```bash
# Terminal 1
python backend.py --serve
```

API will be available at:
- http://localhost:8000
- http://localhost:8000/docs (Swagger UI)

## Step 5: Launch Dashboard

```bash
# Terminal 2
streamlit run dashboard.py
```

Dashboard will open at:
- http://localhost:8501

## Alternative: One-Command Start

```bash
# Uses the automated script
bash start.sh
```

This will:
1. Install dependencies
2. Check .env configuration
3. Train models
4. Start API server
5. Launch dashboard
6. Show all URLs and process IDs

Press `Ctrl+C` to stop all services.

---

## 📱 Using the Dashboard

### Single Customer Analysis Tab

1. **Input customer data** using sliders and number inputs
2. **Click "Analyze Customer"**
3. **View results:**
   - Churn probability gauge
   - Risk level (HIGH/MEDIUM/LOW)
   - Priority status
   - AI-generated insights
   - Similar customer reviews
   - Retention recommendations

### Batch Processing Tab

1. **Download sample template** CSV
2. **Prepare your data** (multiple customers)
3. **Upload CSV file**
4. **Click "Run Batch Prediction"**
5. **Download results** with churn scores and risk levels

### Analytics Dashboard Tab

- Overall churn metrics
- Churn rate by age group
- Customer segmentation charts
- Feature correlation analysis

---

## 🧪 Testing the API

```bash
# Test all endpoints
python test_api.py
```

Or manually:

```bash
# Health check
curl http://localhost:8000/health

# Predict churn
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

---

## 🐛 Troubleshooting

### Issue: "API is not running"

**Solution:**
```bash
# Check if API is running
curl http://localhost:8000/health

# If not, start it
python backend.py --serve
```

### Issue: "Models not trained yet"

**Solution:**
```bash
# Run training first
python backend.py --train

# Then restart API
python backend.py --serve
```

### Issue: "Port already in use"

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>

# Or use different port
python backend.py --serve --port 8001
```

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade
```

### Issue: "OpenRouter API error"

**Solution:**
- Check your API key in `.env`
- Verify you have credits at https://openrouter.ai/
- Check API key format: `sk-or-v1-...`

---

## 📊 What to Expect

After setup, you should have:

✅ **API Server** running on http://localhost:8000
✅ **Dashboard** running on http://localhost:8501
✅ **4 trained models:**
   - Churn prediction (Gradient Boosting)
   - Customer segmentation (K-Means)
   - Review embeddings (SentenceTransformer)
   - Vector store (ChromaDB)

✅ **Performance metrics:**
   - ROC-AUC: ~0.92
   - Training time: 1-3 min
   - Prediction time: <100ms
   - AI analysis time: 5-15s

---

## 🎯 Next Steps

1. **Explore the dashboard** with different customer profiles
2. **Try batch processing** with your own customer data
3. **Review AI insights** for retention strategies
4. **Customize features** in `backend.py`
5. **Enhance dashboard** in `dashboard.py`
6. **Integrate with your CRM** via API

---

## 📚 Learn More

- **Full Documentation:** See `README.md`
- **API Reference:** Visit http://localhost:8000/docs
- **Code Examples:** Check `test_api.py`
- **OpenRouter Docs:** https://openrouter.ai/docs

---

**Need help?** Check the main README.md or review the code comments!
