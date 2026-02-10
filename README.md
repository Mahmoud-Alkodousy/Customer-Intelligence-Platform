# AI Customer Intelligence Platform

Advanced machine learning platform for customer churn prediction and retention optimization.

## 🚀 Deployment Instructions

### 1. Prepare Your Repository

Create the following structure on GitHub:

```
your-repo/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── customers.csv
│   ├── customer_journey.csv
│   ├── customer_reviews_with_analysis.csv
│   ├── engagement_data.csv
│   ├── geography.csv
│   └── products.csv
└── artifacts/
    └── (empty - will be created by training)
```

### 2. Upload Data Files

Place your CSV files in the `data/` folder with these exact names:
- `customers.csv`
- `customer_journey.csv`
- `customer_reviews_with_analysis.csv`
- `engagement_data.csv`
- `geography.csv`
- `products.csv`

### 3. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file path: `app.py`
6. Click "Advanced settings"
7. Add environment variable:
   - Key: `OPENROUTER_API_KEY`
   - Value: Your OpenRouter API key
8. Click "Deploy"

### 4. First Time Setup

After deployment:
1. Wait for the app to load
2. Go to the sidebar
3. Click "🔄 Train/Retrain Models"
4. Wait for training to complete (3-5 minutes)
5. Models will be saved and loaded automatically

## 📋 Features

- **Single Customer Analysis**: Real-time churn prediction with AI insights
- **Batch Analysis**: Upload CSV files for bulk predictions
- **Analytics Dashboard**: Comprehensive visualizations and KPIs
- **AI-Powered Insights**: Automated recommendations using LLM
- **Customer Segmentation**: Intelligent grouping based on behavior

## 🔧 Environment Variables

Required in Streamlit Cloud settings:
- `OPENROUTER_API_KEY`: Your OpenRouter API key for AI insights

## 📊 Data Format

Your CSV files should contain the following columns:

### customers.csv
- CustomerID, Age, GeographyID, etc.

### customer_journey.csv
- CustomerID, Action, Stage, Duration, ProductID, VisitDate, etc.

### customer_reviews_with_analysis.csv
- ReviewID, CustomerID, Rating, ReviewText, review_type, problem_type, etc.

### engagement_data.csv
- EngagementID, ProductID, ViewsClicksCombined, Likes, ContentType, CampaignID, etc.

### geography.csv
- GeographyID, Region, etc.

### products.csv
- ProductID, Price, etc.

## 🛠️ Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📝 Notes

- First deployment requires training (3-5 min)
- Models are cached and persist between sessions
- Analytics require processed data from training
- AI insights require valid OPENROUTER_API_KEY

## 🔐 Security

- Never commit API keys to repository
- Use Streamlit Cloud secrets for sensitive data
- Data is processed locally on Streamlit Cloud servers

## 📞 Support

For issues or questions, please open an issue on GitHub.
