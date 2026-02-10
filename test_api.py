"""
API Testing Script
Test the AI Customer Intelligence Platform API endpoints
"""

import requests
import json
from pprint import pprint

API_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing /health endpoint")
    print("="*60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    pprint(response.json())
    
    return response.status_code == 200


def test_predict_churn():
    """Test churn prediction endpoint"""
    print("\n" + "="*60)
    print("Testing /predict_churn endpoint")
    print("="*60)
    
    # Sample customer data
    customer_data = {
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
    }
    
    print("Input Data:")
    pprint(customer_data)
    
    response = requests.post(
        f"{API_URL}/predict_churn",
        json=customer_data,
        timeout=10
    )
    
    print(f"\nStatus Code: {response.status_code}")
    print("Response:")
    pprint(response.json())
    
    return response.status_code == 200


def test_analyze_customer():
    """Test full customer analysis endpoint"""
    print("\n" + "="*60)
    print("Testing /analyze_customer endpoint")
    print("="*60)
    
    # High-risk customer profile
    customer_data = {
        "Age": 45,
        "total_actions": 10,
        "avg_duration": 80,
        "avg_rating": 2.1,
        "negative_reviews": 4,
        "positive_reviews": 1,
        "total_views": 50,
        "total_clicks": 5,
        "conversion_rate": 0.05,
        "dropoff_rate": 0.7,
        "click_through_rate": 0.1,
        "sentiment_ratio": 0.2,
        "avg_product_price": 300,
        "unique_products_viewed": 2,
        "days_active": 10,
        "total_reviews": 5
    }
    
    print("Input Data (High-Risk Customer):")
    pprint(customer_data)
    
    print("\n⏳ Calling AI analysis (may take 10-20 seconds)...")
    
    response = requests.post(
        f"{API_URL}/analyze_customer",
        json=customer_data,
        timeout=30
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        
        print("\n📊 ANALYSIS RESULTS:")
        print("-" * 60)
        print(f"Churn Probability: {result['churn_probability']:.2%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Priority: {result['priority']}")
        
        print("\n💬 Similar Reviews:")
        for i, review in enumerate(result.get('similar_reviews', [])[:3], 1):
            print(f"  {i}. {review[:100]}...")
        
        print("\n🧠 AI Insights:")
        print(result.get('ai_insights', 'No insights'))
        
        print("\n📋 Customer Summary:")
        pprint(result.get('customer_summary', {}))
    else:
        print("Error Response:")
        pprint(response.json())
    
    return response.status_code == 200


def test_batch_prediction():
    """Test batch prediction with multiple customers"""
    print("\n" + "="*60)
    print("Testing Batch Prediction")
    print("="*60)
    
    customers = [
        {
            "Age": 25,
            "total_actions": 50,
            "avg_duration": 250,
            "avg_rating": 4.5,
            "negative_reviews": 0,
            "positive_reviews": 8,
            "total_views": 300,
            "total_clicks": 80,
            "conversion_rate": 0.3,
            "dropoff_rate": 0.1,
            "click_through_rate": 0.27,
            "sentiment_ratio": 1.0,
            "avg_product_price": 120,
            "unique_products_viewed": 10,
            "days_active": 60,
            "total_reviews": 8
        },
        {
            "Age": 55,
            "total_actions": 5,
            "avg_duration": 60,
            "avg_rating": 1.8,
            "negative_reviews": 6,
            "positive_reviews": 0,
            "total_views": 20,
            "total_clicks": 2,
            "conversion_rate": 0.02,
            "dropoff_rate": 0.9,
            "click_through_rate": 0.1,
            "sentiment_ratio": 0.0,
            "avg_product_price": 400,
            "unique_products_viewed": 1,
            "days_active": 3,
            "total_reviews": 6
        }
    ]
    
    print(f"Testing {len(customers)} customers...\n")
    
    results = []
    for i, customer in enumerate(customers, 1):
        print(f"Customer {i}:")
        response = requests.post(f"{API_URL}/predict_churn", json=customer)
        
        if response.status_code == 200:
            result = response.json()
            results.append(result)
            print(f"  Churn: {result['churn_probability']:.2%} - {result['risk_level']}")
        else:
            print(f"  Error: {response.status_code}")
    
    print(f"\n✅ Successfully processed {len(results)}/{len(customers)} customers")
    
    return len(results) == len(customers)


def run_all_tests():
    """Run all API tests"""
    print("\n" + "="*60)
    print("🧪 AI CUSTOMER INTELLIGENCE PLATFORM - API TESTS")
    print("="*60)
    
    tests = [
        ("Health Check", test_health),
        ("Predict Churn", test_predict_churn),
        ("Analyze Customer", test_analyze_customer),
        ("Batch Prediction", test_batch_prediction)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Connection Error: API is not running!")
            print("Please start the API: python backend.py --serve")
            break
        except Exception as e:
            print(f"\n❌ Test failed: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s} : {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")


if __name__ == "__main__":
    run_all_tests()
