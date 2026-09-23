import os
from dotenv import load_dotenv
import requests

# ✅ Load .env from the correct path
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print(f"🔑 API Key found: {api_key[:10]}..." if api_key else "❌ No API Key found!")
print(f"📌 API Key length: {len(api_key) if api_key else 0}")

if not api_key:
    print("⚠️ Please check your .env file!")
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📁 .env path: {os.path.join(os.getcwd(), '.env')}")
    exit()

question = "Who is Isaac Newton?"
print(f"\n📤 Testing with question: {question}")

# Try different models
models = [
    "gemini-1.5-flash",
    "gemini-1.0-pro", 
    "gemini-pro"
]

for model in models:
    print(f"\n🔄 Testing model: {model}")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    data = {
        "contents": [{"parts": [{"text": question}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 500,
        }
    }
    
    try:
        response = requests.post(
            f"{url}?key={api_key}",
            headers={"Content-Type": "application/json"},
            json=data,
            timeout=10
        )
        
        print(f"📥 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                print(f"✅ SUCCESS! Response: {text[:200]}...")
                print(f"✅ Use this model: {model}")
                break
            else:
                print("⚠️ No candidates in response")
        else:
            print(f"❌ Failed: {response.text[:200]}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")