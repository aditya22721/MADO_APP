import os
from dotenv import load_dotenv
import requests

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print(f"🔑 API Key: {api_key[:10]}..." if api_key else "❌ No API Key")

question = "Who is Isaac Newton?"

# Try different models
models = [
    "gemini-pro",       # Most compatible
    "gemini-1.0-pro",   # Older version
    "gemini-1.5-pro",   # Newer version
    "gemini-1.5-flash"  # Latest
]

for model in models:
    print(f"\n🔄 Testing: {model}")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    data = {
        "contents": [{"parts": [{"text": question}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 500}
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
                print(f"✅ WORKS! Response: {text[:150]}...")
                print(f"✅ Use this model: {model}")
                break
            else:
                print("⚠️ No candidates")
        else:
            print(f"❌ Failed: {response.text[:100]}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")