import requests
from typing import Optional, List
from utils.config import config


class GeminiService:
    """
    AI Service — dynamically queries OpenRouter for FREE models
    and picks the best available one.
    """

    def __init__(self):
        self.api_key = config.OPENROUTER_API_KEY
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.models_url = "https://openrouter.ai/api/v1/models"

        self.usage = 0
        self.max_usage = 50
        self._cached_models: List[str] = []
        self._cache_ts = 0

        print(f"🔑 OpenRouter key: {'loaded' if self.api_key else 'NOT SET'}")
        if self.api_key:
            print(f"🔑 Key prefix: {self.api_key[:18]}...")

    # ----------------------------------------------------------
    def _get_free_models(self) -> List[str]:
        """Fetch currently available free models from OpenRouter."""
        import time
        # Cache for 10 minutes
        if self._cached_models and (time.time() - self._cache_ts < 600):
            return self._cached_models

        try:
            r = requests.get(self.models_url, timeout=15)
            if r.status_code != 200:
                print(f"  ⚠️ Could not fetch models: HTTP {r.status_code}")
                return self._cached_models

            data = r.json().get("data", [])
            free_models = []

            for m in data:
                model_id = m.get("id", "")
                pricing = m.get("pricing", {})

                # Check if this model is free (prompt & completion = 0)
                prompt_cost = float(pricing.get("prompt", "1"))
                completion_cost = float(pricing.get("completion", "1"))

                if prompt_cost == 0 and completion_cost == 0:
                    # Skip junk models
                    if "content-safety" in model_id.lower():
                        continue
                    if "moderation" in model_id.lower():
                        continue
                    if "auto" in model_id.lower() and "router" in model_id.lower():
                        continue
                    # Skip tiny models unlikely to give good text
                    if "1b" in model_id.lower() or "mini" in model_id.lower():
                        # Keep mini but at end of list
                        pass

                    free_models.append(model_id)

            # Sort: prefer larger models first (70b, 72b, 32b > 8b > mini)
            def priority(m):
                ml = m.lower()
                if "70b" in ml or "72b" in ml: return 0
                if "32b" in ml or "34b" in ml: return 1
                if "27b" in ml or "13b" in ml: return 2
                if "8b" in ml or "9b" in ml or "7b" in ml: return 3
                if "mini" in ml or "small" in ml: return 4
                return 5

            free_models.sort(key=priority)

            self._cached_models = free_models
            self._cache_ts = time.time()

            print(f"  📋 Found {len(free_models)} free models:")
            for m in free_models[:5]:
                print(f"     • {m}")

            return free_models
        except Exception as e:
            print(f"  ❌ Error fetching models: {e}")
            return self._cached_models

    # ----------------------------------------------------------
    def get_response(self, question: str) -> Optional[str]:
        if not self.api_key:
            print("  ⚠️ OpenRouter API key missing")
            return None
        if self.usage >= self.max_usage:
            print("  ⚠️ OpenRouter usage limit reached")
            return None

        # Get currently-free models
        free_models = self._get_free_models()
        if not free_models:
            print("  ❌ No free models available")
            return None

        prompt = f"""You are MADO, a friendly and helpful daily-life AI assistant.
Answer the user's question thoroughly and helpfully.

Rules:
- Give a complete answer in 4-8 sentences, or a clear numbered list.
- For recipes: give ingredient list AND step-by-step instructions.
- For how-to questions: give numbered, actionable steps.
- For recommendations: give 3-5 concrete options with reasons.
- Be warm, use emojis occasionally, and stay on topic.
- Do NOT mention you are an AI model.

Question: {question}

Answer:"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://mado-assistant.app",
            "X-Title": "MADO Assistant",
        }

        # Try top 5 free models
        for model in free_models[:5]:
            for attempt in range(2):
                try:
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.8,
                        "max_tokens": 900,
                    }

                    r = requests.post(
                        self.url,
                        headers=headers,
                        json=payload,
                        timeout=30,
                    )
                    print(f"  📡 [{model}] attempt {attempt+1} → HTTP {r.status_code}")

                    if r.status_code == 200:
                        data = r.json()
                        choices = data.get("choices") or []
                        if not choices:
                            print("     ⚠️ No choices, retrying...")
                            continue

                        content = choices[0].get("message", {}).get("content")
                        if not content:
                            print("     ⚠️ Empty content")
                            continue

                        content = content.strip()
                        if len(content) < 20:
                            print(f"     ⚠️ Too short: {content[:60]}")
                            continue

                        # Reject junk responses
                        junk = ["user safety:", "content safety:", "i cannot",
                                "i can't assist", "as an ai"]
                        if any(j in content.lower() for j in junk) and len(content) < 100:
                            print("     ⚠️ Junk, next model...")
                            break

                        self.usage += 1
                        print(f"     ✅ Success ({len(content)} chars)")
                        return content

                    elif r.status_code == 429:
                        print("     ⚠️ Rate-limited, next model...")
                        break
                    elif r.status_code == 404:
                        print("     ⚠️ Model gone, next model...")
                        break
                    else:
                        print(f"     ⚠️ HTTP {r.status_code}: {r.text[:120]}")
                        break

                except Exception as e:
                    print(f"     ❌ {e}")
                    continue

        print("  ❌ All free models failed")
        return None