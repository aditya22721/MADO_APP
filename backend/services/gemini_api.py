import time
import requests
from typing import Optional, List
from utils.config import config


class GeminiService:
    """
    AI Service — dynamically queries OpenRouter for FREE text-generation models
    and filters out rate-limited / non-chat models.
    """

    # Models to ALWAYS skip (junk, safety classifiers, code-only, etc.)
    SKIP_PATTERNS = [
        "content-safety", "moderation", "safety",
        "code", "coding",
        "thinkingmachines",  # agent-only harness
        "auto",
    ]

    def __init__(self):
        self.api_key = config.OPENROUTER_API_KEY
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.models_url = "https://openrouter.ai/api/v1/models"

        self.usage = 0
        self.max_usage = 50
        self._cached_models: List[str] = []
        self._cache_ts = 0.0
        self._failed_models: set = set()   # Skip models that failed in this run

        print(f"🔑 OpenRouter key: {'loaded' if self.api_key else 'NOT SET'}")
        if self.api_key:
            print(f"🔑 Key prefix: {self.api_key[:18]}...")

    # ----------------------------------------------------------
    def _get_free_models(self) -> List[str]:
        """Fetch & cache currently available free text-gen models."""
        if self._cached_models and (time.time() - self._cache_ts < 600):
            return self._cached_models

        try:
            r = requests.get(self.models_url, timeout=15)
            if r.status_code != 200:
                print(f"  ⚠️ Model list HTTP {r.status_code}")
                return self._cached_models

            data = r.json().get("data", [])
            free_models = []

            for m in data:
                model_id = m.get("id", "")
                pricing = m.get("pricing", {})

                try:
                    p_cost = float(pricing.get("prompt", "1"))
                    c_cost = float(pricing.get("completion", "1"))
                except (ValueError, TypeError):
                    continue

                if p_cost != 0 or c_cost != 0:
                    continue

                # Skip junk patterns
                ml = model_id.lower()
                if any(skip in ml for skip in self.SKIP_PATTERNS):
                    continue

                free_models.append(model_id)

            # Sort: bigger models first, prefer known good families
            def priority(m):
                ml = m.lower()
                if "qwen" in ml and "27b" in ml: return 0
                if "llama" in ml and "70b" in ml: return 0
                if "70b" in ml or "72b" in ml: return 1
                if "32b" in ml or "34b" in ml: return 2
                if "27b" in ml or "13b" in ml: return 3
                if "8b" in ml or "9b" in ml or "7b" in ml: return 4
                if "mini" in ml or "small" in ml: return 6
                return 5

            free_models.sort(key=priority)

            self._cached_models = free_models
            self._cache_ts = time.time()

            print(f"  📋 Found {len(free_models)} free text models")
            for m in free_models[:8]:
                print(f"     • {m}")

            return free_models
        except Exception as e:
            print(f"  ❌ Model fetch error: {e}")
            return self._cached_models

    # ----------------------------------------------------------
    def get_response(self, question: str) -> Optional[str]:
        if not self.api_key:
            return None
        if self.usage >= self.max_usage:
            print("  ⚠️ Daily limit reached")
            return None

        free_models = self._get_free_models()
        if not free_models:
            return None

        # Skip models that already failed this session
        candidates = [m for m in free_models if m not in self._failed_models]

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

        # Try top 8 candidates
        for model in candidates[:8]:
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
                print(f"  📡 [{model}] → HTTP {r.status_code}")

                if r.status_code == 200:
                    data = r.json()
                    choices = data.get("choices") or []
                    if not choices:
                        continue

                    content = choices[0].get("message", {}).get("content")
                    if not content:
                        continue

                    content = content.strip()
                    if len(content) < 20:
                        continue

                    # Reject junk
                    junk = ["user safety:", "content safety:", "i cannot",
                            "i can't assist", "as an ai", "i'm sorry"]
                    if any(j in content.lower() for j in junk) and len(content) < 100:
                        self._failed_models.add(model)
                        continue

                    self.usage += 1
                    print(f"     ✅ Success ({len(content)} chars)")
                    return content

                elif r.status_code in (403, 404):
                    print(f"     ⏭️ Skipping (unavailable)")
                    self._failed_models.add(model)
                    continue

                elif r.status_code == 429:
                    print(f"     ⏭️ Rate-limited, next...")
                    self._failed_models.add(model)
                    continue

                else:
                    print(f"     ⚠️ HTTP {r.status_code}")
                    continue

            except Exception as e:
                print(f"     ❌ {e}")
                continue

        print("  ❌ All free models exhausted")
        return None