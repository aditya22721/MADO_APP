import random
import requests
import re
from datetime import datetime
from typing import Optional

from services.gemini_api import GeminiService

try:
    import wikipedia
    WIKI_OK = True
except ImportError:
    WIKI_OK = False
    print("⚠️ wikipedia-api not installed")


class KnowledgeService:
    def __init__(self):
        self.gemini = GeminiService()

        self.local = {
            "capital of france": "Paris is the capital of France.",
            "capital of india": "New Delhi is the capital of India.",
            "capital of usa": "Washington, D.C. is the capital of the USA.",
            "capital of uk": "London is the capital of the UK.",
            "capital of japan": "Tokyo is the capital of Japan.",
            "moon": "🌙 The Moon is Earth's only natural satellite, about 384,400 km away.",
            "sun": "☀️ The Sun is the star at the center of our solar system.",
            "planets": "🪐 8 planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.",
            "dna": "🧬 DNA carries genetic instructions for all living organisms.",
            "photosynthesis": "🌿 Photosynthesis is how plants convert sunlight into chemical energy.",
        }

        self.emergency_instructions = {
            "heart_attack": "🫀 HEART ATTACK FIRST AID:\n1. CALL 911\n2. Sit them down\n3. Loosen clothing\n4. CPR if unconscious",
            "choking": "🫁 CHOKING FIRST AID:\n1. Stand behind them\n2. Fist above navel\n3. Thrust inward/upward\n4. Call 911 if unconscious",
            "bleeding": "🩸 SEVERE BLEEDING:\n1. CALL 911\n2. Apply direct pressure\n3. Elevate above heart",
            "burns": "🔥 BURNS:\n1. Cool water 10-15 min\n2. Cover with gauze\n3. Do NOT use ice",
            "fracture": "🦴 FRACTURE:\n1. Call 911 if severe\n2. Don't move them\n3. Apply ice\n4. Immobilize",
            "seizure": "🧠 SEIZURE:\n1. Stay with them\n2. Clear objects\n3. Soft thing under head\n4. Time the seizure",
            "stroke": "🧠 STROKE — ACT FAST:\nF — Face drooping\nA — Arm weakness\nS — Speech difficulty\nT — Time to call 911",
            "general": "🚨 EMERGENCY:\n1. STAY CALM\n2. CALL 911 / 112 / 999\n3. Stay with them",
        }

    def get_response(self, question: str) -> str:
        q = question.lower().strip()
        print(f"\n🔍 Q: {question}")

        # 1. LOCAL
        for key, val in self.local.items():
            if key in q:
                print(f"  ✅ Local: {key}")
                return val

        # 2. JOKE / FACT / TIME / WEATHER
        if "joke" in q:
            return random.choice([
                "Why don't scientists trust atoms? Because they make up everything! 😄",
                "What do you call a fake noodle? An impasta! 🍝",
                "Why did the scarecrow win an award? He was outstanding in his field! 🌾",
                "Why don't eggs tell jokes? They'd crack each other up! 🥚",
            ])

        if "fact" in q:
            return random.choice([
                "🍯 Honey never spoils — 3000-year-old honey was found edible in Egyptian tombs!",
                "🐙 Octopuses have three hearts and blue blood.",
                "🦈 Sharks existed before trees — over 400 million years ago!",
            ])

        if "time" in q and "weather" not in q:
            return f"🕐 {datetime.now().strftime('%I:%M %p, %A, %B %d, %Y')}."

        if "weather" in q:
            try:
                r = requests.get("https://wttr.in/?format=%l:+%c+%t", timeout=5)
                if r.status_code == 200:
                    return f"🌤️ {r.text.strip()}"
            except Exception:
                pass
            return "🌤️ Weather service unavailable."

        # 3. DUCKDUCKGO — try always (works for many "how to" queries too)
        print("  🦆 Trying DuckDuckGo...")
        ddg = self._duckduckgo(q)
        if ddg:
            print("  ✅ DDG answered")
            return ddg

        # 4. WIKIPEDIA — try always
        if WIKI_OK:
            print("  📚 Trying Wikipedia...")
            wiki = self._wikipedia(q)
            if wiki:
                print("  ✅ Wiki answered")
                return wiki

        # 5. AI (OpenRouter) — always last resort
        print("  🤖 Trying AI (OpenRouter)...")
        ai = self.gemini.get_response(question)
        if ai:
            print("  ✅ AI answered")
            return f"🤖 {ai}"

        # 6. Math
        try:
            expr = re.sub(r"[^0-9+\-*/(). ]", "", question)
            if any(op in expr for op in "+-*/") and len(expr) < 40:
                return f"🧮 {eval(expr)}"
        except Exception:
            pass

        return "🤔 I couldn't find a reliable answer. Try rephrasing."

    # =========================================================
    def _duckduckgo(self, q: str) -> Optional[str]:
        """Try both raw question and extracted topic."""
        try:
            topic = self._extract_topic(q)
            for query in [q, topic]:
                if not query or len(query.strip()) < 2:
                    continue
                try:
                    r = requests.get(
                        f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1",
                        timeout=6,
                    ).json()
                    if r.get("AbstractText"):
                        return f"🦆 {r['AbstractText'][:600]}"
                    if r.get("Answer"):
                        return f"🦆 {r['Answer']}"
                    for t in r.get("RelatedTopics", [])[:3]:
                        if isinstance(t, dict) and t.get("Text"):
                            return f"🦆 {t['Text']}"
                except Exception:
                    continue
        except Exception as e:
            print(f"  ❌ DDG: {e}")
        return None

    def _wikipedia(self, q: str) -> Optional[str]:
        try:
            topic = self._extract_topic(q)
            if len(topic) < 3:
                return None
            try:
                return f"📚 {wikipedia.summary(topic, sentences=6, auto_suggest=True)}"
            except wikipedia.exceptions.DisambiguationError as e:
                for option in e.options[:3]:
                    try:
                        return f"📚 {wikipedia.summary(option, sentences=6)}"
                    except Exception:
                        continue
            except wikipedia.exceptions.PageError:
                for r in wikipedia.search(topic, results=5):
                    try:
                        return f"📚 {wikipedia.summary(r, sentences=6)}"
                    except Exception:
                        continue
        except Exception as e:
            print(f"  ❌ Wiki: {e}")
        return None

    def _extract_topic(self, q: str) -> str:
        ql = q.lower().strip()
        prefixes = [
            "what is", "what are", "what was", "what were",
            "who is", "who are", "who was", "who were",
            "where is", "where are", "where was",
            "when is", "when was", "when did",
            "why is", "why are", "why was", "why did",
            "how to", "how do i", "how can i", "how does", "how did",
            "tell me about", "tell me", "explain", "define", "describe",
            "meaning of", "information about", "info about",
            "what's", "who's", "where's", "when's", "why's", "how's",
            "please", "can you", "could you", "do you know",
        ]
        t = ql
        for p in prefixes:
            if t.startswith(p):
                t = t[len(p):].strip()
                break
        t = t.strip("?.!,;: ")
        return t.title() if t else ql.title()

    def get_emergency_instruction(self, t: str) -> str:
        return self.emergency_instructions.get(t, self.emergency_instructions["general"])

    def get_emergency_type(self, msg: str) -> str:
        m = msg.lower()
        mapping = {
            "heart_attack": ["heart attack", "chest pain", "heart"],
            "bleeding": ["bleed", "blood"],
            "burns": ["burn", "fire"],
            "choking": ["choking", "can't breathe", "not breathing"],
            "stroke": ["stroke", "face drooping", "speech difficulty"],
            "fracture": ["fracture", "broken bone", "broken"],
            "seizure": ["seizure", "fit", "convulsion"],
            "head_injury": ["head injury", "concussion"],
            "poisoning": ["poison", "poisoning", "toxic"],
            "allergic_reaction": ["allergic reaction", "allergy", "anaphylaxis"],
            "unconscious": ["unconscious", "fainting", "passed out"],
        }
        for k, kws in mapping.items():
            for kw in kws:
                if kw in m:
                    return k
        return "general"