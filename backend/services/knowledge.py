import random
import requests
import re
from datetime import datetime
from typing import Optional

from services.gemini_api import GeminiService

try:
    from services.ev_rag.qa_service import EVRAGService
    EV_RAG_AVAILABLE = True
except Exception as e:
    EV_RAG_AVAILABLE = False
    print(f"⚠️ EV_RAG unavailable: {e}")

try:
    import wikipedia
    WIKI_OK = True
except ImportError:
    WIKI_OK = False
    print("⚠️ wikipedia-api not installed")


# EV detection
EV_STRONG = [
    "kia", "ev3", "ev5", "ev6", "ev9", "e-niro", "niro",
    "regenerative braking", "regen braking", "battery level",
    "charging port", "ac charging", "dc charging",
    "smart key", "owners manual", "owner's manual",
    "electric vehicle", "electric car",
]
EV_MEDIUM = ["battery", "charging", "charge", "range", "kwh", "motor", "hybrid", "infotainment", "warranty"]
EV_WEAK = ["ev", "kw"]


def _ev_score(q: str) -> int:
    score = 0
    for kw in EV_STRONG:
        if kw in q:
            score += 10
    for kw in EV_MEDIUM:
        if kw in q:
            score += 3
    for kw in EV_WEAK:
        if kw in q:
            score += 1
    return score


class KnowledgeService:
    def __init__(self):
        self.gemini = GeminiService()
        self.ev_rag = EVRAGService() if EV_RAG_AVAILABLE else None

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

    # =============================================================
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

                # 3. EV_RAG — route + rewrite with AI for clean output
        ev_score = _ev_score(q)
        if ev_score > 0:
            print(f"  🚗 EV score: {ev_score}")

        if self.ev_rag and ev_score >= 3:
            print("  🚗 Trying EV_RAG...")
            try:
                result = self.ev_rag.ask(question)
                if result:
                    conf = result.get("confidence", "low")
                    print(f"  ✅ EV_RAG ({result['score']}, {conf})")

                    # Ask AI to answer using chunks + its own knowledge
                    cleaned, used_manual = self._rewrite_ev_answer(
                        question=question,
                        raw_answer=result["answer"],
                        manual=result.get("manual", "EV manual"),
                    )

                    if cleaned:
                        print("  ✨ AI cleaned up the answer")
                        response = f"🚗 {cleaned}"

                        # Only show source if manual was actually used
                        if used_manual and result.get("page_start"):
                            response += (
                                f"\n\n📖 Source: {result.get('manual', 'EV manual')} "
                                f"(page {result['page_start']})"
                            )
                        return response

                    return EVRAGService.format_answer(result)
            except Exception as e:
                print(f"  ❌ EV_RAG error: {e}")

        # 4. DUCKDUCKGO
        print("  🦆 Trying DuckDuckGo...")
        ddg = self._duckduckgo(q)
        if ddg:
            print("  ✅ DDG answered")
            return ddg

        # 5. WIKIPEDIA
        if WIKI_OK:
            print("  📚 Trying Wikipedia...")
            wiki = self._wikipedia(q)
            if wiki:
                print("  ✅ Wiki answered")
                return wiki

        # 6. AI
        print("  🤖 Trying AI (OpenRouter)...")
        ai = self.gemini.get_response(question)
        if ai:
            print("  ✅ AI answered")
            return f"🤖 {ai}"

        # 7. Math
        try:
            expr = re.sub(r"[^0-9+\-*/(). ]", "", question)
            if any(op in expr for op in "+-*/") and len(expr) < 40:
                return f"🧮 {eval(expr)}"
        except Exception:
            pass

        return "🤔 I couldn't find a reliable answer. Try rephrasing."

        def _rewrite_ev_answer(self, question: str, raw_answer: str, manual: str):
            """Returns (answer_text, used_manual_flag)."""
        from services.ev_rag.text_cleaner import clean_answer

        all_chunks = []
        if self.ev_rag:
            for entry in self.ev_rag.retrievers:
                try:
                    results = entry["retriever"].search(question, top_k=6)
                    for r in results:
                        chunk = r["chunk"]
                        text = chunk.get("text", "").strip()
                        if not text or len(text) < 50:
                            continue
                        if "table of contents" in text.lower()[:100]:
                            continue
                        all_chunks.append({
                            "text": clean_answer(text)[:1500],
                            "heading": chunk.get("heading", ""),
                            "page": chunk.get("page_start") or chunk.get("page"),
                            "manual": entry["name"],
                            "score": r["final_score"],
                        })
                except Exception as e:
                    print(f"  ⚠️ Retrieval: {e}")

        all_chunks.sort(key=lambda c: c["score"], reverse=True)
        top_chunks = all_chunks[:5]

        if top_chunks:
            context = "\n\n---\n\n".join(
                f"[{c['manual']} | {c['heading']} | p.{c['page']}]\n{c['text']}"
                for c in top_chunks
            )[:6000]
        else:
            context = "(No relevant excerpts found.)"

        prompt = f"""You are MADO, an expert assistant for Kia electric vehicles.

Question: "{question}"

RELEVANT MANUAL EXCERPTS:
{context}

Instructions:
1. First, try to answer using the excerpts above.
2. If the excerpts clearly answer the question, base your answer on them and START your response with: [MANUAL]
3. If the excerpts don't contain the answer, use your own knowledge to answer. START your response with: [GENERAL]
4. Give a clear answer in 2-5 sentences.
5. Fix broken words, missing spaces, weird symbols.

Answer:"""

        try:
            raw = self.gemini.get_response(prompt)
            if not raw:
                return None, False

            used_manual = raw.startswith("[MANUAL]")
            # Remove the marker
            cleaned = raw.replace("[MANUAL]", "").replace("[GENERAL]", "").strip()
            return cleaned, used_manual
        except Exception:
            return None, False
    # =============================================================
    def _duckduckgo(self, q: str) -> Optional[str]:
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
        except Exception:
            pass
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
        except Exception:
            pass
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