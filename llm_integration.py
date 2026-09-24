"""
llm_integration.py
------------------
Member 3 — LLM Integration
Fixed version — converted from Gemini to Groq

BUG FIX 4: API key moved to .env (Using GROQ_API_KEY)
BUG FIX 5: Consistent model name (llama-3.3-70b-versatile)
BUG FIX 6: Wrapped in proper class and functions
BUG FIX 7: FinanceRAG class now exists for rag_runner.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("[llm] WARNING: groq package not installed. Run: pip install groq")

try:
    from retriever import search
    RETRIEVER_AVAILABLE = True
except Exception as e:
    print(f"[llm] WARNING: retriever not available: {e}")
    RETRIEVER_AVAILABLE = False

API_KEY = os.getenv("GROQ_API_KEY")


# ─────────────────────────────────────────────
# BUG FIX 7: FinanceRAG class added
# ─────────────────────────────────────────────
class FinanceRAG:
    """
    Full RAG pipeline class.
    Supports Groq with graceful error handling.
    """

    def __init__(self, top_k=5, backend="groq", ollama_model="llama3"):
        self.top_k = top_k
        self.backend = backend
        self.ollama_model = ollama_model

        if not API_KEY:
            print("[FinanceRAG] WARNING: GROQ_API_KEY not set in .env")

        if GROQ_AVAILABLE and API_KEY:
            self.client = Groq(api_key=API_KEY)
        else:
            self.client = None

    def retrieve(self, query, company=None, year=None, top_k=None):
        """Retrieve top-k chunks from FAISS."""
        if not RETRIEVER_AVAILABLE:
            return []
        try:
            kwargs = {}
            if company:
                kwargs["company"] = company
            if year:
                kwargs["year"] = year
            return search(query, top_k=top_k or self.top_k, **kwargs)
        except Exception as e:
            print(f"[FinanceRAG] Retrieval error: {e}")
            return []

    def build_context(self, docs):
        """Build context string from retrieved docs."""
        if not docs:
            return "No relevant context found."
        return "\n\n".join(
            f"[{d.get('company','?')} — {d.get('year','?')}]\n"
            f"{d.get('text') or d.get('content') or str(d)}"
            for d in docs
        )

    def build_prompt(self, query, context):
        """Build the full prompt for the LLM."""
        return f"""You are a highly skilled financial analyst AI.

Use the context below from real annual reports to answer accurately.
If the answer is not in the context, say so clearly.

Context:
{context}

Question: {query}

Answer with specific numbers and figures where available."""

    def ask(self, query, company=None, year=None, top_k=None):
        """
        Full RAG pipeline: retrieve + generate.
        Returns dict with answer and source chunks.
        """
        # Step 1: Retrieve
        docs = self.retrieve(query, company=company, year=year, top_k=top_k)

        # Step 2: Build context
        context = self.build_context(docs)

        # Step 3: Build prompt
        prompt = self.build_prompt(query, context)

        # Step 4: Generate with Groq
        try:
            if not self.client:
                return {
                    "answer": "Groq client not initialized. Check GROQ_API_KEY in .env",
                    "chunks": docs,
                    "model": "none"
                }

            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1024
            )
            answer = response.choices[0].message.content

        except Exception as e:
            answer = f"Error generating answer: {str(e)}"

        return {
            "answer": answer,
            "chunks": docs,
            "model": "llama-3.3-70b-versatile"
        }

    def ask_personal(self, query, personal_context):
        """
        Answer questions about personal finance data.
        Uses personal_context string instead of FAISS retrieval.
        """
        prompt = f"""You are a personal finance advisor AI.

Here is the user's financial data:
{personal_context}

Question: {query}

Give personalized, practical advice based on their actual data.
Include specific numbers where relevant. Be encouraging and helpful."""

        try:
            if not self.client:
                return "Groq client not initialized. Check GROQ_API_KEY in .env"

            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1024
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"Error: {str(e)}"


# ─────────────────────────────────────────────
# Simple function interface (for app.py)
# ─────────────────────────────────────────────
_rag_instance = None

def get_rag():
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = FinanceRAG()
    return _rag_instance


def ask_company(query, company=None, year=None):
    """Simple wrapper for company RAG questions."""
    return get_rag().ask(query, company=company, year=year)["answer"]


def ask_personal(query, personal_context):
    """Simple wrapper for personal finance questions."""
    return get_rag().ask_personal(query, personal_context)


# ─────────────────────────────────────────────
# TEST RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    rag = FinanceRAG()
    result = rag.ask(
        "Summarize Apple's financial performance in 2022",
        company="apple",
        year="2022"
    )
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['chunks'])} chunks retrieved")