"""
rag.py
------
Member 3 — LLM Integration
Fixed version — all bugs resolved
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# BUG FIX 1: API key from .env not hardcoded
# BUG FIX 2: Error handling added
# BUG FIX 5: Consistent model name
# ─────────────────────────────────────────────
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from retriever import search
    RETRIEVER_AVAILABLE = True
except Exception as e:
    print(f"[rag] WARNING: Could not load retriever: {e}")
    RETRIEVER_AVAILABLE = False

# Get API key from .env
API_KEY = os.getenv("GROQ_API_KEY")

def get_client():
    if not GROQ_AVAILABLE:
        raise RuntimeError("groq package not installed. Run: pip install groq")

    if not API_KEY:
        raise RuntimeError("GROQ_API_KEY not found in .env file!")

    return Groq(api_key=API_KEY)


# ─────────────────────────────────────────────
# BUG FIX 3: Check if docs is empty
# BUG FIX 2: Full error handling
# ─────────────────────────────────────────────
def rag_answer(query, company=None, year=None, top_k=5):
    """
    Full RAG pipeline:
    query → FAISS retrieval → Groq answer
    """
    try:
        # Step 1: Retrieve relevant chunks
        if RETRIEVER_AVAILABLE:
            kwargs = {}
            if company:
                kwargs["company"] = company
            if year:
                kwargs["year"] = year
            docs = search(query, top_k=top_k, **kwargs)
        else:
            docs = []

        # Step 2: Build context from retrieved chunks
        if docs:
            context = "\n\n".join(
                f"[Source: {d.get('company','?')} {d.get('year','?')}]\n"
                f"{d.get('text') or d.get('content') or str(d)}"
                for d in docs
            )
        else:
            context = "No relevant context found in the financial documents."

        # Step 3: Build prompt
        prompt = f"""You are a highly skilled financial analyst AI assistant.

Use the context below from real annual reports to answer the question accurately.
If the context doesn't contain the answer, say so clearly.

Context:
{context}

Question: {query}

Provide a clear, structured answer with specific numbers and figures where available."""

        # Step 4: Call Groq
        client = get_client()
        response = client.chat.completions.create(
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

    except RuntimeError as e:
        return f"Configuration Error: {str(e)}"
    except Exception as e:
        return f"Error generating answer: {str(e)}"


# ─────────────────────────────────────────────
# TEST RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    query = "Summarize Apple's financial performance in 2022"
    print(f"Question: {query}")
    print(f"Answer: {rag_answer(query, company='apple', year='2022')}")