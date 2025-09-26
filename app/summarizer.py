# app/summarizer.py
from typing import Sequence, Optional
import os

def lead_n(sentences: Sequence[str], n: int = 3, max_chars: int = 500) -> str:
    out = " ".join(sentences[:n]).strip()
    return out[:max_chars]

def polish_with_llm(text: str, model: str="meta-llama/llama-3.1-8b-instruct") -> str:
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        return text
    from together import Together
    client = Together(api_key=api_key)
    prompt = (
            "Rewrite the following extractive news snippet into 2–3 concise sentences. "
            "Keep it factual, no new info, no advice.\\n\\n" + text
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=0.2, max_tokens=160
    )
    return resp.choices[0].message.content.strip()

def summarize(sentences: Sequence[str], llm: bool=False) -> str:
    base = lead_n(sentences, 3, 500)
    return polish_with_llm(base) if llm else base