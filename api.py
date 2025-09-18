import os
import json
import textwrap
import requests
from fastapi import FastAPI, Query
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults

from fastapi.middleware.cors import CORSMiddleware



load_dotenv()

# ----------------- Config -----------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable.")
if not TAVILY_API_KEY:
    raise ValueError("Missing TAVILY_API_KEY environment variable.")

app = FastAPI(title="Misinformation Checker API", version="1.0")

# allow CORS (dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for dev; lock down in prod to your extension domain or dashboard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Helpers -----------------
def build_llm():
    """Initialize Gemini 2.5 Flash model."""
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def load_page_text(target_url: str) -> str:
    """
    Fetch webpage text.
    Falls back to manual requests if WebBaseLoader fails.
    """
    try:
        loader = WebBaseLoader([target_url])
        docs = loader.load()
        if docs and docs[0].page_content:
            return "\n\n".join([d.page_content for d in docs]).strip()
    except Exception as e:
        print("WebBaseLoader failed, fallback to requests:", e)

    # Manual fallback
    try:
        response = requests.get(target_url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        print("DEBUG: Fallback fetched length =", len(response.text))
        return response.text
    except Exception as e:
        print("ERROR: Could not fetch page with requests:", e)
        return ""

def extract_claims(llm: ChatGoogleGenerativeAI, text: str, k: int) -> list:
    """
    Extract short, atomic factual claims from page text using Gemini.
    Returns list of {"id": int, "text": str}.
    """
    prompt = f"""
Extract up to {k} short, atomic factual claims from this article.
Each claim must be a single verifiable fact.
Return ONLY strict JSON like this:
{{"claims":[{{"id":1,"text":"..."}}]}}

ARTICLE:
{textwrap.shorten(text, width=15000)}
"""
    resp = llm.invoke(prompt)
    raw = resp.content if hasattr(resp, "content") else str(resp)
    print("DEBUG: Raw Gemini response =", raw)

    # Try strict JSON parse first
    try:
        data = json.loads(raw)
        claims = data.get("claims", [])
    except Exception:
        # Fallback: extract JSON block from text
        s = raw.find("{")
        e = raw.rfind("}")
        claims = []
        if s != -1 and e != -1 and e > s:
            try:
                data = json.loads(raw[s:e+1])
                claims = data.get("claims", [])
            except Exception as parse_error:
                print("ERROR parsing JSON fallback:", parse_error)

    # Clean and deduplicate
    clean = []
    seen = set()
    for c in claims:
        t = (c.get("text") or "").strip()
        if t and t not in seen:
            seen.add(t)
            clean.append({"id": c.get("id", len(clean) + 1), "text": t})

    print("DEBUG: Clean claims =", clean)
    return clean

def search_evidence(query: str, k: int) -> list:
    """Search for supporting evidence using Tavily."""
    tool = TavilySearchResults(k=k)
    try:
        results = tool.invoke({"query": query})
        print(f"DEBUG: Tavily results for '{query}' -> {len(results)} items")
        return [r for r in results if r.get("content")]
    except Exception as e:
        print("ERROR Tavily search:", e)
        return []

def verify_claim(llm: ChatGoogleGenerativeAI, claim: str, evidence_items: list, mode: str):
    """
    Verify claim against evidence and classify as SUPPORTS / REFUTES / NOT_ENOUGH_INFO.
    """
    if not evidence_items:
        return {
            "label": "NOT_ENOUGH_INFO",
            "confidence": 0.25,
            "rationale": "No evidence found",
            "citations": []
        }

    evidence_text = "\n\n".join(
        [f"[{i+1}] {e['url']}\n{e['content']}" for i, e in enumerate(evidence_items)]
    )
    strict_note = {
        "conservative": "Be conservative; prefer NOT_ENOUGH_INFO unless strong evidence exists.",
        "balanced": "Be balanced; weigh the evidence carefully.",
        "aggressive": "Be decisive; pick SUPPORTS or REFUTES when evidence leans clearly."
    }[mode]

    verifier_prompt = f"""
Classify the CLAIM based on EVIDENCE into: SUPPORTS, REFUTES, or NOT_ENOUGH_INFO.
Return JSON only: {{"label":"...","confidence":0.0,"rationale":"...","citations":["...","..."]}}

CLAIM:
{claim}

EVIDENCE:
{evidence_text}

Guidelines:
- {strict_note}
- Cite only the provided URLs.
"""
    resp = llm.invoke(verifier_prompt)
    raw = resp.content if hasattr(resp, "content") else str(resp)

    # Try strict JSON parse
    try:
        data = json.loads(raw)
    except Exception:
        s = raw.find("{")
        e = raw.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                data = json.loads(raw[s:e+1])
            except:
                data = {}
        else:
            data = {}

    # Validate fields
    label = str(data.get("label", "NOT_ENOUGH_INFO")).upper().strip()
    if label not in {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}:
        label = "NOT_ENOUGH_INFO"

    try:
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except Exception:
        confidence = 0.5

    rationale = data.get("rationale", "").strip() or "No rationale provided."
    citations = data.get("citations", [])
    allowed = {e["url"] for e in evidence_items if e.get("url")}
    citations = [c for c in citations if c in allowed]

    return {
        "label": label,
        "confidence": confidence,
        "rationale": rationale,
        "citations": citations or [evidence_items[0]["url"]]
    }

# ----------------- API Response Model -----------------
class VerifyResponse(BaseModel):
    id: int
    claim: str
    label: str
    confidence: float
    rationale: str
    citations: List[str]

# ----------------- Routes -----------------
@app.get("/")
def root():
    return {"message": "Misinformation Checker API is running. Visit /docs to test."}

@app.get("/verify", response_model=List[VerifyResponse])
def verify_url(
    url: str = Query(..., description="Target article URL"),
    max_claims: int = Query(5, ge=1, le=10),
    results_per_claim: int = Query(5, ge=2, le=10),
    strictness: str = Query("balanced", regex="^(conservative|balanced|aggressive)$")
):
    llm = build_llm()

    # Step 1: Load page
    page_text = load_page_text(url)
    print("DEBUG: Loaded page length =", len(page_text))
    if not page_text:
        print("ERROR: No page content extracted")
        return []

    # Step 2: Extract claims
    claims = extract_claims(llm, page_text, max_claims)
    if not claims:
        print("ERROR: No claims extracted")
        return []

    # Step 3: Verify each claim
    results = []
    for c in claims:
        evidence = search_evidence(c["text"], results_per_claim)
        print(f"DEBUG: Evidence count for claim {c['id']} = {len(evidence)}")
        verdict = verify_claim(llm, c["text"], evidence, strictness)
        results.append({
            "id": c["id"],
            "claim": c["text"],
            **verdict
        })

    print("DEBUG: Final results =", results)
    return results
