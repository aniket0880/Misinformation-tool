# main.py
import os
import json
import asyncio
import textwrap
import streamlit as st

# ----- Prevent "no current event loop" in Streamlit thread -----
asyncio.set_event_loop(asyncio.new_event_loop())

# ----- Read API keys (secrets first, then env). Fallback to UI fields. -----
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY", ""))

# ----- Lazy imports after keys/Streamlit setup -----
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults

# ------------------------- UI -------------------------
st.set_page_config(page_title="Misinformation Checker", page_icon="✅", layout="wide")
st.title("🔎 Misinformation Checker (Gemini + Web Evidence)")

with st.expander("API keys (only needed once per session)"):
    if not GOOGLE_API_KEY:
        GOOGLE_API_KEY = st.text_input("GOOGLE_API_KEY (Gemini)", type="password")
    if not TAVILY_API_KEY:
        TAVILY_API_KEY = st.text_input("TAVILY_API_KEY (Tavily Search)", type="password")

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

url = st.text_input("Paste a URL to verify:", placeholder="https://example.com/article")
colA, colB, colC = st.columns(3)
with colA:
    max_claims = st.slider("Max claims to check", 1, 10, 5)
with colB:
    results_per_claim = st.slider("Evidence sources per claim", 2, 10, 5)
with colC:
    strictness = st.selectbox("Strictness", ["balanced", "conservative", "aggressive"], index=0)

run_btn = st.button("Run verification")

# --------------------- Helpers ---------------------
def build_llm():
    # Gemini 2.5 Pro (chat) via LangChain
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def load_page_text(target_url: str) -> str:
    loader = WebBaseLoader([target_url])
    docs = loader.load()
    text = "\n\n".join([d.page_content for d in docs if d.page_content])
    return text.strip()

def extract_claims(llm: ChatGoogleGenerativeAI, text: str, k: int) -> list:
    prompt = f"""
You are a professional fact-checking assistant.
From the article text below, extract up to {k} short, atomic, check-worthy factual claims.
- Each claim should be a single verifiable statement (avoid opinions).
- Include date/time/place/quantity if present.
- Output JSON as: {{"claims": [{{"id": 1, "text": "..."}}, ...]}}.

ARTICLE:
{textwrap.shorten(text, width=15000)}
"""
    resp = llm.invoke(prompt)
    # Try to parse JSON robustly
    content = resp.content if hasattr(resp, "content") else str(resp)
    try:
        data = json.loads(content)
        claims = data.get("claims", [])
    except Exception:
        # Fallback: try to find JSON block
        start = content.find("{")
        end = content.rfind("}")
        claims = []
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(content[start:end+1])
                claims = data.get("claims", [])
            except Exception:
                pass
    # Normalize
    clean = []
    seen = set()
    for c in claims:
        t = (c.get("text") or "").strip()
        if t and t not in seen:
            seen.add(t)
            clean.append({"id": c.get("id", len(clean)+1), "text": t})
    return clean

def search_evidence(query: str, k: int) -> list:
    """
    Uses TavilySearchResults (LangChain tool) to get top-k web results.
    Returns list of dicts: {"url": ..., "content": ...}
    """
    tool = TavilySearchResults(k=k)  # requires TAVILY_API_KEY in env
    try:
        results = tool.invoke({"query": query})
        # results is typically a list of {"url": str, "content": str}
        return [r for r in results if r.get("content")]
    except Exception as e:
        return []

def verify_claim(llm: ChatGoogleGenerativeAI, claim: str, evidence_items: list, mode: str):
    """
    evidence_items: list of {"url": ..., "content": ...}
    Returns dict {label, confidence, rationale, citations}
    """
    if not evidence_items:
        return {
            "label": "NOT_ENOUGH_INFO",
            "confidence": 0.25,
            "rationale": "No evidence could be retrieved.",
            "citations": []
        }
    evidence_text = "\n\n".join(
        [f"[{i+1}] URL: {e['url']}\nSNIPPET:\n{e['content']}" for i, e in enumerate(evidence_items)]
    )
    strict_note = {
        "conservative": "Be conservative; prefer NOT_ENOUGH_INFO unless evidence is strong.",
        "balanced": "Be balanced; decide based on weight and quality of evidence.",
        "aggressive": "Be decisive; select SUPPORTS/REFUTES when evidence leans clearly."
    }[mode]

    verifier_prompt = f"""
You are a fact-checking verifier. Determine whether the CLAIM is supported by the EVIDENCE.
Classify as one of: SUPPORTS, REFUTES, NOT_ENOUGH_INFO.
Return strict JSON only with keys: label, confidence (0.0-1.0), rationale (<=2 sentences), citations (list of URLs you used).

Guidelines:
- {strict_note}
- Cite only from the provided URLs.
- If sources conflict or are low quality or unrelated, use NOT_ENOUGH_INFO.
- If the claim includes a specific date/quantity/place, ensure the evidence matches those details.

CLAIM:
{claim}

EVIDENCE:
{evidence_text}

JSON:
{{"label": "...", "confidence": 0.0, "rationale": "...", "citations": ["...", "..."]}}
"""
    resp = llm.invoke(verifier_prompt)
    raw = resp.content if hasattr(resp, "content") else str(resp)
    # Parse JSON
    try:
        data = json.loads(raw)
    except Exception:
        # try to extract JSON block
        s = raw.find("{")
        e = raw.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                data = json.loads(raw[s:e+1])
            except Exception:
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
    rationale = data.get("rationale", "").strip() or "No rationale."
    citations = data.get("citations", [])
    # Filter to provided URLs only
    allowed = {e["url"] for e in evidence_items if e.get("url")}
    citations = [c for c in citations if c in allowed]

    return {
        "label": label,
        "confidence": confidence,
        "rationale": rationale,
        "citations": citations or [evidence_items[0]["url"]]
    }

# --------------------- Pipeline ---------------------
def run_pipeline(target_url: str, k_claims: int, k_sources: int, mode: str):
    if not GOOGLE_API_KEY:
        st.error("Missing GOOGLE_API_KEY. Add it in the expander above.")
        return
    if not TAVILY_API_KEY:
        st.error("Missing TAVILY_API_KEY. Add it in the expander above.")
        return

    llm = build_llm()

    with st.spinner("Loading page..."):
        page_text = load_page_text(target_url)
        if not page_text:
            st.error("Could not load or parse the page.")
            return

    with st.spinner("Extracting claims..."):
        claims = extract_claims(llm, page_text, k_claims)
        if not claims:
            st.error("No check-worthy claims were extracted.")
            return

    st.subheader("🧩 Extracted Claims")
    for c in claims:
        st.markdown(f"- **Claim {c['id']}:** {c['text']}")

    st.subheader("🧪 Verification Results")
    results = []
    for c in claims:
        with st.spinner(f"Verifying claim {c['id']}..."):
            evidence = search_evidence(c["text"], k_sources)
            verdict = verify_claim(llm, c["text"], evidence, mode)
            results.append({"id": c["id"], "claim": c["text"], **verdict})

            # Display card
            st.markdown("---")
            badge = {"SUPPORTS": "✅", "REFUTES": "❌", "NOT_ENOUGH_INFO": "⚠️"}[verdict["label"]]
            st.markdown(f"### {badge} Claim {c['id']}: {verdict['label']} (conf: {verdict['confidence']:.2f})")
            st.write(verdict["rationale"])
            st.caption("Citations:")
            for u in verdict["citations"]:
                st.markdown(f"- {u}")

    # Overall summary
    supports = sum(1 for r in results if r["label"] == "SUPPORTS")
    refutes = sum(1 for r in results if r["label"] == "REFUTES")
    neis    = sum(1 for r in results if r["label"] == "NOT_ENOUGH_INFO")
    st.markdown("## 📊 Summary")
    st.write(f"SUPPORTS: **{supports}**, REFUTES: **{refutes}**, NOT_ENOUGH_INFO: **{neis}**")
    st.info("This tool provides a best-effort, source-backed assessment. Always review sources yourself—especially for health, legal, or safety-critical claims.")

# --------------------- Run ---------------------
if run_btn and url:
    run_pipeline(url, max_claims, results_per_claim, strictness)
elif run_btn and not url:
    st.warning("Please paste a URL.")
