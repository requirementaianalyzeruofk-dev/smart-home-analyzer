import os
import io
import re
import math
import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple, Dict, Any
# *** IMPORT PLOTLY FOR PIE CHART ***
import plotly.express as px

# optional imports for file parsing
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx
except Exception:
    docx = None

# ------------------------------
# Configuration
# ------------------------------
MODEL_PATH = "finetuned_roberta_promise"  # must exist (folder with model + tokenizer)
CHUNK_SIZE = 32  # batch size for tokenization/prediction (adjust for memory)

# ------------------------------
# UI: page config & styling (Enhanced for HCI)
# ------------------------------
st.set_page_config(page_title="AI-Based Requirements Analyzer",
                   layout="wide")
st.markdown("""
    <style>
    /* General Styling */
    .stApp {background-color: #F8FAFC; color: #0F172A; font-family: 'Segoe UI', sans-serif;}
    h1 {text-align:center; color:#1E3A8A; font-size: 2.5em; margin-bottom: 0px;}
    .subtitle {text-align:center; color:#475569; font-size: 1.1em; margin-top: 0px; margin-bottom: 20px;}
    .block {padding:10px; border-radius:8px; background:#FFFFFF; box-shadow: 0 1px 3px rgba(0,0,0,0.04);}
    
    /* Text Area Styling (Input Box) */
    div[data-testid="stTextarea"] textarea {
        border: 2px solid #93C5FD; /* Light blue border */
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 70, 190, 0.1);
        transition: border-color 0.3s;
    }
    div[data-testid="stTextarea"] textarea:focus {
        border-color: #1E3A8A; /* Darker blue on focus */
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #1E3A8A;
        background-color: #1E3A8A;
        color: white;
        padding: 10px 24px;
        font-weight: bold;
        transition: background-color 0.3s, transform 0.1s;
    }
    .stButton>button:hover {
        background-color: #1D4ED8; /* Darker blue on hover */
        border-color: #1D4ED8;
        transform: translateY(-1px);
    }
    </style>
    """, unsafe_allow_html=True)

# العنوان وتفصيل النظام تحت العنوان
st.title("💡 AI-Based Requirements Analyzer")
st.markdown("<p class='subtitle'>for Smart Home Access Systems</p>", unsafe_allow_html=True)

st.write("Classify requirements (Functional vs Non-Functional), detect conflicts and get smart suggestions. Supports single input or batch files (CSV / PDF / DOCX).")

st.markdown("---")

# ------------------------------
# Model loading (cached)
# ------------------------------
@st.cache_resource
def load_model_and_tokenizer(model_path: str):
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model folder not found: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

try:
    tokenizer, model = load_model_and_tokenizer(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model from '{MODEL_PATH}': {e}")
    st.stop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ------------------------------
# Helpers: file parsers
# ------------------------------
def read_csv_file(uploaded) -> pd.DataFrame:
    uploaded.seek(0)
    return pd.read_csv(uploaded)

def read_docx_file(uploaded) -> List[str]:
    if docx is None:
        raise RuntimeError("python-docx is not installed. Install with pip install python-docx.")
    doc = docx.Document(uploaded)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return paragraphs

def read_pdf_file(uploaded) -> List[str]:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is not installed. Install with pip install pdfplumber.")
    texts = []
    # يجب إعادة التعيين قبل القراءة لضمان العمل بعد محاولات سابقة
    uploaded.seek(0) 
    with pdfplumber.open(io.BytesIO(uploaded.read())) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                # split pages into lines / sentences
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                texts.extend(lines)
    return texts

def normalize_text(t: str) -> str:
    return re.sub(r'\s+', ' ', t.strip()).lower()

# ------------------------------
# Classification function (MODIFIED to include Confidence Score)
# ------------------------------
def classify_requirements_texts(texts: List[str]) -> List[Tuple[str, float]]:
    """Classify a list of strings and return (Label, Confidence)."""
    # NOTE: Functional (1) vs Non-Functional (0) in training
    labels_map = {0: "Non-Functional", 1: "Functional"} 
    results = [] # list of (label, confidence)
    
    # process in chunks
    for i in range(0, len(texts), CHUNK_SIZE):
        batch = texts[i:i+CHUNK_SIZE]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        
        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            # Calculate probabilities and max confidence
            probs = torch.softmax(logits, dim=1)
            confidences, pred_ids = torch.max(probs, dim=1)
            
            pred_ids = pred_ids.cpu().tolist()
            confidences = confidences.cpu().tolist()

            for pid, conf in zip(pred_ids, confidences):
                label = labels_map.get(int(pid), "Unknown")
                results.append((label, conf))
    return results

# ------------------------------
# Subcategory detection (keywords-based)
# ------------------------------
SUBCATEGORY_KEYWORDS = {
    "Security": ["encrypt", "encryption", "auth", "authentication", "secure", "tls", "aes", "sha", "password", "credential", "vulnerability"],
    "Performance": ["response time", "latency", "throughput", "ms", "seconds", "performance", "fast", "speed"],
    "Usability": ["easy", "user-friendly", "intuitive", "usability", "look & feel", "ui", "ux", "interface"],
    "Reliability": ["uptime", "availability", "fault", "fail", "recover", "recovery", "reliable", "error"],
    "Maintainability": ["maintain", "upgrade", "update", "patch", "maintainability", "modular"],
    "Portability": ["portable", "platform", "os", "windows", "linux", "android", "ios", "browser"],
    "Scalability": ["scale", "concurrent", "users", "load", "capacity", "handle"],
    "Privacy": ["personal data", "pii", "privacy", "GDPR", "consent", "sensitive"],
}

def detect_subcategory(text: str) -> str:
    t = normalize_text(text)
    scores = {}
    for cat, kws in SUBCATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                scores[cat] = scores.get(cat, 0) + 1
    if not scores:
        return "Other Non-Functional" # Better label for HCI
    # return the category with max matches
    return max(scores.items(), key=lambda x: x[1])[0]

# ------------------------------
# Intelligent suggestions (heuristics)
# ------------------------------
AMBIGUOUS_WORDS = ["fast", "good", "easy", "secure", "efficient", "robust", "reliable", "adequate", "simple", "quickly"]

def intelligent_suggestion(text: str, predicted_label: str) -> List[str]:
    suggestions = []
    t = normalize_text(text)

    # 1. If ambiguous wording present
    for w in AMBIGUOUS_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", t):
            suggestions.append(f"Clarify the ambiguous term **'{w}'** with measurable criteria (e.g., response time, % uptime, score > 80).")

    # 2. If Non-Functional but missing measurable metric
    if predicted_label == "Non-Functional":
        # Check if any measurable metric is present
        if not any(re.search(r'\d', t) or k in t for k in ["%", "percent", "ms", "seconds", "score", "level"]):
            # Suggest metric, especially if it's a known NFR subcategory (heuristic)
            if any(k in t for k in ["performance", "fast", "slow", "latency", "throughput", "uptime", "reliable", "secure"]):
                 suggestions.append("This **Non-Functional** requirement needs a specific, measurable target (e.g., 'response time ≤ 2 seconds' or '99.9% uptime').")

    # 3. If Functional but lacks commitment ('should' vs 'must')
    if predicted_label == "Functional":
        if "should " in t and "must " not in t and "shall " not in t:
            suggestions.append("If this is a mandatory feature, replace 'should' with **'must'** or **'shall'** for clarity.")

    # 4. Encourage Acceptance Criteria
    if "acceptance" not in t and "criteria" not in t:
        suggestions.append("💡 Always consider adding clear **acceptance criteria** (conditions that prove the requirement is met) to improve testability.")

    # Remove duplicates
    seen = set()
    result = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result

# ------------------------------
# Conflict detection (simple rules)
# ------------------------------
# Contradiction pairs: using a stronger list
CONTRADICTION_PAIRS = [("allow", "deny"), ("enable", "disable"), ("must not", "shall"), ("shall", "must not")]

def detect_conflicts_in_list(requirements_list: List[str]) -> List[str]:
    """Return a set of human-readable conflict warnings detected across the list."""
    conflicts = []
    normalized = [normalize_text(r) for r in requirements_list]

    # 1) exact duplicates (same meaning)
    seen = {}
    for idx, r in enumerate(normalized):
        seen.setdefault(r, []).append(idx)
    for text, idxs in seen.items():
        if len(idxs) > 1:
            conflicts.append(f"🔴 **Duplicate** found (appears {len(idxs)} times): \"{requirements_list[idxs[0]][:70]}...\"")

    # 2) contradictory words within same requirement
    for idx, r in enumerate(normalized):
        for a, b in CONTRADICTION_PAIRS:
            if a in r and b in r:
                conflicts.append(f"⚠️ **Internal Conflict** in requirement #{idx+1}: contains both '{a}' and '{b}'.")

    # 3) cross-requirement contradictions (simple heuristic)
    # Check for opposite modal verbs
    must_not_reqs = [r for r in normalized if "must not" in r or "shall not" in r]
    must_reqs = [r for r in normalized if "must" in r and "not" not in r]

    if must_not_reqs and must_reqs:
        # A very basic check: if we have both 'must' and 'must not' in the document
        conflicts.append("⚠️ **Modal Conflict** detected: some requirements use **'must'** while others use **'must not'**. Review mandatory requirements.")

    return conflicts

# ------------------------------
# UI layout
# ------------------------------
col1, col2 = st.columns([2, 1])
with col1:
    st.header("Input Method")
    input_mode = st.radio("Choose input method:", ("Manual single requirement", "Upload file (CSV / PDF / DOCX)"))

# (Hidden model details for cleaner look as requested)

st.markdown("----")

# ------------------------------
# Manual single requirement flow
# ------------------------------
result_df = None
if input_mode == "Manual single requirement":
    st.subheader("Enter Requirement Text")
    
    # Text area with a nice border (via CSS)
    manual_text = st.text_area("Type requirement here:", height=140, placeholder="e.g. The system shall encrypt all user credentials at rest.")
    
    colA, colB, colC = st.columns([1.5, 1.5, 4]) # Adjust columns for button size
    
    # Run button (enhanced styling applied via CSS)
    with colA:
        run_btn = st.button("🔍 Analyze & Suggest", help="Classify the requirement, detect conflicts, and generate suggestions.")
    
    # Clear button (custom styling for different look)
    with colB:
        # Using markdown to create a secondary-style button
        st.markdown(
            f"""
            <style>
                .stButton button[data-testid*="stButton-after-Analyze & Suggest"] {{
                    background-color: #F8FAFC; 
                    color: #475569; 
                    border: 1px solid #CBD5E1;
                }}
                .stButton button[data-testid*="stButton-after-Analyze & Suggest"]:hover {{
                    background-color: #E2E8F0;
                }}
            </style>
            """, unsafe_allow_html=True
        )
        clear_btn = st.button("Clear Input")

    if clear_btn:
        # Requires rerun to clear state, but streamlit handles this automatically
        pass

    if run_btn:
        if not manual_text or not manual_text.strip():
            st.warning("⚠️ Please enter a requirement first.")
        else:
            reqs = [manual_text.strip()]
            with st.spinner("Classifying and analyzing..."):
                # Returns list of (label, confidence)
                results = classify_requirements_texts(reqs)
                pred, conf = results[0] 
                subcat = detect_subcategory(reqs[0]) if pred == "Non-Functional" else "N/A"
                suggestions = intelligent_suggestion(reqs[0], pred)
                conflicts = detect_conflicts_in_list(reqs)

            # --- Display Results ---
            st.success(f"✅ Analysis Complete.")
            
            # Use columns for better alignment of key metrics
            col_pred, col_conf, col_subcat = st.columns(3)
            with col_pred:
                 st.metric(label="Predicted Label", value=pred, delta=f"{conf*100:.1f}% Confidence")
            with col_conf:
                 # Display Confidence explicitly
                 st.metric(label="Confidence Score", value=f"{conf*100:.2f}%")
            with col_subcat:
                 st.metric(label="Subcategory (NFRs)", value=subcat)

            st.markdown("### Smart Suggestions")
            if suggestions:
                for s in suggestions:
                    st.info(s)
            else:
                st.info("👍 Requirement looks well-specified (no immediate ambiguities or mandatory criteria suggestions).")

            st.markdown("### Conflict Warnings")
            if conflicts:
                for c in conflicts:
                    st.warning(c)
            else:
                st.info("🎉 No obvious conflicts detected for this single requirement.")


# ------------------------------
# File upload / batch flow
# ------------------------------
else:
    st.subheader("Upload Requirements Document (CSV, PDF, or DOCX)")
    uploaded = st.file_uploader("Choose a file", type=["csv", "pdf", "docx"])
    parse_preview = None
    if uploaded is not None:
        filename = uploaded.name.lower()
        st.markdown(f"**File Uploaded:** `{uploaded.name}`")
        # parse by extension
        try:
            if filename.endswith(".csv"):
                df_in = read_csv_file(uploaded)
                # try guess column name variations
                if "requirement" not in df_in.columns:
                    lower_cols = [c.lower() for c in df_in.columns]
                    candidates = [c for c, lc in zip(df_in.columns, lower_cols) if "require" in lc or "text" == lc or "name" in lc]
                    if candidates:
                        use_col = candidates[0]
                        st.info(f"Using column **'{use_col}'** from CSV as the requirement text column.")
                        df_in = df_in.rename(columns={use_col: "requirement"})
                    else:
                        st.error("CSV does not contain a 'requirement' column or a suitable alternative. Please check column names.")
                        df_in = None
                if df_in is not None:
                    df_in["requirement"] = df_in["requirement"].astype(str).str.strip()
                    df_in = df_in[df_in["requirement"].str.len() > 5] # remove short strings
                    st.write(f"Preview (first 5 rows of {len(df_in)} requirements):")
                    st.dataframe(df_in.head(5))
                    parse_preview = df_in
            elif filename.endswith(".docx"):
                # (Reading docx and pdf logic remains the same for simplicity)
                if docx is None: st.error("`python-docx` not installed. Install with pip install python-docx.")
                else:
                    paragraphs = read_docx_file(uploaded)
                    df_in = pd.DataFrame({"requirement": [p for p in paragraphs if len(p) > 5]})
                    st.write(f"Preview (first 5 rows of {len(df_in)} requirements):")
                    st.dataframe(df_in.head(5))
                    parse_preview = df_in
            elif filename.endswith(".pdf"):
                if pdfplumber is None: st.error("`pdfplumber` not installed. Install with pip install pdfplumber.")
                else:
                    paragraphs = read_pdf_file(uploaded)
                    df_in = pd.DataFrame({"requirement": [p for p in paragraphs if len(p) > 5]})
                    st.write(f"Preview (first 5 rows of {len(df_in)} requirements):")
                    st.dataframe(df_in.head(5))
                    parse_preview = df_in
            else:
                st.error("Unsupported file type. Please upload CSV, PDF, or DOCX.")
        except Exception as e:
            st.error(f"Error parsing file: {e}")
            parse_preview = None

        # If parse succeeded
        if parse_preview is not None:
            st.markdown("---")
            col_run, col_save = st.columns([1, 1])
            with col_run:
                analyze_btn = st.button("🔍 Analyze & Suggest (Batch)", help="Run classification, conflict detection, and suggestion generation on all requirements.")
            with col_save:
                download_name = st.text_input("Output filename (CSV):", value="predicted_results.csv")

            if analyze_btn:
                df = parse_preview.copy()
                texts = df["requirement"].astype(str).tolist()
                total = len(texts)
                st.info(f"Classifying **{total}** requirements...")
                progress = st.progress(0)
                
                results_all = [] # list of (label, confidence)
                subcats = []
                suggestions_all = []
                
                # classify in chunks and build suggestions & subcats
                for start in range(0, total, CHUNK_SIZE):
                    chunk_texts = texts[start:start+CHUNK_SIZE]
                    results_chunk = classify_requirements_texts(chunk_texts)
                    results_all.extend(results_chunk)
                    
                    # Generate subcats and suggestions
                    for idx, (txt, res) in enumerate(zip(chunk_texts, results_chunk)):
                        pred_label = res[0]
                        subcats.append(detect_subcategory(txt) if pred_label == "Non-Functional" else "N/A")
                        suggestions_all.append("; ".join(intelligent_suggestion(txt, pred_label)))
                        
                    progress.progress(min(100, math.floor((start+CHUNK_SIZE)/total*100)))
                
                # Unpack results
                df["predicted_label"] = [r[0] for r in results_all]
                df["confidence_pct"] = [f"{r[1]*100:.2f}%" for r in results_all] # Added Confidence
                df["subcategory"] = subcats
                df["suggestions"] = suggestions_all

                # conflict detection across the batch (after all other processing)
                conflicts = detect_conflicts_in_list(texts)

                st.success("✅ Batch analysis finished.")
                st.markdown("### Processed Results (First 20 Rows)")
                st.dataframe(df.head(20))

                # --- Download Button ---
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Full Results (CSV)", data=csv_bytes, file_name=download_name, mime="text/csv")
                st.markdown("---")

                # --- Statistics and Conflicts ---
                
                st.markdown("### 📊 Quick Statistics")
                # Prepare data for Plotly and Table
                stats = df["predicted_label"].value_counts()
                total = len(df)
                
                stats_data = pd.DataFrame({
                    "Category": ["Functional", "Non-Functional"],
                    "Count": [int(stats.get("Functional", 0)), int(stats.get("Non-Functional", 0))],
                })
                # Add Percentage for table
                stats_data["Percentage"] = stats_data["Count"].apply(lambda x: f"{x/total*100:.1f}%" if total > 0 else "0.0%")
                
                # Create two columns for better layout (Table and Chart side-by-side)
                col_stats, col_chart = st.columns([1, 1])

                with col_stats:
                    st.subheader("Count Distribution")
                    st.table(stats_data.set_index('Category'))

                with col_chart:
                    # Plotly Pie Chart (التعديل المطلوب)
                    if total > 0:
                        fig = px.pie(
                            stats_data, 
                            values='Count', 
                            names='Category', 
                            title='Functional vs. Non-Functional Requirements',
                            color='Category',
                            color_discrete_map={'Functional':'#1E3A8A', 'Non-Functional':'#93C5FD'}
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No data to display chart.")


                st.markdown("### 🔴 Conflict Warnings")
                if conflicts:
                    for c in conflicts:
                        st.warning(c)
                else:
                    st.info("🎉 No major conflicts detected across the uploaded requirements.")

                st.markdown("### Example Suggestions (First 10 Rows)")
                st.dataframe(df[["requirement", "predicted_label", "confidence_pct", "subcategory", "suggestions"]].head(10))

st.markdown("---")
st.caption("Built with RoBERTa fine-tuned on PROMISE.")