# app.py — Self-OA Scanner (Jotform API edition)
# Replaces Excel workflow with live API pulls, adds date/state/supervisor filters,
# scans targeted image fields + fridge matrix, and shows field label with thumbnail.

import io, re, math, threading, base64, json
from typing import Dict, List, Optional, Tuple, Set
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.set_page_config(
    page_title="Team Hishmeh Self-OA Scanner",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🧪 Team Hishmeh Self-OA Scanner")

st.write(
    "Automatically scan self-OA submissions from Jotform for signs of negligence with this tool. "
    "Reports missing entries and low-effort images. Use the filters below to narrow results by state, supervisor, or date range."
)

# =========================
# CONFIG & CONSTANTS
# =========================

# --- Your Jotform Enterprise info ---
FORM_ID = "212986747087977"
API_BASE = f"https://dpzwest.jotform.com/API/form/{FORM_ID}/submissions"

# Add your API key in .streamlit/secrets.toml as:
# JOTFORM_API_KEY = "xxxxxxxxxxxxxxxx"
API_KEY = st.secrets.get("JOTFORM_API_KEY", "")

REQUEST_TIMEOUT_SEC = 20
MAX_WORKERS = 8

# Image analysis thresholds (tune in sidebar)
DEFAULT_EDGE_THRESH = 0.0005
DEFAULT_ENTROPY_THRESH = 8.0
DEFAULT_BLUR_THRESH = 4.0

# Supervisor mapping (state → supervisors)
STATE_TO_SUPS = {
    "CA": ["Alex", "Colby", "Erika", "Luis", "Neyda", "Dylan", "Rodrigo"],
    "AZ": ["Amy", "Jerrod", "Mike R", "Nolan", "Raven", "Tailor"]
}

# Full selectable supervisor list (union of both, keep your provided names)
ALL_SUPERVISORS = sorted(
    set(STATE_TO_SUPS["CA"]) | set(STATE_TO_SUPS["AZ"])
)

# Targeted image fields to scan (match by the "text" field in Jotform answers)
IMAGE_FIELDS_TO_SCAN = [
    "Submit photo of communication board showing current refrigeration temperatures",
    "Photo of documentation",
    "Photo of sanitizer test strip, showing the level of sanitizer in the 3 compartment sink sanitizer compartment",
    "Pizza #1 Photo", "Pizza #2 Photo", "Pizza #3 Photo", "Pizza #4 Photo",
    "Pizza #5 Top of Pizza", "Pizza #5 Bottom of Pizza", "Pizza #5 Cut Test",
    "Side Item #1 Photo", "Side Item #2 Photo", "Side Item #3 Photo",
    "Photo of scales",
    "Photo of production area that includes the makeline and the dough table",
    "Photo of the chalk wall",
    "Photo of walkin gaskets",
    "Photo of inside of makeline including gaskets",
    "Photo of sidewalk and parking lot",
    "Photo of red focal wall showing logo and lettering",
    "Photo of safe",
]

# Fridge temperature matrix identifiers & row labels to check
FRIDGE_MATRIX_FIELD_ID = "227"  # "All refrigerated products..." matrix
FRIDGE_TEMP_ROWS = [
    "Walkin (33°F-38°F)",
    "Make Line Bins (33°F-41°F)",
    "Make Line Cabinet (33°F-41°F)",
    "Coke Cooler (33°F-41°F)",
]
# Rule: FLAG if >= 2 temp entries are blank
FRIDGE_MISSING_FLAG_THRESHOLD = 2

# =========================
# HTTP SESSION (with retries)
# =========================

def make_session():
    s = requests.Session()
    retries = Retry(
        total=5, connect=3, read=3, backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"]
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

SESSION = make_session()
CACHE_LOCK = threading.Lock()
IMG_CACHE: Dict[str, Tuple[str, List[str]]] = {}
THUMB_CACHE: Dict[str, bytes] = {}

# =========================
# IMAGE ANALYSIS
# =========================

def _pil_to_gray(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"))

def _entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=256, range=(0, 255))
    p = hist.astype(float)
    total = p.sum()
    if not total:
        return 0.0
    p /= total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def _edge_density(gray: np.ndarray) -> float:
    import cv2
    edges = cv2.Canny(gray, 100, 200)
    return float(edges.sum()) / float(gray.size * 255.0)

def _blur_var(gray: np.ndarray) -> float:
    import cv2
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def analyze_img_bytes(b: bytes, EDGE_THRESH: float, ENTROPY_THRESH: float, BLUR_THRESH: float) -> Tuple[str, List[str], bytes]:
    try:
        img = Image.open(io.BytesIO(b))
        img.load()
    except Exception:
        return "BLANK_SUSPECT", ["unreadable_image"], b""

    max_side = 1600
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)))

    prev = img.copy()
    prev.thumbnail((140, 140))
    buf = io.BytesIO()
    prev.save(buf, format="PNG")
    thumb_bytes = buf.getvalue()

    g = _pil_to_gray(img)
    ent = _entropy(g)
    ed = _edge_density(g)
    bv = _blur_var(g)
    std = float(g.std())

    edge_hit = (ed <= EDGE_THRESH)
    ent_hit = (ent <= ENTROPY_THRESH)
    std_hit = (std < 2.0)
    blur_hit = (bv <= BLUR_THRESH and ent <= ENTROPY_THRESH)

    blank_like = (edge_hit and ent_hit) or std_hit
    blurry_like = blur_hit

    if blank_like:
        return "BLANK_SUSPECT", ["Blank suspected"], thumb_bytes
    if blurry_like:
        return "BLURRY_SUSPECT", ["Blurry suspected"], thumb_bytes
    return "OK", ["OK"], thumb_bytes

def fetch_img(url: str) -> Optional[bytes]:
    """
    Enterprise tenants often require the API key in a header to access upload URLs.
    """
    try:
        r = SESSION.get(url, timeout=REQUEST_TIMEOUT_SEC, headers={"APIKEY": API_KEY})
        if r.status_code != 200:
            return None
        ct = (r.headers.get("Content-Type") or "").lower()
        if ("image" not in ct) and not re.search(r"\.(png|jpe?g|webp|bmp|tiff?)", url, re.I):
            return None
        return r.content
    except Exception:
        return None

def analyze_url(url: str, EDGE_THRESH: float, ENTROPY_THRESH: float, BLUR_THRESH: float) -> Tuple[str, List[str]]:
    with CACHE_LOCK:
        if url in IMG_CACHE:
            return IMG_CACHE[url]
    b = fetch_img(url)
    if not b:
        result = ("SKIPPED", ["download_failed_or_non_image"])
    else:
        status, reasons, thumb = analyze_img_bytes(b, EDGE_THRESH, ENTROPY_THRESH, BLUR_THRESH)
        if status in ("BLANK_SUSPECT", "BLURRY_SUSPECT"):
            with CACHE_LOCK:
                THUMB_CACHE[url] = thumb
        result = (status, reasons)
    with CACHE_LOCK:
        IMG_CACHE[url] = result
    return result

def thumbs_html(items: List[Tuple[str, str]]) -> str:
    """
    items: list of (url, caption) — caption here is our FIELD NAME for display.
    """
    parts = []
    for u, caption in items:
        b = THUMB_CACHE.get(u)
        safe_caption = (caption or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if not b:
            parts.append(
                f'<div style="margin:4px">'
                f'<a href="{u}" target="_blank" rel="noopener">link</a>'
                f'<div style="font-size:11px;color:#555;margin-top:2px">{safe_caption}</div>'
                f'</div>'
            )
            continue
        b64 = base64.b64encode(b).decode("ascii")
        parts.append(
            f'<div style="margin:4px;max-width:160px">'
            f'<a href="{u}" target="_blank" rel="noopener">'
            f'<img src="data:image/png;base64,{b64}" '
            f'style="width:120px;height:auto;object-fit:cover;border-radius:8px;border:1px solid #ddd;" />'
            f'</a>'
            f'<div style="font-size:11px;color:#555;margin-top:4px">{safe_caption}</div></div>'
        )
    return "<div style='display:flex;flex-wrap:wrap;gap:6px'>" + "".join(parts) + "</div>"

# =========================
# JOTFORM HELPERS
# =========================

HTML_TAG_RE = re.compile(r"<[^>]+>")

def strip_html(s: str) -> str:
    return HTML_TAG_RE.sub("", s or "").replace("\xa0", " ").strip()

def build_filter_params(start_date: date, end_date: date, selected_states: List[str], selected_supers: List[str]) -> Dict[str, str]:
    """
    Build Jotform API filter JSON.
    - If no supervisor restrictions (All states & All supervisors), we pass only date filters.
    - Otherwise, we pass OR conditions on the Supervisor Region field (ID 8).
    """
    # Expand states to supervisors (union)
    state_supers: Set[str] = set()
    for st_code in selected_states:
        state_supers.update(STATE_TO_SUPS.get(st_code, []))

    # If they picked supervisors explicitly, union with state expansion
    selected = set(selected_supers) | state_supers

    # Base date range (inclusive-ish)
    f = {
        "created_at:gt": str(start_date),
        "created_at:lt": str(end_date)
    }

    # Only include supervisor filter if a subset is chosen
    if selected and len(selected) < len(ALL_SUPERVISORS):
        # Supervisor Region is field ID 8; filter syntax accepts {"submission:8":"Alex"} and $or list.
        f["$or"] = [{"submission:8": s} for s in sorted(selected)]

    return {"apiKey": API_KEY, "limit": 1000, "filter": json.dumps(f)}

def fetch_all_submissions(params: Dict[str, str]) -> List[Dict]:
    """
    Paginates through submissions using limit/offset.
    """
    out: List[Dict] = []
    offset = 0
    while True:
        p = dict(params)
        p["offset"] = offset
        r = SESSION.get(API_BASE, params=p, timeout=REQUEST_TIMEOUT_SEC)
        r.raise_for_status()
        payload = r.json()
        batch = payload.get("content", []) or []
        out.extend(batch)
        if len(batch) < int(p.get("limit", 1000)):
            break
        offset += len(batch)
    return out

def extract_image_answers(ans: Dict[str, Dict]) -> List[Tuple[str, str]]:
    """
    Return list of (field_label, url) for ALL widget-upload fields present in a submission.
    """
    out: List[Tuple[str, str]] = []
    for field in ans.values():
        if field.get("type") == "control_widget":
            label = strip_html(field.get("text", ""))
            url = field.get("answer")
            if isinstance(url, str) and url.startswith("http"):
                out.append((label, url))
    return out

def extract_target_images(ans: Dict[str, Dict]) -> Dict[str, Optional[str]]:
    """
    From IMAGE_FIELDS_TO_SCAN, pick those that exist in this submission and return {label: url_or_None}.
    If the field exists but is blank, url will be None.
    """
    # Build label -> url map of all widget answers
    present = {strip_html(f.get("text", "")): f.get("answer") for f in ans.values() if f.get("type") == "control_widget"}
    result: Dict[str, Optional[str]] = {}
    for label in IMAGE_FIELDS_TO_SCAN:
        url = present.get(label)
        if isinstance(url, str) and url.strip():
            result[label] = url
        else:
            result[label] = None
    return result

def parse_fridge_matrix(ans: Dict[str, Dict]) -> Dict[str, Optional[str]]:
    """
    Reads matrix field 227 and returns {clean_row_label: temperature_string_or_None}.
    Each matrix cell value looks like a JSON string: ["YES","", "40"] -> we want element [2].
    Matrix keys may contain HTML spans, so we strip tags before matching.
    """
    out: Dict[str, Optional[str]] = {k: None for k in FRIDGE_TEMP_ROWS}
    field = ans.get(FRIDGE_MATRIX_FIELD_ID)
    if not field:
        return out
    matrix = field.get("answer") or {}
    for raw_label, raw_value in matrix.items():
        clean = strip_html(raw_label)
        # Find which target row this corresponds to (prefix/contains-safe)
        match_key = None
        for target in FRIDGE_TEMP_ROWS:
            if clean.startswith(target) or target in clean:
                match_key = target
                break
        if not match_key:
            continue
        temp_str = None
        try:
            parsed = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
            if isinstance(parsed, list) and len(parsed) >= 3:
                val = parsed[2]
                if isinstance(val, str) and val.strip():
                    temp_str = val.strip()
        except Exception:
            temp_str = None
        out[match_key] = temp_str
    return out

# =========================
# UI
# =========================




# Sidebar thresholds
with st.sidebar.expander("Scanner Settings (Advanced)", expanded=False):
    EDGE_THRESH = st.number_input("Max edge density (blank-ish)", value=DEFAULT_EDGE_THRESH, format="%.5f")
    ENTROPY_THRESH = st.number_input("Max entropy (blank-ish)", value=DEFAULT_ENTROPY_THRESH, format="%.2f")
    BLUR_THRESH = st.number_input("Max Laplacian variance (blurry)", value=DEFAULT_BLUR_THRESH, format="%.1f")

# Filters
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    states = st.multiselect("States", ["CA", "AZ"], default=["CA","AZ"])

with col2:
    supervisors = st.multiselect("Supervisors (optional)", ALL_SUPERVISORS, default=[])

today = date.today()
with col3:
    start = st.date_input("Start date", today - timedelta(days=7))
with col4:
    end = st.date_input("End date", today)


st.caption("If you leave Supervisors empty and keep both States selected, the scanner will query **all** supervisors (date filter only).")



if not API_KEY:
    st.error("Missing JOTFORM_API_KEY in Streamlit secrets.")
    st.stop()

if st.button("Start Scan", type="primary"):
    # --- Build filters & fetch submissions
    params = build_filter_params(start, end, states, supervisors)
    try:
        submissions = fetch_all_submissions(params)
    except Exception as e:
        st.error(f"Failed to fetch submissions: {e}")
        st.stop()

    # --- Collect targeted image URLs first (unique), for progress bar
    unique_urls: Set[str] = set()
    parsed_payloads = []  # keep per-submission parsed info for later display

    for sub in submissions:
        ans: Dict[str, Dict] = sub.get("answers", {})
        # Targeted images for this submission
        target_images = extract_target_images(ans)
        # Fridge matrix parsing
        fridge = parse_fridge_matrix(ans)

        # Collect URLs for analysis
        for label, url in target_images.items():
            if isinstance(url, str) and url.startswith("http"):
                unique_urls.add(url)

        parsed_payloads.append({
            "submission_id": sub.get("id"),
            "created_at": sub.get("created_at"),
            "store_number": ans.get("84", {}).get("answer"),
            "supervisor_region": ans.get("8", {}).get("answer"),
            "auditor_name": ans.get("3", {}).get("answer"),
            "target_images": target_images,   # {label: url or None}
            "fridge": fridge                 # {label: temp or None}
        })

    # --- Analyze images with progress
    total_urls = len(unique_urls)
    prog = st.progress(0.0)
    status = st.empty()
    if total_urls > 0:
        urls_list = list(unique_urls)
        CHUNK = max(10, total_urls // 50)
        done = 0
        for i in range(0, total_urls, CHUNK):
            for u in urls_list[i:i+CHUNK]:
                analyze_url(u, EDGE_THRESH, ENTROPY_THRESH, BLUR_THRESH)
            done += len(urls_list[i:i+CHUNK])
            prog.progress(min(done / total_urls, 1.0))
            status.write(f"Analyzed {done}/{total_urls} images…")
    prog.progress(1.0)
    status.write("Image analysis complete.")

    # --- Build flagged results
    out_rows: List[Dict] = []

    for item in parsed_payloads:
        reasons, flagged_thumbs = [], []

        # 1) Fridge temp rule: flag if >= 2 blanks
        missing_temps = [k for k, v in item["fridge"].items() if not (isinstance(v, str) and v.strip())]
        if len(missing_temps) >= FRIDGE_MISSING_FLAG_THRESHOLD:
            reasons.append(f"Fridge temps incomplete ({len(missing_temps)} missing): " + "; ".join(missing_temps))

        # 2) Targeted images: missing URL OR quality flagged by analyzer
        flagged_images = 0
        for label, url in item["target_images"].items():
            if not url:
                flagged_images += 1
                # Placeholder entry for missing image (no thumb)
                flagged_thumbs.append(("about:blank", f"{label} — missing"))
                continue
            status_tag, _reasons = IMG_CACHE.get(url, ("OK", []))
            if status_tag in ("BLANK_SUSPECT", "BLURRY_SUSPECT", "SKIPPED"):
                flagged_images += 1
                # Show field name as caption
                flagged_thumbs.append((url, label))

        if flagged_images > 0:
            reasons.append(f"{flagged_images} image issues")

        if reasons:
            out_rows.append({
                "Created At": item["created_at"],
                "Supervisor Region": item["supervisor_region"],
                "Store Number": item["store_number"],
                "Auditor": item["auditor_name"],
                "Reasons": "; ".join(reasons),
                "Thumbnails": thumbs_html([(u, c) for (u, c) in flagged_thumbs if u != "about:blank"])
            })

    # --- Output
    if out_rows:
        df = pd.DataFrame(out_rows)

        # Downloadable CSV (without HTML thumbs)
        csv_df = df.drop(columns=["Thumbnails"], errors="ignore")
        st.download_button(
            "Download flagged submissions (CSV)",
            data=csv_df.to_csv(index=False).encode("utf-8"),
            file_name="flagged_submissions.csv",
            mime="text/csv",
            key="download_flagged_submissions_button"
        )

        st.markdown("""
        <style>
        table.dataframe {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;  /* Enforce fixed widths */
        }

        table.dataframe th,
        table.dataframe td {
            padding: 6px 8px;
            border: 1px solid #ddd;
            vertical-align: top;
            word-wrap: break-word;
        }

        table.dataframe th {
            background: #f6f6f6;
            position: sticky;
            top: 0;
            z-index: 1;
            text-align: center; /* Centers column headers */
        }

        /* Optional: Center-align content for specific columns */
        table.dataframe td:nth-child(1),
        table.dataframe td:nth-child(2),
        table.dataframe td:nth-child(3),
        table.dataframe td:nth-child(4) {
            text-align: center;
        }

        /* Narrower "Reasons" column */
        table.dataframe th:nth-child(5),
        table.dataframe td:nth-child(5) {
            width: 25%;
            max-width: 250px;
        }

        /* Give Thumbnails column a little more breathing room */
        table.dataframe th:nth-child(6),
        table.dataframe td:nth-child(6) {
            width: 35%;
        }
        </style>
        """, unsafe_allow_html=True)


        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

    else:
        st.info("No submissions flagged. ✨")
