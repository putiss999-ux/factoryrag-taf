# ============================================
# app.py  (FULL + Free AI on HF, Spaces-ready)
# FactoryRAG-TAF — Chat & Expert Evaluation
# with Result Explanation (Rule-based TH/EN) + (Optional) Free AI via HuggingFace
# ============================================

# ---------- Hugging Face Spaces: fix write/caching perms ----------
import os
from pathlib import Path

RUNTIME_DIR = "/tmp/.streamlit"
Path(RUNTIME_DIR).mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = "/tmp"
os.environ["XDG_CACHE_HOME"] = RUNTIME_DIR
os.environ["XDG_CONFIG_HOME"] = RUNTIME_DIR
os.environ["XDG_STATE_HOME"] = RUNTIME_DIR
os.environ["STREAMLIT_CACHE_DIR"] = os.path.join(RUNTIME_DIR, "cache")
os.environ["HF_HOME"] = "/tmp"          # HF cache
os.environ["TRANSFORMERS_CACHE"] = "/tmp"

# =====================================
# Fix Streamlit PermissionError on Hugging Face
# =====================================
import os
from pathlib import Path

TMP_DIR = "/tmp/streamlit_fix"
Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
os.environ["STREAMLIT_HOME"] = TMP_DIR
os.environ["STREAMLIT_CACHE_DIR"] = os.path.join(TMP_DIR, "cache")
os.environ["XDG_CONFIG_HOME"] = TMP_DIR
os.environ["XDG_CACHE_HOME"] = TMP_DIR
os.environ["XDG_STATE_HOME"] = TMP_DIR
os.environ["HOME"] = TMP_DIR

cfg_path = Path(TMP_DIR) / "config.toml"
if not cfg_path.exists():
    with open(cfg_path, "w") as f:
        f.write("[server]\nheadless = true\nport = 8501\n")

# =====================================
# Now safe to import streamlit
# =====================================
import streamlit as st
import pandas as pd
import numpy as np
import json, re, math, hashlib, time, textwrap
from collections import Counter
from datetime import datetime

os.environ.setdefault("HOME", "/app")
os.environ.setdefault("STREAMLIT_BROWSER_GATHERUSAGESTATS", "false")
os.environ.setdefault("STREAMLIT_CONFIG_DIR", "/app/.streamlit")
Path("./.streamlit").mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="FactoryRAG-TAF Chat + Expert Eval", layout="wide")
# ─────────────────────────────────────────────────────────

# -----------------------------
# CONFIG paths
# -----------------------------
PATH_CMMS   = Path("cmms_logs.csv")
PATH_SENSOR = Path("sensor_features.csv")
PATH_MAN    = Path("manual_index.json")
PATH_Q      = Path("queries.csv")  # optional
PATH_WEIGHTS = Path("task_aware_best_weights.json")
LOG_PATH = Path("expert_chat_logs.csv")

# -----------------------------
# LOAD DATA (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_all():
    cmms = pd.read_csv(PATH_CMMS) if PATH_CMMS.exists() else pd.DataFrame()
    sensor = pd.read_csv(PATH_SENSOR) if PATH_SENSOR.exists() else pd.DataFrame()
    manuals = []
    if PATH_MAN.exists():
        try:
            with open(PATH_MAN, "r", encoding="utf-8") as f:
                manuals = json.load(f)
        except Exception:
            manuals = []
    queries = pd.read_csv(PATH_Q) if PATH_Q.exists() else pd.DataFrame()
    weights = {}
    if PATH_WEIGHTS.exists():
        try:
            with open(PATH_WEIGHTS, "r", encoding="utf-8") as f:
                weights = json.load(f)
        except Exception:
            weights = {}
    return cmms, sensor, manuals, queries, weights

cmms, sensor, manuals, queries, weights = load_all()

# -----------------------------
# TOKENIZER (same as eval script)
# -----------------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9\.\-\+×/]+")
def tokenize(text: str):
    if not isinstance(text, str): return []
    return [t.lower() for t in TOKEN_RE.findall(text)]

# -----------------------------
# Build corpora & BM25
# -----------------------------
man_docs = [(m.get("chunk_id", f"man_{i}"), tokenize(m.get("text",""))) for i,m in enumerate(manuals)]
cmms_docs = []
if not cmms.empty:
    for idx, r in cmms.iterrows():
        doc_id = r.get("log_id", None)
        if pd.isna(doc_id) or str(doc_id).strip() == "":
            doc_id = f"cmms_{int(idx)}"
        txt = " ".join([
            str(r.get("symptom_text","")),
            str(r.get("action_text","")),
            str(r.get("cause_text","")),
            str(r.get("fault_label",""))
        ])
        cmms_docs.append((doc_id, tokenize(txt)))

all_docs = man_docs + cmms_docs

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.doc_len = np.array([len(t) for _,t in docs], dtype=float) if self.N>0 else np.array([])
        self.avgdl = self.doc_len.mean() if self.N else 0.0
        df = Counter()
        for _,t in docs: df.update(set(t))
        self.idf = {w: math.log((self.N - c + 0.5)/(c + 0.5) + 1.0) for w,c in df.items()}
        self.tf  = [Counter(t) for _,t in docs]

    def topk(self, query_text, k=5):
        if self.N == 0: return []
        q = tokenize(query_text)
        s = np.zeros(self.N, dtype=float)
        for qt in q:
            if qt not in self.idf: continue
            idf = self.idf[qt]
            for i in range(self.N):
                f = self.tf[i].get(qt, 0.0)
                if f == 0: continue
                denom = f + self.k1*(1 - self.b + self.b*(self.doc_len[i]/(self.avgdl + 1e-9)))
                s[i] += idf * (f*(self.k1+1)) / (denom + 1e-9)
        idx = np.argsort(-s)[:k]
        return [(self.docs[i][0], float(s[i])) for i in idx if s[i] > 0]

bm25_manuals = BM25(man_docs)
bm25_all     = BM25(all_docs)

# -----------------------------
# SENSOR kNN (cosine)
# -----------------------------
FEATURE_COLS = ["rms","kurtosis","peak_freq_hz","band_energy_2k_3k","one_x_rpm","two_x_rpm","temp_max_c"]
sensor_feat = sensor.copy()
if not sensor_feat.empty:
    for c in FEATURE_COLS:
        if c in sensor_feat.columns: sensor_feat[c] = sensor_feat[c].astype(float)
        else: sensor_feat[c] = 0.0
    X = sensor_feat[FEATURE_COLS].values
    mu = X.mean(axis=0) if len(X)>0 else np.zeros(len(FEATURE_COLS))
    sd = X.std(axis=0) + 1e-9 if len(X)>0 else np.ones(len(FEATURE_COLS))
    Xz = (X - mu) / sd
else:
    X = np.zeros((0,len(FEATURE_COLS))); mu = np.zeros(len(FEATURE_COLS)); sd = np.ones(len(FEATURE_COLS)); Xz = X

def cos_sim(a, b):
    a = np.asarray(a); b = np.asarray(b)
    denom = (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12)
    return float(np.dot(a,b) / denom) if denom>0 else 0.0

fault_labels = set(sensor_feat["label"].unique().tolist()) if "label" in sensor_feat.columns else set()

def infer_label_from_query(q: str):
    ql = q.lower()
    for lbl in fault_labels:
        if lbl.replace("_"," ") in ql or lbl in ql: return lbl
    if "soft-foot" in ql or "soft foot" in ql: return "soft_foot" if "soft_foot" in fault_labels else None
    if "cavitation" in ql or "flow" in ql:     return "clogging" if "clogging" in fault_labels else None
    if "grease" in ql or "lubricat" in ql:     return "lubrication_issue" if "lubrication_issue" in fault_labels else None
    if "seal" in ql or "leak" in ql:           return "seal_wear" if "seal_wear" in fault_labels else None
    if "1x" in ql and "2x" in ql:              return "misalignment" if "misalignment" in fault_labels else None
    if "1x" in ql:                             return "unbalance" if "unbalance" in fault_labels else None
    if "bearing" in ql or "bpfi" in ql:        return "bearing_defect" if "bearing_defect" in fault_labels else None
    return None

NUM_RE_HZ  = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(k?hz)", re.IGNORECASE)
NUM_RE_DEG = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*°?c", re.IGNORECASE)

def query_to_feature_vector(q: str):
    ql = q.lower()
    v = mu.copy()
    if "1x" in ql:
        v[FEATURE_COLS.index("one_x_rpm")] = 1.0
    if "2x" in ql:
        v[FEATURE_COLS.index("two_x_rpm")] = 1.0
    m = NUM_RE_HZ.search(ql)
    if m:
        val, unit = m.groups()
        hz = float(val) * (1000.0 if unit.lower() == "khz" else 1.0)
        v[FEATURE_COLS.index("peak_freq_hz")] = hz
    m = NUM_RE_DEG.search(ql)
    if m:
        degc = float(m.group(1))
        v[FEATURE_COLS.index("temp_max_c")] = degc
    if ("bearing" in ql or "bpfi" in ql):
        idx = FEATURE_COLS.index("band_energy_2k_3k")
        v[idx] = max(mu[idx], 0.45)
    if ("cavitation" in ql or "broadband" in ql):
        idx = FEATURE_COLS.index("band_energy_2k_3k")
        v[idx] = mu[idx] + 0.2
    if ("misalignment" in ql or "2x" in ql):
        v[FEATURE_COLS.index("rms")] = mu[0] + 0.08
    if ("unbalance" in ql or ("1x" in ql and "2x" not in ql)):
        v[FEATURE_COLS.index("rms")] = mu[0] + 0.06
    if ("grease" in ql or "lubricat" in ql):
        v[FEATURE_COLS.index("temp_max_c")] = max(mu[-1], 80.0)
    return (v - mu) / sd

def sensor_retrieve_knn(query_text: str, k=10, label_bonus=0.15):
    if len(Xz)==0: return []
    qv = query_to_feature_vector(query_text)
    sims = np.apply_along_axis(lambda x: cos_sim(x, qv), 1, Xz)
    lbl = infer_label_from_query(query_text)
    if lbl is not None and "label" in sensor_feat.columns:
        sims = sims + (sensor_feat["label"].eq(lbl).astype(float) * label_bonus).values
    idx = np.argsort(-sims)[:k]
    out = []
    for i in idx:
        sid = sensor_feat.iloc[i].get("sensor_id", f"S{i}")
        out.append((sid, float(sims[i])))
    return out

# -----------------------------
# Symptom clues & rules
# -----------------------------
CLUE_RE_LIST = [
    r"\b1x\b", r"\b2x\b", r"\bpeak\b", r"\bhz\b", r"\bkhz\b",
    r"\brpm\b", r"\btemp\b", r"อุณหภูมิ", r"ความถี่", r"ค่า\s*rms", r"\bkurtosis\b"
]
CLUE_RE = re.compile("|".join(CLUE_RE_LIST), re.IGNORECASE)

def has_sensor_clues(q: str) -> bool:
    if not isinstance(q, str) or not q.strip():
        return False
    return bool(CLUE_RE.search(q))

SYMPTOM_RULES = [
    (re.compile(r"(ไม่.*ไหล|น้ำไม่.*ไหล|flow\s*(low|drop)|แรงดัน(ตก|ต่ำ))", re.I), "clogging"),
    (re.compile(r"(คาวิเทช|cavitat|ตะแกรง|วาล์ว.*(ปิด|อั้น))", re.I), "clogging"),
    (re.compile(r"(น้ำ.*(รั่ว|ซึม)|ซีล|packing|seal)", re.I), "seal_wear"),
    (re.compile(r"(ร้อน|อุณหภูมิสูง|จาระบี|หล่อลื่น)", re.I), "lubrication_issue"),
    (re.compile(r"(ไม่ทำงาน|สตาร์ทไม่ติด|trip|เบรกเกอร์|ฟิวส์)", re.I), "not_running"),
    (re.compile(r"(เสียงดัง|สั่นแรง|ไวเบรชันสูง)", re.I), "unbalance"),
]
def guess_label_from_symptom(q: str) -> str | None:
    for pat, lbl in SYMPTOM_RULES:
        if pat.search(q or ""):
            return lbl
    if re.search(r"(alignment|แนวเพลา|ตั้งศูนย์)", re.I):
        return "misalignment"
    return None

# -----------------------------
# Task categorization
# -----------------------------
def categorize(q):
    ql = q.lower()
    if "root cause" in ql or "one-pass fix" in ql or "abnormal condition" in ql:
        return "diagnosis"
    if "alignment offset" in ql or "angular limits" in ql or "provide page/section" in ql:
        return "parameter_lookup"
    if "soft-foot test" in ql or "list the soft-foot" in ql or "steps" in ql:
        return "procedure"
    return "diagnosis"

def znormalize(arr):
    arr = np.array(arr, dtype=float)
    return (arr - arr.mean())/(arr.std()+1e-9) if len(arr)>0 else arr

def fuse_results(query_text, w_text, w_sensor, k=5):
    # ปรับน้ำหนักแบบไดนามิก: ปิด sensor หากไม่มีคำใบ้สัญญาณ
    use_sensor = has_sensor_clues(query_text) and len(Xz) > 0
    wt_eff = float(w_text)
    ws_eff = float(w_sensor) if use_sensor else 0.0

    t = bm25_all.topk(query_text, k=10)
    s = sensor_retrieve_knn(query_text, k=10) if use_sensor else []

    t_ids, t_scores = zip(*t) if t else ([],[])
    s_ids, s_scores = zip(*s) if s else ([],[])
    t_z = znormalize(list(t_scores)) if t_scores else np.array([])
    s_z = znormalize(list(s_scores)) if s_scores else np.array([])

    fused = {}
    for i, docid in enumerate(t_ids):
        fused[docid] = fused.get(docid, 0.0) + wt_eff*float(t_z[i])
    for i, sid in enumerate(s_ids):
        fused[sid] = fused.get(sid, 0.0) + ws_eff*float(s_z[i])

    ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]
    return ranked, dict(t), dict(s), {"use_sensor": use_sensor, "wt_eff": wt_eff, "ws_eff": ws_eff}

# =========================
# RESULT EXPLANATION (Rule-based TH/EN)
# =========================
LABEL_EXPLANATIONS = {
    "misalignment": {
        "th": {
            "title": "แนวเพลาไม่ตรง (Misalignment)",
            "desc": "พบพฤติกรรมการสั่นแบบ 1× และ 2× RPM เด่น อาจเกิดจากการตั้งศูนย์เพลา/คัปปลิงไม่ตรง หรือฐานยึดคลาย",
            "checks": [
                "ตรวจตั้งศูนย์เพลา (laser alignment/ dial gauge)",
                "ตรวจคัปปลิงสึก/ยุบ/ขันแน่น",
                "ตรวจฐานยึด แผ่นชิม และ soft-foot"
            ]
        },
        "en": {
            "title": "Shaft Misalignment",
            "desc": "Dominant 1× and 2× components suggest angular/parallel misalignment or loose mounting.",
            "checks": [
                "Perform laser/dial alignment check",
                "Inspect coupling wear and fastener tightness",
                "Check base flatness and soft-foot"
            ]
        }
    },
    "unbalance": {
        "th": {
            "title": "ไม่สมดุล (Unbalance)",
            "desc": "พบ 1× RPM เด่น บ่งชี้จุดศูนย์มวลคลาดเคลื่อน",
            "checks": [
                "ทำการถ่วงสมดุล (balancing)",
                "ตรวจสิ่งอุดตัน/คราบเกาะบนใบพัด",
                "ตรวจเพลางอ/ใบพัดบิด"
            ]
        },
        "en": {
            "title": "Rotor Unbalance",
            "desc": "Dominant 1× component indicates mass eccentricity.",
            "checks": [
                "Dynamic balancing",
                "Clean deposits on impeller/rotor",
                "Check bent shaft or blade deformation"
            ]
        }
    },
    "bearing_defect": {
        "th": {
            "title": "ตลับลูกปืนมีรอยชำรุด (Bearing Defect)",
            "desc": "มี broadband/ความถี่ลักษณะ BPFI/BPFO/BSF/FTF และค่า temp อาจสูงขึ้น",
            "checks": [
                "ตรวจหล่อลื่นและอายุจาระบี",
                "ตรวจเพลย์/หลวม และเสียงผิดปกติ",
                "วางแผนเปลี่ยนตลับลูกปืน"
            ]
        },
        "en": {
            "title": "Rolling Bearing Fault",
            "desc": "Broadband energy and bearing characteristic frequencies (BPFI/BPFO/BSF/FTF) with possible temp rise.",
            "checks": [
                "Verify lubrication schedule/grease condition",
                "Check clearance/noise",
                "Plan bearing replacement"
            ]
        }
    },
    "lubrication_issue": {
        "th": {
            "title": "ปัญหาการหล่อลื่น",
            "desc": "อุณหภูมิสูงและเสียง/การสั่นเพิ่มขึ้น บ่งชี้จาระบีเสื่อม/ใส่มากไป/น้อยไป",
            "checks": [
                "ตรวจชนิด/ปริมาณจาระบีให้ถูกสเปค",
                "ปรับรอบการอัดจาระบี",
                "ตรวจซีลรั่วซึม"
            ]
        },
        "en": {
            "title": "Lubrication Issue",
            "desc": "Elevated temperature and vibration; grease degradation or improper fill.",
            "checks": [
                "Verify grease type/quantity per spec",
                "Adjust regreasing interval",
                "Inspect seal leakage"
            ]
        }
    },
    "seal_wear": {
        "th": {
            "title": "ซีลสึก/รั่ว (Seal Wear/Leak)",
            "desc": "พบอุณหภูมิสูง/การรั่วซึม ส่งผลให้ประสิทธิภาพปั๊มลดลง",
            "checks": [
                "ตรวจซีลเพลา/แพ็กกิ้ง",
                "ตรวจรอยรั่วและร่องรอยของเหลว",
                "พิจารณาเปลี่ยนซีล"
            ]
        },
        "en": {
            "title": "Seal Wear / Leakage",
            "desc": "Seal degradation leads to leakage and reduced efficiency.",
            "checks": [
                "Inspect shaft seal/packing",
                "Check traces of fluid leakage",
                "Plan seal replacement"
            ]
        }
    },
    "clogging": {
        "th": {
            "title": "อุดตัน/การไหลติดขัด (Clogging/Cavitation-like)",
            "desc": "พลังงานย่านสูง (2–3 kHz) เพิ่ม/การไหลตก อาจเกิดการอุดตันหรือคาวิเทชัน",
            "checks": [
                "ตรวจตะแกรง/ท่อดูด-จ่าย สิ่งแปลกปลอม",
                "ตรวจ NPSH/ระดับของเหลว",
                "ตรวจวาล์วอั้น/เปิดไม่สุด"
            ]
        },
        "en": {
            "title": "Clogging / Cavitation-like",
            "desc": "Elevated high-band energy and flow drop suggest blockage or cavitation.",
            "checks": [
                "Check strainers/suction-discharge lines",
                "Verify NPSH/level",
                "Inspect throttling valves"
            ]
        }
    },
    "soft_foot": {
        "th": {
            "title": "Soft-foot (ฐานเครื่องไม่เสมอ)",
            "desc": "ฐานวางเครื่องไม่เรียบ ทำให้ alignment เพี้ยน/สั่นผิดปกติ",
            "checks": [
                "ทำ soft-foot test และปรับแผ่นชิม",
                "ขันน็อตฐานให้ถูกแรงบิด",
                "ตรวจความเรียบของฐาน"
            ]
        },
        "en": {
            "title": "Soft-foot",
            "desc": "Base flatness issues distort alignment and elevate vibration.",
            "checks": [
                "Perform soft-foot test and shim correction",
                "Torque base bolts correctly",
                "Check base flatness"
            ]
        }
    }
}

def explain_labels_from_results(fused_rank, sensor_df, lang="th", top_n=2):
    if sensor_df is None or sensor_df.empty:
        return []
    counts = {}
    for rid, _ in fused_rank:
        if "sensor_id" in sensor_df.columns:
            m = sensor_df[sensor_df["sensor_id"].astype(str) == str(rid)]
            if not m.empty and "label" in m.columns:
                lbl = str(m.iloc[0]["label"])
                counts[lbl] = counts.get(lbl, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: -x[1])[:top_n]
    explanations = []
    for lbl, _ in ranked:
        info = LABEL_EXPLANATIONS.get(lbl)
        if not info: continue
        pack = info.get(lang, info.get("th"))
        explanations.append({"label": lbl, "title": pack["title"], "desc": pack["desc"], "checks": pack["checks"]})
    return explanations

# -----------------------------
# Free AI — polish rule-based into operator-friendly bullets
# -----------------------------
def _rule_based_operator_summary(query_text, fused_rank, sensor_df, lang="th"):
    exps = explain_labels_from_results(fused_rank, sensor_df, lang=lang, top_n=1)

    if not exps:
        # fallback ตามอาการในคำถาม
        lbl_sym = guess_label_from_symptom(query_text)
        if lbl_sym and lbl_sym in LABEL_EXPLANATIONS:
            pack = LABEL_EXPLANATIONS[lbl_sym].get(lang, LABEL_EXPLANATIONS[lbl_sym]["th"])
            exps = [{"label": lbl_sym, "title": pack["title"], "desc": pack["desc"], "checks": pack["checks"]}]
        elif lbl_sym == "not_running":
            if lang == "th":
                return ("• สาเหตุที่เป็นไปได้: ระบบไฟ/อินเตอร์ล็อกตัด เครื่องจ่ายไฟไม่ครบ หรือชุดขับติดขัด\n"
                        "• สิ่งที่ควรตรวจ: เบรกเกอร์/ฟิวส์/โอเวอร์โหลด • ปุ่มฉุกเฉิน/สวิตช์ความดัน/การ์ดอินเตอร์ล็อก • คัปปลิงหลุด/มอเตอร์ค้าง\n"
                        "• วิธีแก้เบื้องต้น: เคลียร์ trip รีเซ็ตอุปกรณ์ ตรวจการหมุนอิสระก่อนสตาร์ทใหม่\n"
                        "• ข้อควรระวัง: ทำ LOTO และทดสอบฉนวนก่อนทำงานไฟฟ้า")
            else:
                return ("• Probable causes: power/interlock trips or drive seizure\n"
                        "• Checks: breaker/fuse/overload • E-stop/pressure switch/guard interlock • coupling/motor free\n"
                        "• Quick fixes: clear trips, reset, ensure free rotation before restart\n"
                        "• Safety: LOTO and electrical safety")

    if exps:
        e = exps[0]
        if lang == "th":
            return (f"• สาเหตุที่เป็นไปได้: {e['title']}\n"
                    f"• สิ่งที่ควรตรวจ: " + " ; ".join(e["checks"][:3]) + "\n"
                    f"• วิธีแก้เบื้องต้น: ปรับตั้ง/ขันแน่น/หล่อลื่นตามสเปก\n"
                    f"• ข้อควรระวัง: ปิดเครื่องและทำ LOTO ก่อนตรวจวัดทุกครั้ง")
        else:
            return (f"• Probable causes: {e['title']}\n"
                    f"• Checks: " + " ; ".join(e["checks"][:3]) + "\n"
                    f"• Quick fixes: re-align/tighten/lubricate per spec\n"
                    f"• Safety: LOTO before any work")

    # ถ้ายังไม่มีอะไรเลย
    if lang == "th":
        return ("• สาเหตุที่เป็นไปได้: ตรวจเพลา/การไหล/แรงดัน/หล่อลื่น\n"
                "• สิ่งที่ควรตรวจ: 1×/2×, อุณหภูมิปลายเพลา, คัปปลิง, ซีล, แบริ่ง\n"
                "• วิธีแก้เบื้องต้น: ตั้งศูนย์/ขันแน่น/ปรับหล่อลื่นตามสเปก\n"
                "• ข้อควรระวัง: ปิดเครื่องและทำ LOTO ก่อนทำงาน")
    else:
        return ("• Probable causes: shaft/flow/pressure/lubrication issues\n"
                "• Checks: 1×/2×, shaft temp, coupling, seal, bearing\n"
                "• Quick fixes: re-align, tighten, lubricate per spec\n"
                "• Safety: LOTO before work")

@st.cache_resource(show_spinner=False)
def load_hf_pipeline(model_name: str, max_new_tokens: int = 80):
    from transformers import pipeline
    task = "text2text-generation" if any(k in model_name.lower() for k in ["flan","t5","ul2"]) else "text-generation"
    gen = pipeline(
        task=task,
        model=model_name,
        device_map="auto",
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.05,
        truncation=True,
        pad_token_id=0,
    )
    return gen

def ai_explain_free_hf(query_text, fused_rank, manuals, cmms, sensor_df, model_name, lang_code="th"):
    # 1) ทำร่าง 4 บรรทัดแบบเสถียร
    draft = _rule_based_operator_summary(query_text, fused_rank, sensor_df, lang=lang_code)

    # 2) ให้โมเดล “รีไรต์” โดยห้ามเพิ่มเนื้อหา
    if lang_code == "th":
        polish_prompt = (
            "โปรดปรับปรุงข้อความต่อไปนี้ให้เป็นภาษาช่างหน้างาน อ่านง่าย กระชับ และยังคง 4 หัวข้อเดิมเป๊ะ "
            "(ห้ามเพิ่ม/ลดหัวข้อ และห้ามเติมข้อมูลใหม่)\n\n"
            "รูปแบบที่ต้องการ:\n"
            "• สาเหตุที่เป็นไปได้: …\n"
            "• สิ่งที่ควรตรวจ: …\n"
            "• วิธีแก้เบื้องต้น: …\n"
            "• ข้อควรระวัง: …\n\n"
            f"{draft}"
        )
    else:
        polish_prompt = (
            "Rewrite the following into clear field-tech bullets keeping EXACTLY these 4 headings "
            "(do not add new facts):\n"
            "• Probable causes: …\n"
            "• Checks: …\n"
            "• Quick fixes: …\n"
            "• Safety: …\n\n"
            f"{draft}"
        )

    try:
        gen = load_hf_pipeline(model_name=model_name, max_new_tokens=80)
        out = gen(polish_prompt)
        text = out[0].get("generated_text","").strip()

        # clean: ให้ขึ้นต้นด้วย • และเอาแค่ 4 บรรทัด
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        cleaned = []
        for ln in lines:
            if ln.startswith("- "): ln = "• " + ln[2:].strip()
            if not ln.startswith("•"): ln = "• " + ln
            cleaned.append(ln)
            if len(cleaned) >= 4: break
        if len(cleaned) < 4:
            return draft
        return "\n".join(cleaned)
    except Exception:
        return draft

# -----------------------------
# UI
# -----------------------------
st.title("🤖 FactoryRAG-TAF — Chat & Expert Evaluation")

# Sidebar: Evaluator
st.sidebar.header("👩‍🏭 Evaluator")
evaluator_code = st.sidebar.text_input("Evaluator code (จะถูกแฮช, ไม่ต้องใช้ชื่อจริง)", "")
role = st.sidebar.selectbox("Role", ["Maintenance Engineer","Process Engineer","Safety Officer","Operator","Other"])
years = st.sidebar.number_input("Years of experience", 0, 50, 5)
domain = st.sidebar.text_input("Factory domain", "Sugar Mill")

# Sidebar: Retrieval Settings
st.sidebar.header("⚙️ Retrieval Settings")
task_mode = st.sidebar.selectbox("Task", ["auto","diagnosis","parameter_lookup","procedure"], index=0)

default_wt = 0.7; default_ws = 0.3  # ค่าเริ่มต้นเน้นข้อความ (เร็ว+เสถียร)
if weights: st.sidebar.info("Loaded task-aware weights from file.")
else:       st.sidebar.warning("No task_aware_best_weights.json found — using manual sliders below.")

if task_mode != "auto" and weights.get(task_mode):
    default_wt = float(weights[task_mode].get("w_text", default_wt))
    default_ws = float(weights[task_mode].get("w_sensor", default_ws))

w_text = st.sidebar.slider("Weight: Text", 0.0, 1.0, default_wt, 0.05)
w_sensor = st.sidebar.slider("Weight: Sensor", 0.0, 1.0, default_ws, 0.05)
top_k = st.sidebar.slider("Top-K", 3, 10, 5)

# Sidebar: Explanation
st.sidebar.header("🧠 Explanation")
lang_choice = st.sidebar.radio("Language / ภาษา", ["ไทย", "English"], index=0, horizontal=True)
lang_code = "th" if lang_choice == "ไทย" else "en"

# Sidebar: Free AI (HF)
st.sidebar.header("🤖 Free AI (Hugging Face)")
use_hf_ai = st.sidebar.toggle("Use Free AI (HF)", value=False, help="สรุปผลด้วยโมเดลโอเพนซอร์สฟรี (เน้นสั้น/อ่านง่าย)")
model_name = st.sidebar.selectbox(
    "Model (แนะนำ: เร็วมากบน CPU)",
    [
        "google/flan-t5-small",                 # เร็วมากบน CPU (default)
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct"      # ช้าบน CPU
    ],
    index=0
)

# Main input
query_default = (queries["query_text"].iloc[0] if ("query_text" in queries.columns and len(queries)>0) else "")
query = st.text_area("พิมพ์คำถามที่นี่:", value=query_default, height=120,
                     placeholder="เช่น 'ปั๊มป้อนน้ำอ้อยสั่นผิดปกติหลัง PM ควรตรวจอะไร'")
run = st.button("🔎 Run")

if run and query.strip():
    # determine task
    task = categorize(query) if task_mode == "auto" else task_mode

    # weights
    if weights.get(task):
        wt = float(weights[task].get("w_text", w_text))
        ws = float(weights[task].get("w_sensor", w_sensor))
    else:
        wt, ws = w_text, w_sensor

    t0 = time.time()
    fused_rank, text_dict, sensor_dict, wtinfo = fuse_results(query, wt, ws, k=top_k)
    latency_ms = int((time.time() - t0)*1000)

    st.subheader(f"📌 Task = **{task}**  |  Weights: text={wtinfo['wt_eff']:.2f}, sensor={wtinfo['ws_eff']:.2f}  |  Latency: {latency_ms} ms")
    if not wtinfo["use_sensor"]:
        st.caption("ℹ️ ปิดฝั่ง sensor อัตโนมัติ เพราะคำถามไม่มีคำใบ้เชิงสัญญาณ (1×/2×, Hz, °C ฯลฯ)")

    # Display fused ranking with provenance
    st.markdown("### 🔗 Top Results (Fused)")
    for rank, (rid, score) in enumerate(fused_rank, start=1):
        src = None; detail = None
        # manual
        if any(rid == m.get("chunk_id") for m in manuals):
            m = next(m for m in manuals if m.get("chunk_id")==rid)
            src = f"Manual: {m.get('doc')} (sec {m.get('section')}, pages {m.get('pages')})"
            detail = m.get("text","")
        # cmms
        if src is None and not cmms.empty:
            match = cmms[cmms["log_id"].astype(str) == str(rid)] if "log_id" in cmms.columns else pd.DataFrame()
            if not match.empty:
                row = match.iloc[0]
                src = f"CMMS log #{row.get('log_id')}"
                detail = " | ".join([str(row.get("symptom_text","")), str(row.get("action_text","")), str(row.get("cause_text",""))])
        # sensor
        if src is None and not sensor_feat.empty:
            match = sensor_feat[sensor_feat["sensor_id"].astype(str) == str(rid)] if "sensor_id" in sensor_feat.columns else pd.DataFrame()
            if not match.empty:
                row = match.iloc[0]
                src = f"Sensor #{row.get('sensor_id')} (label={row.get('label','-')})"
                vals = ", ".join([f"{c}={row.get(c)}" for c in ["rms","kurtosis","peak_freq_hz","band_energy_2k_3k","one_x_rpm","two_x_rpm","temp_max_c"] if c in row])
                detail = vals

        tsc = text_dict.get(rid, None)
        ssc = sensor_dict.get(rid, None)

        with st.expander(f"{rank}. {rid}  |  fused={score:.3f}  (text={tsc if tsc is not None else '-'}, sensor={ssc if ssc is not None else '-'})", expanded=False):
            if src: st.markdown(f"**Source:** {src}")
            if detail: st.write(detail)
            if src and src.startswith("Manual"):
                md = next(m for m in manuals if m.get("chunk_id")==rid)
                st.caption(f"Doc={md.get('doc')} | Section={md.get('section')} | Pages={md.get('pages')}")

    # --- Rule-based Explanation ---
    st.markdown("---")
    st.markdown("### 🧠 Result Explanation (Rule-based)")
    exps = explain_labels_from_results(fused_rank, sensor_feat, lang=lang_code, top_n=2)
    if not exps:
        # โชว์ตามอาการหากไม่มี sensor label
        lbl_sym = guess_label_from_symptom(query)
        if lbl_sym and lbl_sym in LABEL_EXPLANATIONS:
            pack = LABEL_EXPLANATIONS[lbl_sym].get(lang_code, LABEL_EXPLANATIONS[lbl_sym]["th"])
            exps = [{"label": lbl_sym, "title": pack["title"], "desc": pack["desc"], "checks": pack["checks"]}]
        elif lbl_sym == "not_running":
            if lang_code == "th":
                st.markdown("**เครื่องไม่ทำงาน / สตาร์ทไม่ติด**")
                st.write("เน้นตรวจระบบไฟและ interlock ก่อนจะเปิดเครื่อง")
                st.markdown("- **สิ่งที่ควรตรวจ:**")
                for c in ["ตรวจไฟหลัก/เบรกเกอร์/ฟิวส์/โอเวอร์โหลด", "ปุ่มฉุกเฉิน/สวิตช์ความดัน/การ์ดอินเตอร์ล็อก", "คัปปลิงหลุด/มอเตอร์ติดขัด"]:
                    st.markdown(f"  - {c}")
            else:
                st.markdown("**Pump not running / cannot start**")
                st.write("Check electrical supply and interlocks before mechanical work.")
                st.markdown("- **Checks:**")
                for c in ["Main breaker/fuse/overload", "E-stop/pressure switch/guard interlock", "Coupling shear/loose, motor seized"]:
                    st.markdown(f"  - {c}")

    if exps:
        for e in exps:
            st.markdown(f"**{e['title']}**")
            st.write(e["desc"])
            st.markdown("- **สิ่งที่ควรตรวจ:**" if lang_code == "th" else "- **Recommended checks:**")
            for c in e["checks"]: st.markdown(f"  - {c}")
            st.markdown("")
    elif not exps and lang_code != "th":
        st.caption("No clear sensor-based evidence to summarize.")

    # --- Free AI Explanation (HF) ---
    if use_hf_ai:
        st.markdown("### 🤖 Result Explanation (Free AI — Hugging Face)")
        with st.spinner(f"กำลังสรุปผลด้วยโมเดลฟรี: {model_name} ..."):
            ai_text = ai_explain_free_hf(query, fused_rank, manuals, cmms, sensor_feat, model_name=model_name, lang_code=lang_code)
        st.markdown(ai_text if ai_text else "_ไม่สามารถสร้างคำอธิบายด้วยโมเดลฟรีได้_")

    # --- Expert rating panel
    st.markdown("---")
    st.subheader("📝 Expert Rating (บันทึกข้อมูลเพื่อใช้ในบทความ)")
    col1, col2, col3 = st.columns(3)
    with col1:
        rated_accuracy = st.slider("Accuracy", 1, 5, 4)
        rated_completeness = st.slider("Completeness", 1, 5, 4)
        rated_groundedness = st.slider("Groundedness", 1, 5, 5)
    with col2:
        rated_helpful = st.slider("Helpfulness", 1, 5, 5)
        rated_actionable = st.slider("Actionability", 1, 5, 5)
        rated_safety = st.slider("Safety", 1, 5, 5)
    with col3:
        rated_overall = st.slider("Overall", 1, 5, 5)
        hallucination = st.checkbox("พบ Hallucination/ข้อมูลคลาดเคลื่อน", value=False)
        policy_violation = st.checkbox("พบประเด็นไม่ปลอดภัย/ผิดนโยบาย", value=False)

    comment = st.text_area("ความเห็นเพิ่มเติม", height=100, placeholder="ข้อดี/ข้อจำกัด/ข้อเสนอแนะ")
    decision = st.radio("ตัดสินใจ (accept/reject)", ["accept","reject"], horizontal=True)

    if st.button("💾 Save rating"):
        rater_hash = hashlib.sha1((evaluator_code or "anon").encode()).hexdigest()[:8]
        row = {
            "session_id": f"S-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "latency_ms": latency_ms,
            "rater_hash": rater_hash,
            "rater_role": role,
            "years_experience": years,
            "factory_domain": domain,
            "task": task,
            "query_text": query,
            "w_text": wtinfo['wt_eff'], "w_sensor": wtinfo['ws_eff'], "top_k": top_k,
            "rated_accuracy": rated_accuracy,
            "rated_completeness": rated_completeness,
            "rated_groundedness": rated_groundedness,
            "rated_helpfulness": rated_helpful,
            "rated_actionability": rated_actionable,
            "rated_safety": rated_safety,
            "rated_overall": rated_overall,
            "hallucination_flag": hallucination,
            "policy_violation_flag": policy_violation,
            "decision": decision,
            "comment": comment,
            "fused_ids": ";".join([rid for rid,_ in fused_rank]),
            "text_scores_json": json.dumps({k: float(v) for k,v in text_dict.items()}),
            "sensor_scores_json": json.dumps({k: float(v) for k,v in sensor_dict.items()})
        }
        try:
            if LOG_PATH.exists():
                df = pd.read_csv(LOG_PATH)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            else:
                df = pd.DataFrame([row])
            df.to_csv(LOG_PATH, index=False)
            st.success(f"บันทึกเรียบร้อย → {LOG_PATH.name}")
        except Exception as e:
            st.error(f"บันทึกล้มเหลว: {e}")

else:
    st.info("พิมพ์คำถามแล้วกด Run เพื่อดูผลลัพธ์จาก manuals + CMMS + sensor และ (ถ้าเปิด) สรุปผลด้วยโมเดลฟรีบน Hugging Face")




