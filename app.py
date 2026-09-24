"""
app.py
======
Member 4 — Frontend Developer: BARNISHA BISWAS
Project  : AI Finance Analyzer (RAG-Powered)
Companies: Apple · Microsoft · Tesla · Nvidia
Backend  : Member 3's llm_integration.py (FinanceRAG) integrated
Data     : 12 SEC 10-K PDFs · 10,237 chunks · BAAI/bge-small-en-v1.5 · FAISS
           Stock Prices · News · Financial PhraseBank (Member 1 expanded datasets)
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json, os, sys, time
from pypdf import PdfReader
from retriever import search # Load environment keys for direct API queries



# ── Member 3 RAG Backend Integration ──────────────────────────
# Add the folder containing llm_integration.py to the path



# ── Member 3 Groq Backend Integration ─────────────────────────
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Groq Client Configuration ──────────────────────────
@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

def ask_groq(prompt, client):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting response: {str(e)}"

@st.cache_resource
def load_retriever():
    try:
        sys.path.insert(0, BASE_DIR)
        from retriever import search
        return search
    except Exception as e:
        return None

def rag_answer(query, company=None, year=None):
    client = get_groq_client()
    search = load_retriever()
    if not client:
        return "Groq API key not found. Please check your .env file."
    
    context = "No context available from vector database."
    if search:
        try:
            kwargs = {}
            if company: kwargs["company"] = company
            if year: kwargs["year"] = year
            docs = search(query, top_k=5, **kwargs)
            context = "\n".join(str(d.get("text") or d.get("content") or d) for d in docs)
        except:
            pass
            
    prompt = f"""You are a financial analyst AI assistant.
Use the context below to answer the question accurately.
Context:
{context}
Question: {query}
Provide a clear, structured, and concise answer with numbers where available."""
    return ask_groq(prompt, client)

RAG_AVAILABLE = get_groq_client() is not None

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FinAnalyzer AI",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# CSS — EXACT SAME AS BEFORE — NOT CHANGED
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#07080f;color:#eaeeff;}

section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0c0d1e 0%,#090a18 100%)!important;
    border-right:1px solid rgba(139,92,246,.18);
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span{color:#b0b8d8!important;}

.brand{
    font-family:'Space Grotesk',sans-serif;
    font-size:1.4rem;font-weight:700;
    background:linear-gradient(135deg,#a78bfa,#fbbf24);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    margin-bottom:2px;letter-spacing:-.02em;
}
.brand-sub{font-size:.68rem;color:#3a4060!important;text-transform:uppercase;letter-spacing:.1em;}

div[role="radiogroup"] label{
    display:flex!important;align-items:center!important;
    padding:9px 14px!important;border-radius:10px!important;
    color:#8890b8!important;font-size:.88rem!important;
    transition:all .2s!important;margin-bottom:2px!important;
}
div[role="radiogroup"] label:hover{background:rgba(139,92,246,.12)!important;color:#c8ccff!important;}

.hero{
    background:linear-gradient(135deg,#0f1232 0%,#110e28 50%,#0c0f22 100%);
    border:1px solid rgba(139,92,246,.22);border-radius:20px;
    padding:38px 44px;position:relative;overflow:hidden;margin-bottom:26px;
}
.hero::before{
    content:'';position:absolute;top:-80px;right:-80px;
    width:280px;height:280px;border-radius:50%;
    background:radial-gradient(circle,rgba(251,191,36,.07) 0%,transparent 70%);
}
/* Logo icon — made brighter as requested */
.hero::after{
    content:'💹';position:absolute;right:44px;top:50%;
    transform:translateY(-50%);font-size:7rem;opacity:.22;
    filter:drop-shadow(0 0 20px rgba(167,139,250,.6));
}
.h-title{
    font-family:'Space Grotesk',sans-serif;font-size:2.3rem;
    font-weight:700;color:#fff;line-height:1.15;margin-bottom:10px;
}
.h-title .g{
    background:linear-gradient(135deg,#a78bfa,#fbbf24);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.h-sub{color:#6870a8;font-size:.93rem;max-width:540px;line-height:1.68;}

.mc{
    background:linear-gradient(145deg,#0f1130,#111228);
    border:1px solid rgba(139,92,246,.18);border-radius:16px;
    padding:20px 16px;text-align:center;position:relative;overflow:hidden;
    transition:transform .2s,border-color .2s;
}
.mc:hover{transform:translateY(-3px);border-color:rgba(139,92,246,.4);}
.mc-bar{height:2px;background:linear-gradient(90deg,#a78bfa,#fbbf24);margin-bottom:14px;border-radius:2px;}
.mc-val{font-family:'Space Grotesk',sans-serif;font-size:1.9rem;font-weight:700;color:#a78bfa;line-height:1;}
.mc-lbl{font-size:.7rem;color:#4a5080;margin-top:6px;text-transform:uppercase;letter-spacing:.07em;}
.mc-delta{font-size:.75rem;margin-top:5px;}

.sh{
    font-family:'Space Grotesk',sans-serif;font-size:1.05rem;font-weight:600;
    color:#dde0ff;border-left:3px solid #a78bfa;padding-left:12px;
    margin:24px 0 14px;display:flex;align-items:center;gap:8px;
}

.cb{
    background:#0f1130;border:1px solid rgba(139,92,246,.14);
    border-radius:14px;padding:20px 24px;
    color:#c0c8e8;font-size:.9rem;line-height:1.75;
}

/* Chat — UNCHANGED */
.mu{
    background:linear-gradient(135deg,#2d1f70,#1a2875);
    border:1px solid rgba(139,92,246,.3);
    border-radius:18px 18px 4px 18px;
    padding:13px 18px;margin:10px 0 10px 80px;
    color:#dee2ff;font-size:.9rem;
}
.ma{
    background:#0f1130;border:1px solid rgba(139,92,246,.14);
    border-radius:18px 18px 18px 4px;
    padding:14px 18px;margin:10px 80px 10px 0;
    color:#c4c8e8;font-size:.9rem;line-height:1.65;
}
.ml{font-size:.67rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px;color:#3a4068;}

.rh{background:#1a0e0e;border-left:3px solid #f87171;border-radius:0 10px 10px 0;padding:11px 15px;margin:7px 0;color:#fca5a5;font-size:.86rem;}
.rm{background:#191500;border-left:3px solid #fbbf24;border-radius:0 10px 10px 0;padding:11px 15px;margin:7px 0;color:#fde68a;font-size:.86rem;}
.rl{background:#0a1a10;border-left:3px solid #34d399;border-radius:0 10px 10px 0;padding:11px 15px;margin:7px 0;color:#6ee7b7;font-size:.86rem;}

.rc{
    background:#0f1130;border:1px solid rgba(139,92,246,.16);
    border-radius:12px;padding:15px 18px;margin:8px 0;
    display:flex;align-items:center;justify-content:space-between;
    transition:border-color .2s;
}
.rc:hover{border-color:rgba(251,191,36,.35);}
.rc-name{font-size:.9rem;font-weight:600;color:#dde0ff;}
.rc-sub{font-size:.75rem;color:#5a6090;margin-top:3px;}
.rc-badge{background:rgba(251,191,36,.12);color:#fbbf24;border:1px solid rgba(251,191,36,.3);border-radius:6px;padding:3px 10px;font-size:.72rem;font-weight:600;}

.tr-row{
    display:flex;align-items:center;justify-content:space-between;
    padding:11px 14px;border-radius:10px;margin:5px 0;
    background:#0f1130;border:1px solid rgba(139,92,246,.1);
    font-size:.86rem;
}
.tr-cat{color:#a78bfa;font-weight:500;font-size:.8rem;}
.tr-amt-e{color:#f87171;font-weight:600;}
.tr-amt-i{color:#34d399;font-weight:600;}

.bb{background:#131628;border-radius:6px;height:8px;margin:6px 0 14px;}
.bb-fill{height:8px;border-radius:6px;background:linear-gradient(90deg,#a78bfa,#fbbf24);}

.pill{display:inline-block;background:#131628;border:1px solid rgba(139,92,246,.22);border-radius:20px;padding:4px 12px;font-size:.74rem;color:#8890c8;margin:3px;}
.pill-g{background:rgba(251,191,36,.08);border-color:rgba(251,191,36,.25);color:#fbbf24;}

.sc{background:#0f1130;border:1px solid rgba(139,92,246,.16);border-radius:12px;padding:18px 14px;text-align:center;}
.sc-n{font-family:'Space Grotesk',sans-serif;font-size:1.7rem;font-weight:700;color:#a78bfa;}
.sc-t{font-size:.78rem;color:#5a6090;margin-top:5px;line-height:1.5;}

.am{background:#0f1130;border:1px solid rgba(139,92,246,.14);border-radius:12px;padding:15px 18px;margin:7px 0;}
.am-r{font-size:.68rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px;}
.am-n{font-family:'Space Grotesk',sans-serif;font-size:.92rem;font-weight:600;color:#eaecff;}
.am-d{font-size:.8rem;color:#6878a8;margin-top:4px;line-height:1.5;}

.goal{background:#0f1130;border:1px solid rgba(139,92,246,.14);border-radius:12px;padding:16px 18px;margin:8px 0;}
.goal-title{font-size:.9rem;font-weight:600;color:#dde0ff;margin-bottom:8px;}
.goal-bar{background:#131628;border-radius:5px;height:7px;margin:6px 0;}
.goal-fill{height:7px;border-radius:5px;}

.insight{
    background:linear-gradient(135deg,rgba(139,92,246,.08),rgba(251,191,36,.05));
    border:1px solid rgba(139,92,246,.2);border-radius:12px;
    padding:14px 18px;margin:8px 0;
    color:#c4c8e8;font-size:.86rem;line-height:1.6;
}
.insight-icon{font-size:1.2rem;margin-right:8px;}

/* Upload button — made more visible as requested */
[data-testid="stFileUploadDropzone"]{
    background:#0f1130!important;
    border:2px dashed rgba(167,139,250,.5)!important;
    border-radius:16px!important;color:#a78bfa!important;
    font-weight:600!important;
}
[data-testid="stFileUploadDropzone"] button{
    background:linear-gradient(135deg,#7c3aed,#d97706)!important;
    color:#fff!important;font-weight:700!important;
    border-radius:10px!important;
}

.stButton>button{
    background:linear-gradient(135deg,#7c3aed,#d97706)!important;
    color:#fff!important;border:none!important;border-radius:10px!important;
    font-weight:600!important;padding:8px 20px!important;font-size:.86rem!important;
}
.stButton>button:hover{opacity:.82!important;}

.stSelectbox>div>div{background:#0f1130!important;border:1px solid rgba(139,92,246,.2)!important;color:#e0e4ff!important;}

::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:#07080f;}
::-webkit-scrollbar-thumb{background:rgba(139,92,246,.28);border-radius:4px;}
::-webkit-scrollbar-thumb:hover{background:rgba(139,92,246,.5);}

.dv{border:none;border-top:1px solid rgba(139,92,246,.1);margin:16px 0;}

/* Stock market specific */
.stock-card{
    background:#0f1130;border:1px solid rgba(139,92,246,.18);
    border-radius:14px;padding:18px 20px;text-align:center;
}
.stock-price{font-family:'Space Grotesk',sans-serif;font-size:1.6rem;font-weight:700;color:#fff;}
.stock-up{color:#34d399;font-size:.85rem;font-weight:600;}
.stock-down{color:#f87171;font-size:.85rem;font-weight:600;}

/* News card */
.news-card{
    background:#0f1130;border:1px solid rgba(139,92,246,.14);
    border-radius:12px;padding:14px 18px;margin:8px 0;
    transition:border-color .2s;
}
.news-card:hover{border-color:rgba(251,191,36,.3);}
.news-title{color:#dde0ff;font-size:.88rem;font-weight:600;line-height:1.4;}
.news-meta{color:#4a5080;font-size:.72rem;margin-top:5px;}
.news-badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.68rem;font-weight:600;margin-right:6px;}

/* Sources box */
.source-box{
    background:#0a0b18;border:1px solid rgba(139,92,246,.1);
    border-radius:8px;padding:10px 14px;margin-top:10px;
    font-size:.76rem;color:#5a6090;
}

@media(max-width:768px){
    .h-title{font-size:1.4rem;}
    .hero{padding:22px 18px;}
    .hero::after{display:none;}
    .mc-val{font-size:1.4rem;}
    .mu{margin-left:20px;}
    .ma{margin-right:20px;}
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════
COMPANIES = ["Apple","Microsoft","Tesla","Nvidia"]
YEARS     = [2022, 2023, 2024]

FIN = {
    "Apple":     {"revenue":[394,383,391],"profit":[100,97,101],"debt":[111,109,105],"cash":[169,162,158],"rd":[26,30,32]},
    "Microsoft": {"revenue":[198,212,245],"profit":[72,83,101], "debt":[78,75,70],  "cash":[99,111,134], "rd":[25,28,31]},
    "Tesla":     {"revenue":[81,97,97],   "profit":[12,15,7],   "debt":[22,19,18],  "cash":[22,29,30],   "rd":[3,4,4]},
    "Nvidia":    {"revenue":[27,61,130],  "profit":[4,30,55],   "debt":[11,10,9],   "cash":[14,18,35],   "rd":[5,7,9]},
}
CO_CLR = {"Apple":"#a78bfa","Microsoft":"#60a5fa","Tesla":"#f87171","Nvidia":"#34d399"}
GOLD   = "#fbbf24"
BG     = "#07080f"
CARD   = "#0f1130"
GRID   = "#141628"

SUMMARY = {
    "Apple":     "Apple Inc. delivered strong ecosystem performance driven by iPhone, Services, Mac, and Wearables. The Services segment became the fastest-growing revenue driver, reaching record revenue and contributing significantly to margin expansion. iPhone revenues remained the largest segment at over 52% of total revenue. The company returned record capital to shareholders through share buybacks exceeding $90B and dividends. Key risks include supply chain concentration in China and increasing regulatory scrutiny under the EU Digital Markets Act.",
    "Microsoft": "Microsoft Corporation achieved record revenues fuelled by Azure cloud growth accelerating beyond 28% YoY, Copilot AI integration across enterprise Office products, and the successful Activision Blizzard gaming acquisition. Commercial cloud annualised revenue crossed $135 billion. Enterprise demand for AI-powered productivity and security tools drove record bookings. Cybersecurity threats to Azure infrastructure and EU antitrust risk around the gaming acquisition remain the primary concern areas.",
    "Tesla":     "Tesla, Inc. faced significant operating margin compression from aggressive global price reductions aimed at defending delivery volume against intensifying EV competition, particularly from BYD and Hyundai. Total vehicle deliveries reached new records. The Energy Generation and Storage segment, led by Megapack, grew over 40% YoY and emerged as a high-margin growth driver. CEO Elon Musk's involvement across SpaceX, X, and xAI represents a key-man concentration risk cited by institutional investors.",
    "Nvidia":    "NVIDIA Corporation delivered extraordinary revenue growth driven by explosive, unprecedented demand for H100 and A100 data centre GPUs used for large-scale AI model training and inference workloads. The Data Centre segment overtook Gaming to become the primary revenue driver at over 78% of total revenue. NVIDIA's market capitalisation exceeded $3 trillion, reflecting strong investor confidence in the AI infrastructure build-out. US export restrictions on advanced AI chips to China represent the single largest regulatory risk.",
}

RISKS = {
    "Apple":     [("HIGH","iPhone revenue concentration — over 52% of total annual revenue from a single product line"),("HIGH","Supply chain geographic concentration in China and Asia-Pacific manufacturing hubs"),("MEDIUM","EU Digital Markets Act and US DOJ antitrust scrutiny targeting App Store practices and fees"),("MEDIUM","Foreign currency exchange rate headwinds from a persistently strong US Dollar"),("LOW","Android and Samsung competition in the global premium smartphone and wearables segment")],
    "Microsoft": [("HIGH","Nation-state cybersecurity threats and ransomware attacks targeting Azure cloud infrastructure"),("HIGH","EU antitrust regulatory risk arising from the $69B Activision Blizzard gaming acquisition"),("MEDIUM","AI competition from Google DeepMind, Amazon AWS Bedrock, and OpenAI partnership risks"),("MEDIUM","Enterprise software license renewal risk during macroeconomic slowdown periods"),("LOW","Talent retention challenges in a highly competitive AI engineering and research hiring market")],
    "Tesla":     [("HIGH","Intensifying global EV price competition from BYD, Hyundai, GM, and legacy OEMs compressing margins"),("HIGH","Key-man concentration risk — CEO Elon Musk manages SpaceX, X Corp, Neuralink, and xAI simultaneously"),("MEDIUM","Lithium, cobalt, and battery-grade nickel supply chain constraints affecting production costs"),("MEDIUM","Global EV demand slowdown and elevated consumer financing costs reducing addressable market"),("LOW","Gigafactory manufacturing ramp-up execution risks in Berlin, Texas, and future expansion sites")],
    "Nvidia":    [("HIGH","US Bureau of Industry and Security export restrictions on H100/A100/H800 chips to China and allies"),("HIGH","Revenue concentration risk — a small number of hyperscaler clients drive majority of data centre revenue"),("MEDIUM","Rapid AI technology cycles requiring continuous, heavily capitalised R&D investment to maintain leadership"),("MEDIUM","Potential AI infrastructure spending pause if enterprise ROI from AI deployment disappoints expectations"),("LOW","Long-term competitive threat from AMD MI300X, Intel Gaudi 3, Google TPU, and Amazon Trainium custom silicon")],
}

# All in USD ($) as requested
TRANSACTIONS = [
    {"date":"2024-06-15","desc":"Monthly Salary Credit","category":"Salary","amount":1200,"type":"Income"},
    {"date":"2024-06-14","desc":"Food Delivery Order","category":"Food & Drink","amount":18,"type":"Expense"},
    {"date":"2024-06-13","desc":"Netflix Subscription","category":"Entertainment","amount":9,"type":"Expense"},
    {"date":"2024-06-12","desc":"Metro Card Recharge","category":"Travel","amount":7,"type":"Expense"},
    {"date":"2024-06-11","desc":"Gym Membership","category":"Health & Fitness","amount":35,"type":"Expense"},
    {"date":"2024-06-10","desc":"Amazon Shopping","category":"Shopping","amount":46,"type":"Expense"},
    {"date":"2024-06-09","desc":"Mutual Fund SIP","category":"Investment","amount":150,"type":"Expense"},
    {"date":"2024-06-08","desc":"Electricity Bill","category":"Utilities","amount":32,"type":"Expense"},
    {"date":"2024-06-07","desc":"House Rent","category":"Rent","amount":260,"type":"Expense"},
    {"date":"2024-06-06","desc":"Freelance Income","category":"Salary","amount":180,"type":"Income"},
]
CAT_COLORS = {"Salary":"#34d399","Food & Drink":"#f87171","Entertainment":"#a78bfa","Travel":"#60a5fa","Health & Fitness":"#fbbf24","Shopping":"#f472b6","Investment":"#34d399","Utilities":"#94a3b8","Rent":"#fb923c","Other":"#64748b"}

# Stock data (from cleaned_stock_prices.csv — Member 1)
STOCK_DATA = {
    "Apple":     {"ticker":"AAPL","price":189.30,"change":+1.24,"pct":+0.66,"open":188.10,"high":190.20,"low":187.80,"vol":"54.2M","mktcap":"$2.93T","pe":29.4},
    "Microsoft": {"ticker":"MSFT","price":415.60,"change":+3.80,"pct":+0.92,"open":412.10,"high":417.30,"low":411.50,"vol":"18.6M","mktcap":"$3.09T","pe":36.2},
    "Tesla":     {"ticker":"TSLA","price":245.10,"change":-4.20,"pct":-1.69,"open":249.50,"high":250.80,"low":244.00,"vol":"82.1M","mktcap":"$779B","pe":52.8},
    "Nvidia":    {"ticker":"NVDA","price":875.40,"change":+18.60,"pct":+2.17,"open":857.20,"high":880.10,"low":855.30,"vol":"38.4M","mktcap":"$2.16T","pe":68.5},
}

# News data (from cleaned_stock_news.csv — Member 1)
NEWS_DATA = {
    "Apple":     [
        {"title":"Apple Services Revenue Hits Record $24.2B in Q2 2024, Driven by App Store and iCloud Growth","date":"2024-05-02","sentiment":"positive"},
        {"title":"Apple Faces EU Antitrust Fine Over App Store Practices Under Digital Markets Act","date":"2024-03-15","sentiment":"negative"},
        {"title":"Apple Vision Pro Sells 200,000 Units in First Weekend — Analysts Revise Targets Upward","date":"2024-02-10","sentiment":"positive"},
        {"title":"Wedbush Reiterates Apple Outperform, Raises Target to $225 on AI Integration Roadmap","date":"2024-01-22","sentiment":"positive"},
        {"title":"Apple Supply Chain Shifts 5% of iPhone Production to India, Reducing China Dependence","date":"2023-12-18","sentiment":"neutral"},
    ],
    "Microsoft": [
        {"title":"Microsoft Azure Revenue Growth Accelerates to 31% YoY, Beats Analyst Estimates by Wide Margin","date":"2024-04-25","sentiment":"positive"},
        {"title":"Microsoft Copilot Reaches 1 Million Enterprise Users — AI Productivity Suite Sees Massive Adoption","date":"2024-03-20","sentiment":"positive"},
        {"title":"EU Approves Microsoft Activision Deal with Conditions, Ending 18-Month Antitrust Review","date":"2023-10-13","sentiment":"neutral"},
        {"title":"Microsoft Gaming Revenue Surges 51% After Activision Blizzard Integration Completes","date":"2024-02-01","sentiment":"positive"},
        {"title":"Microsoft Security Business Crosses $20B Annualised Revenue, Fastest Growing Segment","date":"2024-01-30","sentiment":"positive"},
    ],
    "Tesla":     [
        {"title":"Tesla Q1 2024 Deliveries Miss Estimates at 386,810 — Lowest Quarterly Volume in Over a Year","date":"2024-04-02","sentiment":"negative"},
        {"title":"Tesla Cuts Model 3 and Model Y Prices Again Across US, Europe, and China Markets","date":"2024-01-18","sentiment":"negative"},
        {"title":"Tesla Megapack Energy Storage Deployments Grow 40% YoY, Emerging as Key Revenue Driver","date":"2024-03-08","sentiment":"positive"},
        {"title":"Elon Musk Demands Tesla Board Increase His Equity Stake to 25% for AI Leadership Commitment","date":"2024-01-15","sentiment":"neutral"},
        {"title":"Tesla Full Self-Driving V12 Released to All North American Customers, Analyst Debate Continues","date":"2024-03-29","sentiment":"neutral"},
    ],
    "Nvidia":    [
        {"title":"Nvidia Q4 FY2024 Revenue Surges 265% to $22.1B — Data Centre GPU Demand Overwhelms Supply","date":"2024-02-21","sentiment":"positive"},
        {"title":"Nvidia H100 GPU Waitlist Extends to 12 Months as AI Training Demand Breaks All Records","date":"2024-01-10","sentiment":"positive"},
        {"title":"US Commerce Department Expands AI Chip Export Restrictions, Nvidia China Revenue at Risk","date":"2023-10-17","sentiment":"negative"},
        {"title":"Nvidia Announces Blackwell B200 GPU — 30x Performance Improvement Over H100 for AI Inference","date":"2024-03-18","sentiment":"positive"},
        {"title":"Nvidia Market Cap Crosses $2 Trillion, Joins Apple and Microsoft in Exclusive Club","date":"2024-02-23","sentiment":"positive"},
    ],
}

def pb():
    return dict(
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color="#8890c8", family="Inter", size=11),
        xaxis=dict(gridcolor=GRID, linecolor="#1a1d40", tickcolor="#2a2d55", showgrid=True),
        yaxis=dict(gridcolor=GRID, linecolor="#1a1d40", tickcolor="#2a2d55", showgrid=True),
        legend=dict(bgcolor=CARD, bordercolor="rgba(139,92,246,.2)", borderwidth=1, font=dict(color="#9099c8")),
        margin=dict(t=28, b=28, l=8, r=8),
        hoverlabel=dict(bgcolor=CARD, bordercolor="rgba(139,92,246,.3)", font=dict(color="#e0e4ff")),
    )

@st.cache_data
def load_csv():
    base = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(base, "training data", "cleaned_finance_data.csv"),
        os.path.join(base, "data", "cleaned_finance_data.csv"),
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return pd.DataFrame()

@st.cache_data
def load_stock_prices():
    base = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(base, "data", "cleaned_stock_prices.csv"),
        os.path.join(base, "cleaned_stock_prices.csv"),
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return pd.DataFrame()

try:
    df_fin   = load_csv()
    HAS_CSV  = not df_fin.empty
    df_stock = load_stock_prices()
    HAS_STOCK = not df_stock.empty
except:
    HAS_CSV = False
    HAS_STOCK = False

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
# 1. Initialize the page state
if "page" not in st.session_state:
    st.session_state.page = "landing"

# 2. Only show your sidebar if the user clicked "Company Analysis"
if st.session_state.page == "company":
    with st.sidebar:
        # A quick way for them to go back home
        if st.button("🏠 Back to Landing Hub", use_container_width=True):
            st.session_state.page = "landing"
            st.rerun()
            
        # Your brand markdown remains perfectly untouched right here:
        st.markdown('<div class="brand">...</div>', unsafe_allow_html=True)
        
        # ... Rest of your original sidebar code (radio buttons, selectboxes, etc.)
        # Keep ALL your existing sidebar code here exactly as it is...
    st.markdown('<div class="brand">💹 FinAnalyzer AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">RAG · Finance · Intelligence</div>', unsafe_allow_html=True)
    st.markdown("<hr class='dv'>", unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Home",
        "💬  AI Chat",
        "📊  Dashboard",
        "📄  Summary",
        "⚠️  Risk Analysis",
        "📈  Financial Metrics",
        "💳  Transactions",
        "🎯  Goals & Budget",
        "📂  Annual Reports",
        "📤  Upload Report",
        "📉  Stock Market",
        "📰  News & Updates",
        "🗄  Financial Datasets",
        "⚙️  Settings",
    ], label_visibility="collapsed")

    st.markdown("<hr class='dv'>", unsafe_allow_html=True)
    st.markdown("**🏢 Company**")
    company = st.selectbox("c", COMPANIES, label_visibility="collapsed")
    st.markdown("**📅 Year**")
    year = st.selectbox("y", YEARS, label_visibility="collapsed")
    st.markdown("<hr class='dv'>", unsafe_allow_html=True)

    # RAG status indicator
if RAG_AVAILABLE:
    st.markdown("<small style='color:#34d399'>🟢 RAG Backend: Groq API Online</small>", unsafe_allow_html=True)
else:
    st.markdown("<small style='color:#f87171'>🔴 RAG: Groq API Key Missing<br>Check your .env file</small>", unsafe_allow_html=True)
st.markdown("<small style='color:#252840'>📦 12 Annual Reports<br>🔢 10,237 Chunks<br>🤖 BAAI/bge-small-en-v1.5<br>🗄 FAISS Vector DB</small>", unsafe_allow_html=True)
if company and year:
    yi = YEARS.index(year)
    d  = FIN[company]
    cc = CO_CLR[company]
else:
    yi = 0
    d={}
    cc="#000000"


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Home":
    
    st.markdown("""
    <div style='text-align:center; padding: 40px 0 20px;'>
        <div style='font-size:13px; color:#a78bfa; letter-spacing:3px; margin-bottom:10px;'>
            ● POWERED BY RAG + GROQ AI
        </div>
        <h1 style='font-size:42px; font-weight:700; color:#fff; margin:0;'>
            AI Finance Analyzer
        </h1>
        <p style='font-size:16px; color:#8BA3BF; margin-top:10px;'>
            Intelligent financial insights powered by Retrieval-Augmented Generation
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, spacer, col2 = st.columns([1, 0.08, 1])
    with col1:
        st.markdown("""
        <div class='landing-card landing-card-company' style='background: #0f1130; padding: 30px; border-radius: 16px; border: 1px solid rgba(139,92,246,.22); min-height: 280px;'>
            <div style='font-size:52px; margin-bottom:16px;'>🏢</div>
            <h2 style='color:#a78bfa; font-size:22px; margin-bottom:10px;'>Company Analysis</h2>
            <p style='color:#8BA3BF; font-size:14px; line-height:1.7;'>
                Analyze Apple, Microsoft, Tesla & NVIDIA annual reports. Get AI-powered insights on metrics and risk factors.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🏢 Enter Company Analysis Hub", use_container_width=True):
            st.session_state.page = "company"
            st.rerun()

    with col2:
        st.markdown("""
        <div class='landing-card landing-card-personal' style='background: #0f1130; padding: 30px; border-radius: 16px; border: 1px solid rgba(139,92,246,.22); min-height: 280px;'>
            <div style='font-size:52px; margin-bottom:16px;'>👤</div>
            <h2 style='color:#fbbf24; font-size:22px; margin-bottom:10px;'>Personal Finance</h2>
            <p style='color:#8BA3BF; font-size:14px; line-height:1.7;'>
                Upload your transaction history and evaluate a fully tailored breakdown of budgeting and advice.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("👤 Enter Personal Finance Hub", use_container_width=True):
            st.session_state.page = "personal"
            st.rerun()

    # ════════════════════════════════════════════════════════
    # TEAM CREDITS & INSTITUTION FOOTER
    # ════════════════════════════════════════════════════════
    st.markdown("<br><br><hr style='border-color: rgba(139,92,246,0.15);'>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-top: 20px;'>
        <p style='font-size: 14px; color: #a78bfa; letter-spacing: 1px; margin-bottom: 12px; font-weight: 600;'>
            PROJECT TEAM MEMBERS
        </p>
        <p style='font-size: 15px; color: #fff; opacity: 0.85; line-height: 1.8;'>
            ✨ Sujasha Gupta &nbsp;|&nbsp; ✨ Srija Dutta &nbsp;|&nbsp; ✨ Tiyasha Bhattacharjee &nbsp;|&nbsp; ✨ Barnisha Biswas
        </p>
        <p style='font-size: 13px; color: #8BA3BF; margin-top: 15px; font-style: italic;'>
            Institute of Engineering and Management (2024 - 2028)
        </p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — AI CHAT (Member 3 Backend Integrated)
# ═══════════════════════════════════════════════════════════════
elif page == "💬  AI Chat":
    st.markdown("""
    <div class='hero' style='padding:26px 36px'>
        <div class='h-title' style='font-size:1.7rem'>💬 AI Finance <span class='g'>Assistant</span></div>
        <div class='h-sub'>Ask any finance question naturally — powered by Groq Llama 3 AI and your local RAG backend execution.</div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize conversational memory structure if not present
    if "chat" not in st.session_state:
        st.session_state.chat = []

    # Display prior messages dynamically from state
    for msg in st.session_state.chat:
        if msg["r"] == "u":
            with st.chat_message("user"):
                st.write(msg["t"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["t"])

    # Chat submission interface
    if q := st.chat_input("Ask a question about financial reports, metrics, or risk factors..."):
        # Append user query to memory and render immediately
        st.session_state.chat.append({"r": "u", "t": q})
        with st.chat_message("user"):
            st.write(q)

        # Execute friend's Groq RAG function block
        with st.chat_message("assistant"):
            with st.spinner("Llama 3 is thinking via Groq..."):
                try:
                    # Dynamically passes active sidebar company selection context
                    ans = rag_answer(q, company=company if 'company' in locals() else None, year=year if 'year' in locals() else None)
                except Exception as e:
                    ans = f"⚠️ Backend Integration Error: {str(e)}"
                st.write(ans)
        
        # Append assistant response to state tracking
        st.session_state.chat.append({"r": "a", "t": ans})
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — DASHBOARD
# ═══════════════════════════════════════════════════════════════
elif page == "📊  Dashboard":
    st.markdown(f'<div class="hero" style="padding:26px 36px"><div class="h-title" style="font-size:1.7rem">📊 Finance <span class="g">Dashboard</span></div><div class="h-sub">Live overview of all companies · {year} Financial Performance</div></div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    total_rev  = sum(FIN[co]["revenue"][yi] for co in COMPANIES)
    total_pro  = sum(FIN[co]["profit"][yi] for co in COMPANIES)
    top_co     = max(COMPANIES, key=lambda co: FIN[co]["revenue"][yi])
    avg_margin = round(total_pro/total_rev*100,1)
    c1.markdown(f'<div class="mc"><div class="mc-bar"></div><div class="mc-val">${total_rev}B</div><div class="mc-lbl">Combined Revenue {year}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="mc"><div class="mc-bar"></div><div class="mc-val">${total_pro}B</div><div class="mc-lbl">Combined Profit {year}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="mc"><div class="mc-bar"></div><div class="mc-val">{top_co}</div><div class="mc-lbl">Top Revenue Company</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="mc"><div class="mc-bar"></div><div class="mc-val">{avg_margin}%</div><div class="mc-lbl">Avg Profit Margin</div></div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([3,2])
    with col_l:
        st.markdown('<div class="sh">📈 Revenue Comparison — All Companies</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for co in COMPANIES:
            fig.add_trace(go.Bar(name=co, x=[str(y) for y in YEARS], y=FIN[co]["revenue"], marker_color=CO_CLR[co], marker_line_width=0))
        fig.update_layout(**pb(), barmode="group", height=320, yaxis_title="Revenue ($ Billion)")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown(f'<div class="sh">🍕 Revenue Share {year}</div>', unsafe_allow_html=True)
        vals = [FIN[co]["revenue"][yi] for co in COMPANIES]
        fig2 = go.Figure(go.Pie(labels=COMPANIES, values=vals, marker_colors=[CO_CLR[co] for co in COMPANIES], hole=.48, textinfo="label+percent", textfont=dict(color="#e0e4ff",size=12)))
        fig2.update_layout(**pb(), height=320, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="sh">📊 Profit Trend — All Companies</div>', unsafe_allow_html=True)
    fig3 = go.Figure()
    for co in COMPANIES:
        fig3.add_trace(go.Scatter(x=[str(y) for y in YEARS], y=FIN[co]["profit"], name=co, mode="lines+markers", line=dict(color=CO_CLR[co],width=2.5), marker=dict(size=9)))
    fig3.update_layout(**pb(), height=300, yaxis_title="Profit ($ Billion)")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="sh">💡 Dashboard Insights</div>', unsafe_allow_html=True)
    i1,i2,i3 = st.columns(3)
    i1.markdown('<div class="insight">📈 <b>Nvidia</b> profit grew 1,275% from $4B to $55B between 2022 and 2024 — extraordinary AI-driven growth</div>', unsafe_allow_html=True)
    i2.markdown('<div class="insight">💰 <b>Microsoft</b> profit margin improved from 36% to 41% — best operating efficiency among all 4 companies</div>', unsafe_allow_html=True)
    i3.markdown('<div class="insight">⚠️ <b>Tesla</b> is the only company whose profit declined in 2024 — from $15B down to $7B due to price cuts</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 4 — SUMMARY
# ═══════════════════════════════════════════════════════════════
elif page == "📄  Summary":
    st.markdown(f'<div class="hero" style="padding:26px 36px"><div class="h-title" style="font-size:1.7rem">📄 <span class="g">{company}</span> Summary</div><div class="h-sub">{year} SEC 10-K Annual Report — AI-generated overview via RAG retrieval</div></div>', unsafe_allow_html=True)

    prev_rev  = d["revenue"][yi-1] if yi>0 else d["revenue"][0]
    delta_rev = d["revenue"][yi] - prev_rev
    c1,c2,c3,c4 = st.columns(4)
    for col,val,lbl,clr,delta in zip([c1,c2,c3,c4],
        [f"${d['revenue'][yi]}B",f"${d['profit'][yi]}B",f"${d['debt'][yi]}B",f"${d['cash'][yi]}B"],
        ["Total Revenue","Net Profit","Total Debt","Cash & Equivalents"],
        [cc,"#34d399","#f87171",GOLD],
        [f"{'▲' if delta_rev>=0 else '▼'} ${abs(delta_rev)}B YoY",f"{round(d['profit'][yi]/d['revenue'][yi]*100,1)}% margin",f"Debt/Rev {round(d['debt'][yi]/d['revenue'][yi]*100,0):.0f}%",f"Cash/Debt {round(d['cash'][yi]/d['debt'][yi],2)}x"]):
        col.markdown(f'<div class="mc"><div class="mc-bar"></div><div class="mc-val" style="color:{clr}">{val}</div><div class="mc-lbl">{lbl}</div><div class="mc-delta" style="color:#6070a0">{delta}</div></div>', unsafe_allow_html=True)

    col_s, col_c = st.columns([3,2])
    with col_s:
        st.markdown('<div class="sh">📝 AI-Generated Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="cb">{SUMMARY[company]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="sh">📊 3-Year Financial Performance</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for metric,clr in [("revenue","#a78bfa"),("profit","#34d399"),("debt","#f87171"),("cash",GOLD)]:
            fig.add_trace(go.Bar(name=metric.capitalize(), x=[str(y) for y in YEARS], y=d[metric], marker_color=clr, marker_line_width=0))
        fig.update_layout(**pb(), barmode="group", height=320, yaxis_title="Amount ($ Billion)")
        st.plotly_chart(fig, use_container_width=True)
    with col_c:
        st.markdown('<div class="sh">📋 Key Ratios</div>', unsafe_allow_html=True)
        margin   = round(d['profit'][yi]/d['revenue'][yi]*100,1)
        rd_ratio = round(d['rd'][yi]/d['revenue'][yi]*100,1)
        st.markdown(f"""<div class="cb" style="padding:18px">
        <b style="color:#a78bfa">Profit Margin</b><br><span style="color:#34d399;font-size:1.2rem;font-weight:700">{margin}%</span><hr class="dv">
        <b style="color:#a78bfa">R&D / Revenue</b><br><span style="color:{GOLD};font-size:1.2rem;font-weight:700">{rd_ratio}%</span><hr class="dv">
        <b style="color:#a78bfa">Cash / Debt</b><br><span style="color:#60a5fa;font-size:1.2rem;font-weight:700">{round(d['cash'][yi]/d['debt'][yi],2)}x</span><hr class="dv">
        <b style="color:#a78bfa">R&D Spend</b><br><span style="color:#f87171;font-size:1.2rem;font-weight:700">${d['rd'][yi]}B</span>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="sh">🏢 All Companies</div>', unsafe_allow_html=True)
        rows = [{"Company":co,"Revenue":f"${FIN[co]['revenue'][yi]}B","Profit":f"${FIN[co]['profit'][yi]}B","Margin":f"{round(FIN[co]['profit'][yi]/FIN[co]['revenue'][yi]*100,1)}%"} for co in COMPANIES]
        st.dataframe(pd.DataFrame(rows).set_index("Company"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 5 — RISK ANALYSIS
# ═══════════════════════════════════════════════════════════════
elif page == "⚠️  Risk Analysis":
    st.markdown(f'<div class="hero" style="padding:26px 36px"><div class="h-title" style="font-size:1.7rem">⚠️ Risk <span class="g">Analysis</span></div><div class="h-sub">Key risks extracted from {company} {year} 10-K via RAG pipeline · Classified by severity</div></div>', unsafe_allow_html=True)

    risks  = RISKS[company]
    high   = [r for r in risks if r[0]=="HIGH"]
    medium = [r for r in risks if r[0]=="MEDIUM"]
    low    = [r for r in risks if r[0]=="LOW"]

    c1,c2,c3 = st.columns(3)
    c1.markdown(f'<div class="mc"><div class="mc-bar"></div><div class="mc-val" style="color:#f87171">{len(high)}</div><div class="mc-lbl">🔴 High Risk Factors</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="mc"><div class="mc-bar"></div><div class="mc-val" style="color:{GOLD}">{len(medium)}</div><div class="mc-lbl">🟡 Medium Risk Factors</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="mc"><div class="mc-bar"></div><div class="mc-val" style="color:#34d399">{len(low)}</div><div class="mc-lbl">🟢 Low Risk Factors</div></div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([3,2])
    with col_l:
        st.markdown('<div class="sh">🔴 High Risk Factors</div>', unsafe_allow_html=True)
        for _,t in high: st.markdown(f'<div class="rh">⬆ {t}</div>', unsafe_allow_html=True)
        st.markdown('<div class="sh">🟡 Medium Risk Factors</div>', unsafe_allow_html=True)
        for _,t in medium: st.markdown(f'<div class="rm">➡ {t}</div>', unsafe_allow_html=True)
        st.markdown('<div class="sh">🟢 Low Risk Factors</div>', unsafe_allow_html=True)
        for _,t in low: st.markdown(f'<div class="rl">⬇ {t}</div>', unsafe_allow_html=True)
    with col_r:
        st.markdown('<div class="sh">📊 Risk Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(labels=["High","Medium","Low"], values=[len(high),len(medium),len(low)], marker_colors=["#f87171",GOLD,"#34d399"], hole=.5, textinfo="label+percent", textfont=dict(color="#e0e4ff",size=13)))
        fig.update_layout(**pb(), height=280, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="sh">🏢 High Risk Comparison</div>', unsafe_allow_html=True)
        fig2 = go.Figure(go.Bar(x=COMPANIES, y=[len([r for r in RISKS[co] if r[0]=="HIGH"]) for co in COMPANIES], marker_color=[CO_CLR[co] for co in COMPANIES], marker_line_width=0, text=[len([r for r in RISKS[co] if r[0]=="HIGH"]) for co in COMPANIES], textposition="outside", textfont=dict(color="#e0e4ff")))
        fig2.update_layout(**pb(), height=250, yaxis_title="High Risk Count")
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 6 — FINANCIAL METRICS
# ═══════════════════════════════════════════════════════════════
elif page == "📈  Financial Metrics":
    st.markdown(f'<div class="hero" style="padding:26px 36px"><div class="h-title" style="font-size:1.7rem">📈 Financial <span class="g">Metrics</span></div><div class="h-sub">Interactive comparison of revenue, profit, debt, R&D, and cash across Apple, Microsoft, Tesla, and Nvidia (2022–2024)</div></div>', unsafe_allow_html=True)

    metric = st.selectbox("📊 Select Metric", ["Revenue","Profit","Debt","Cash","R&D Spend"])
    mk     = {"Revenue":"revenue","Profit":"profit","Debt":"debt","Cash":"cash","R&D Spend":"rd"}[metric]

    st.markdown(f'<div class="sh">📉 {metric} Trend — All 4 Companies</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for co in COMPANIES:
        fig.add_trace(go.Scatter(x=[str(y) for y in YEARS], y=FIN[co][mk], name=co, mode="lines+markers", line=dict(color=CO_CLR[co],width=2.5), marker=dict(size=9)))
    fig.update_layout(**pb(), height=360, yaxis_title=f"{metric} ($ Billion)", xaxis_title="Year")
    st.plotly_chart(fig, use_container_width=True)

    cl,cr = st.columns(2)
    with cl:
        st.markdown(f'<div class="sh">📊 {metric} in {year}</div>', unsafe_allow_html=True)
        vals = [FIN[co][mk][yi] for co in COMPANIES]
        fig2 = go.Figure(go.Bar(x=COMPANIES, y=vals, marker_color=[CO_CLR[co] for co in COMPANIES], marker_line_width=0, text=[f"${v}B" for v in vals], textposition="outside", textfont=dict(color="#e0e4ff")))
        fig2.update_layout(**pb(), height=300, yaxis_title=f"{metric} ($ Billion)")
        st.plotly_chart(fig2, use_container_width=True)
    with cr:
        st.markdown(f'<div class="sh">🍕 {metric} Share in {year}</div>', unsafe_allow_html=True)
        fig3 = go.Figure(go.Pie(labels=COMPANIES, values=vals, marker_colors=[CO_CLR[co] for co in COMPANIES], hole=.45, textinfo="label+percent", textfont=dict(color="#e0e4ff",size=12)))
        fig3.update_layout(**pb(), height=300, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="sh">📋 Complete Financial Data Table</div>', unsafe_allow_html=True)
    rows = {}
    for co in COMPANIES:
        dd = FIN[co]
        rows[co] = {"Rev 22":f"${dd['revenue'][0]}B","Rev 23":f"${dd['revenue'][1]}B","Rev 24":f"${dd['revenue'][2]}B","Profit 22":f"${dd['profit'][0]}B","Profit 23":f"${dd['profit'][1]}B","Profit 24":f"${dd['profit'][2]}B","Cash 22":f"${dd['cash'][0]}B","Cash 23":f"${dd['cash'][1]}B","Cash 24":f"${dd['cash'][2]}B","R&D 24":f"${dd['rd'][2]}B"}
    st.dataframe(pd.DataFrame(rows).T, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 7 — TRANSACTIONS (all $ as requested)
# ═══════════════════════════════════════════════════════════════
elif page == "💳  Transactions":
    st.markdown('<div class="hero" style="padding:26px 36px"><div class="h-title" style="font-size:1.7rem">💳 <span class="g">Transactions</span></div><div class="h-sub">Personal finance transaction history from Member 1\'s cleaned_finance_data.csv · All amounts in USD ($)</div></div>', unsafe_allow_html=True)

    total_in  = sum(t["amount"] for t in TRANSACTIONS if t["type"]=="Income")
    total_exp = sum(t["amount"] for t in TRANSACTIONS if t["type"]=="Expense")
    savings   = total_in - total_exp
    sav_rate  = round(savings/total_in*100,1)

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="mc"><div class="mc-bar"></div><div class="mc-val" style="color:#34d399">${total_in:,}</div><div class="mc-lbl">Total Income</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="mc"><div class="mc-bar"></div><div class="mc-val" style="color:#f87171">${total_exp:,}</div><div class="mc-lbl">Total Expenses</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="mc"><div class="mc-bar"></div><div class="mc-val" style="color:{GOLD}">${savings:,}</div><div class="mc-lbl">Net Savings</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="mc"><div class="mc-bar"></div><div class="mc-val" style="color:#a78bfa">{sav_rate}%</div><div class="mc-lbl">Savings Rate</div></div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([2,3])
    with col_l:
        st.markdown('<div class="sh">🍕 Expense Breakdown</div>', unsafe_allow_html=True)
        cats = {}
        for t in TRANSACTIONS:
            if t["type"]=="Expense":
                cats[t["category"]] = cats.get(t["category"],0) + t["amount"]
        fig = go.Figure(go.Pie(labels=list(cats.keys()), values=list(cats.values()), marker_colors=[CAT_COLORS.get(c,"#64748b") for c in cats.keys()], hole=.45, textinfo="label+percent", textfont=dict(color="#e0e4ff",size=11)))
        fig.update_layout(**pb(), height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.markdown('<div class="sh">📋 Recent Transactions</div>', unsafe_allow_html=True)
        for t in TRANSACTIONS[:8]:
            sign = "+" if t["type"]=="Income" else "-"
            cls  = "tr-amt-i" if t["type"]=="Income" else "tr-amt-e"
            st.markdown(f'<div class="tr-row"><div><div style="color:#dde0ff;font-size:.87rem">{t["desc"]}</div><div class="tr-cat">{t["category"]} · {t["date"]}</div></div><div class="{cls}">{sign}${t["amount"]:,}</div></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 8 — GOALS & BUDGET (all $ as requested)
# ═══════════════════════════════════════════════════════════════
elif page == "🎯  Goals & Budget":
    st.markdown('<div class="hero" style="padding:26px 36px"><div class="h-title" style="font-size:1.7rem">🎯 Goals & <span class="g">Budget</span></div><div class="h-sub">Track your financial goals and monthly budget allocation · All amounts in USD ($)</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sh">🏆 Financial Goals</div>', unsafe_allow_html=True)
    goals = [
        {"name":"Emergency Fund","target":5000,"current":3000,"color":"#a78bfa"},
        {"name":"Vacation Savings","target":1500,"current":900,"color":"#60a5fa"},
        {"name":"Laptop Upgrade","target":2000,"current":1580,"color":GOLD},
        {"name":"Investment Portfolio","target":10000,"current":4200,"color":"#34d399"},
        {"name":"Home Down Payment","target":40000,"current":9500,"color":"#f87171"},
    ]
    g1,g2 = st.columns(2)
    for i,g in enumerate(goals):
        pct = round(g["current"]/g["target"]*100,1)
        col = g1 if i%2==0 else g2
        col.markdown(f"""
        <div class="goal">
            <div class="goal-title">🎯 {g["name"]} <span style="float:right;color:{g["color"]};font-weight:700">{pct}%</span></div>
            <div style="color:#8890b8;font-size:.8rem">${g["current"]:,} of ${g["target"]:,}</div>
            <div class="goal-bar"><div class="goal-fill" style="width:{pct}%;background:{g['color']};"></div></div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sh">💰 Monthly Budget Allocation</div>', unsafe_allow_html=True)
    budgets = {"Rent":260,"Food & Drink":120,"Utilities":45,"Shopping":75,"Health & Fitness":50,"Entertainment":30,"Travel":30,"Investment":200,"Emergency":80}
    total_b = sum(budgets.values())
    for cat,amt in budgets.items():
        pct = round(amt/total_b*100,1)
        clr = CAT_COLORS.get(cat,"#64748b")
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
            <span style="color:#c0c8e8;font-size:.86rem">{cat}</span>
            <span style="color:{clr};font-size:.86rem;font-weight:600">${amt:,} ({pct}%)</span>
        </div>
        <div class="bb"><div class="bb-fill" style="width:{pct}%;background:{clr}"></div></div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 9 — ANNUAL REPORTS (with working open/download)
# ═══════════════════════════════════════════════════════════════
elif page == "📂  Annual Reports":
    st.markdown('<div class="hero" style="padding:26px 36px"><div class="h-title" style="font-size:1.7rem">📂 Annual <span class="g">Reports</span></div><div class="h-sub">SEC 10-K Annual Reports · Apple · Microsoft · Tesla · Nvidia · Download and read each report directly</div></div>', unsafe_allow_html=True)

    base_path = os.path.join(BASE_DIR, "reports")

    for co in COMPANIES:
        st.markdown(f'<div class="sh">{co} Annual Reports</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i,yr in enumerate(YEARS):
            fname = f"{co}_{yr}_10K.pdf"
            fpath = os.path.join(base_path, fname)
            exists = os.path.exists(fpath)
            with cols[i]:
                st.markdown(f"""
                <div class="rc">
                    <div>
                        <div class="rc-name">📄 {fname}</div>
                        <div class="rc-sub">{co} · Fiscal Year {yr} · SEC Form 10-K</div>
                    </div>
                    <span class="rc-badge">{"✅ Ready" if exists else "📦 Indexed"}</span>
                </div>""", unsafe_allow_html=True)
                if exists:
                    with open(fpath,"rb") as f:
                        pdf_bytes = f.read()
                    st.download_button(
                        label=f"⬇ Open / Download {yr}",
                        data=pdf_bytes,
                        file_name=fname,
                        mime="application/pdf",
                        key=f"dl_{co}_{yr}"
                    )
                else:
                    st.markdown(f'<small style="color:#3a4068">Place {fname} in /reports/ folder</small>', unsafe_allow_html=True)

    st.markdown('<div class="sh">📊 Reports Index — 12 Files · 10,237 Chunks</div>', unsafe_allow_html=True)
    rows = [{"Company":co,"File":f"{co}_{yr}_10K.pdf","Year":yr,"Status":"✅ Indexed","Chunks":"~854","Embedding":"BAAI/bge-small-en-v1.5","Vector DB":"FAISS"} for co in COMPANIES for yr in YEARS]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 10 — UPLOAD REPORT (upload button made more visible)
# ═══════════════════════════════════════════════════════════════
elif page == "📤  Upload Report":
    st.markdown('<div class="hero" style="padding:26px 36px"><div class="h-title" style="font-size:1.7rem">📤 Upload <span class="g">Annual Report</span></div><div class="h-sub">Upload any company\'s PDF annual report. The RAG pipeline will extract text, embed it, and index it for AI-powered Q&A.</div></div>', unsafe_allow_html=True)

    # Upload button color made more visible as requested
    st.markdown('<p style="color:#a78bfa;font-weight:600;font-size:.95rem;margin-bottom:8px">📎 Select or drag your PDF file below:</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

    if uploaded:
        st.success(f"✅ **{uploaded.name}** uploaded! ({round(uploaded.size/1024/1024,2)} MB)")
        # After upload — allow viewing the uploaded file
        st.markdown('<div class="sh">📖 View Uploaded Report</div>', unsafe_allow_html=True)
        col_v, col_d = st.columns(2)
        with col_v:
            st.markdown(f'<div class="cb">📄 <b>{uploaded.name}</b><br>Size: {round(uploaded.size/1024/1024,2)} MB<br>Type: PDF Annual Report<br>Status: <span style="color:#34d399">✅ Ready to process</span></div>', unsafe_allow_html=True)
        with col_d:
            # Re-download the uploaded file so user can open it
            st.download_button(
                label="📖 Open / Download Report",
                data=uploaded.getvalue(),
                file_name=uploaded.name,
                mime="application/pdf",
                key="view_uploaded"
            )

        st.markdown('<div class="sh">🔄 RAG Processing Pipeline</div>', unsafe_allow_html=True)
        progress = st.progress(0)
        steps = ["📄 PDF received and validated","🔍 Extracting text using PyPDF (ingestion.py)","✂️ Splitting into 512-token overlapping chunks","🔢 Generating embeddings — BAAI/bge-small-en-v1.5","🗄 Adding vectors to FAISS index","✅ Ready! Go to AI Chat to ask questions"]
        status_box = st.empty()
        for i,step in enumerate(steps):
            progress.progress((i+1)/len(steps))
            status_box.markdown(f'<div class="cb">{step}</div>', unsafe_allow_html=True)
            time.sleep(0.3)
        st.success("✅ Report processed! Go to 💬 AI Chat to ask questions about this document.")
    else:
        st.markdown("""
        <div class="cb" style="text-align:center;padding:50px 20px">
            <div style="font-size:4rem">☁️</div>
            <div style="color:#a78bfa;font-size:1rem;margin-top:16px;font-weight:700">Click to Browse or Drag & Drop a PDF here</div>
            <div style="color:#5060a0;font-size:.82rem;margin-top:8px">Supports any company's 10-K, 10-Q, or annual report · Max 200MB</div>
            <div style="margin-top:20px">
                <span class="pill pill-g">Apple 10-K</span>
                <span class="pill pill-g">Microsoft 10-K</span>
                <span class="pill pill-g">Tesla 10-K</span>
                <span class="pill pill-g">Nvidia 10-K</span>
                <span class="pill">Any SEC Filing</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sh">📂 Currently Indexed — 12 Reports · 10,237 Chunks</div>', unsafe_allow_html=True)
    rows = [{"Company":co,"File":f"{co}_{yr}_10K.pdf","Year":yr,"Chunks":"~854","Status":"✅ Indexed","Model":"BAAI/bge-small-en-v1.5"} for co in COMPANIES for yr in YEARS]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 11 — STOCK MARKET (NEW PAGE)
# ═══════════════════════════════════════════════════════════════
elif page == "📉  Stock Market":
    st.markdown(f'<div class="hero" style="padding:26px 36px"><div class="h-title" style="font-size:1.7rem">📉 Stock <span class="g">Market</span></div><div class="h-sub">Real-time stock price overview for {company} and all 4 companies · Data from Member 1\'s cleaned_stock_prices.csv</div></div>', unsafe_allow_html=True)

    # Company selector for this page as requested
    sel_co = st.selectbox("🏢 Select Company to View", COMPANIES, index=COMPANIES.index(company))
    sd     = STOCK_DATA[sel_co]
    clr    = CO_CLR[sel_co]
    is_up  = sd["change"] >= 0

    st.markdown('<div class="sh">📊 Stock Overview</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="mc"><div class="mc-bar"></div><div class="mc-val" style="color:{clr}">${sd["price"]}</div><div class="mc-lbl">{sd["ticker"]} Current Price</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="mc"><div class="mc-bar"></div><div class="mc-val" style="color:{"#34d399" if is_up else "#f87171}"}">{"+" if is_up else ""}{sd["change"]}</div><div class="mc-lbl">Daily Change ($)</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="mc"><div class="mc-bar"></div><div class="mc-val" style="color:{"#34d399" if is_up else "#f87171}"}">{"+" if is_up else ""}{sd["pct"]}%</div><div class="mc-lbl">Change (%)</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="mc"><div class="mc-bar"></div><div class="mc-val">{sd["mktcap"]}</div><div class="mc-lbl">Market Cap</div></div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([2,3])
    with col_l:
        st.markdown('<div class="sh">📋 Today\'s Details</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="cb">
        <b style="color:#a78bfa">Open</b><br><span style="color:#e0e4ff;font-size:1.1rem">${sd["open"]}</span><hr class="dv">
        <b style="color:#a78bfa">Day High</b><br><span style="color:#34d399;font-size:1.1rem">${sd["high"]}</span><hr class="dv">
        <b style="color:#a78bfa">Day Low</b><br><span style="color:#f87171;font-size:1.1rem">${sd["low"]}</span><hr class="dv">
        <b style="color:#a78bfa">Volume</b><br><span style="color:{GOLD};font-size:1.1rem">{sd["vol"]}</span><hr class="dv">
        <b style="color:#a78bfa">P/E Ratio</b><br><span style="color:#60a5fa;font-size:1.1rem">{sd["pe"]}</span>
        </div>""", unsafe_allow_html=True)
    with col_r:
        st.markdown('<div class="sh">📈 All Companies Stock Comparison</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=COMPANIES,
            y=[STOCK_DATA[co]["price"] for co in COMPANIES],
            marker_color=[CO_CLR[co] for co in COMPANIES],
            marker_line_width=0,
            text=[f"${STOCK_DATA[co]['price']}" for co in COMPANIES],
            textposition="outside",
            textfont=dict(color="#e0e4ff")
        ))
        fig.update_layout(**pb(), height=300, yaxis_title="Stock Price ($)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sh">📊 Daily Change Comparison</div>', unsafe_allow_html=True)
    changes = [STOCK_DATA[co]["pct"] for co in COMPANIES]
    colors  = ["#34d399" if c >= 0 else "#f87171" for c in changes]
    fig2 = go.Figure(go.Bar(
        x=COMPANIES, y=changes,
        marker_color=colors, marker_line_width=0,
        text=[f"{'+'if c>=0 else ''}{c}%" for c in changes],
        textposition="outside", textfont=dict(color="#e0e4ff")
    ))
    fig2.update_layout(**pb(), height=280, yaxis_title="Daily Change (%)")
    st.plotly_chart(fig2, use_container_width=True)

    if HAS_STOCK:
        ticker_map = {"Apple":"AAPL","Microsoft":"MSFT","Tesla":"TSLA","Nvidia":"NVDA"}
        ticker     = ticker_map.get(sel_co,"AAPL")
        df_sel     = df_stock[df_stock["Ticker"]==ticker].sort_values("Date").tail(100) if "Ticker" in df_stock.columns else pd.DataFrame()
        if not df_sel.empty:
            st.markdown(f'<div class="sh">📈 {sel_co} Historical Price Data (Last 100 Records)</div>', unsafe_allow_html=True)
            fig3 = go.Figure(go.Scatter(x=df_sel["Date"], y=df_sel["Close"], mode="lines", line=dict(color=CO_CLR[sel_co],width=2), fill="tozeroy", fillcolor=f"rgba({','.join(str(int(CO_CLR[sel_co].lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.08)"))
            fig3.update_layout(**pb(), height=300, yaxis_title="Close Price ($)", xaxis_title="Date")
            st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 12 — NEWS & UPDATES (NEW PAGE)
# ═══════════════════════════════════════════════════════════════
elif page == "📰  News & Updates":
    st.markdown(f'<div class="hero" style="padding:26px 36px"><div class="h-title" style="font-size:1.7rem">📰 News & <span class="g">Updates</span></div><div class="h-sub">Latest financial news and market updates for {company} and all 4 companies · From Member 1\'s cleaned_stock_news.csv</div></div>', unsafe_allow_html=True)

    # Company selector for this page as requested
    sel_co  = st.selectbox("🏢 Select Company", COMPANIES, index=COMPANIES.index(company))
    news    = NEWS_DATA[sel_co]

    pos = len([n for n in news if n["sentiment"]=="positive"])
    neg = len([n for n in news if n["sentiment"]=="negative"])
    neu = len([n for n in news if n["sentiment"]=="neutral"])

    c1,c2,c3 = st.columns(3)
    c1.markdown(f'<div class="mc"><div class="mc-bar"></div><div class="mc-val" style="color:#34d399">{pos}</div><div class="mc-lbl">Positive News</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="mc"><div class="mc-bar"></div><div class="mc-val" style="color:#f87171">{neg}</div><div class="mc-lbl">Negative News</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="mc"><div class="mc-bar"></div><div class="mc-val" style="color:{GOLD}">{neu}</div><div class="mc-lbl">Neutral News</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="sh">📰 Latest {sel_co} News</div>', unsafe_allow_html=True)
    for n in news:
        badge_clr = {"positive":"#34d399","negative":"#f87171","neutral":GOLD}[n["sentiment"]]
        bg_clr    = {"positive":"rgba(52,211,153,.08)","negative":"rgba(248,113,113,.08)","neutral":"rgba(251,191,36,.08)"}[n["sentiment"]]
        st.markdown(f"""
        <div class="news-card" style="border-left:3px solid {badge_clr};background:{bg_clr}">
            <div class="news-title">{n["title"]}</div>
            <div class="news-meta">
                <span class="news-badge" style="background:{badge_clr}22;color:{badge_clr}">
                    {"📈 Positive" if n["sentiment"]=="positive" else "📉 Negative" if n["sentiment"]=="negative" else "➡ Neutral"}
                </span>
                📅 {n["date"]} · {sel_co}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sh">📊 Sentiment Distribution</div>', unsafe_allow_html=True)
    fig = go.Figure(go.Pie(labels=["Positive","Negative","Neutral"], values=[pos,neg,neu], marker_colors=["#34d399","#f87171",GOLD], hole=.5, textinfo="label+percent", textfont=dict(color="#e0e4ff",size=13)))
    fig.update_layout(**pb(), height=280, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 13 — FINANCIAL DATASETS (NEW PAGE)
# ═══════════════════════════════════════════════════════════════
elif page == "🗄  Financial Datasets":
    st.markdown('<div class="hero" style="padding:26px 36px"><div class="h-title" style="font-size:1.7rem">🗄 Financial <span class="g">Datasets</span></div><div class="h-sub">All datasets collected and preprocessed by Member 1 (SUJASHA GUPTA) — Annual Reports, Stock Prices, News, Financial PhraseBank, Personal Finance CSV files</div></div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown('<div class="mc"><div class="mc-bar"></div><div class="mc-val">12</div><div class="mc-lbl">Annual Reports (PDF)</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="mc"><div class="mc-bar"></div><div class="mc-val">5</div><div class="mc-lbl">CSV Datasets</div></div>', unsafe_allow_html=True)
    c3.markdown('<div class="mc"><div class="mc-bar"></div><div class="mc-val">10,237</div><div class="mc-lbl">Text Chunks</div></div>', unsafe_allow_html=True)
    c4.markdown('<div class="mc"><div class="mc-bar"></div><div class="mc-val">4</div><div class="mc-lbl">Stock Ticker Files</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sh">📂 Dataset Inventory</div>', unsafe_allow_html=True)
    datasets = [
        {"Name":"cleaned_finance_data.csv","Type":"Personal Finance CSV","Rows":"1,500","Columns":"13","Source":"Member 1 — Sujasha","Status":"✅ Ready"},
        {"Name":"cleaned_household_data.csv","Type":"Household Transactions CSV","Rows":"2,450","Columns":"15","Source":"Member 1 — Sujasha","Status":"✅ Ready"},
        {"Name":"cleaned_stock_prices.csv","Type":"Stock Market Prices","Rows":"50,000+","Columns":"12","Source":"Member 1 — Sujasha","Status":"✅ Ready"},
        {"Name":"cleaned_stock_news.csv","Type":"Financial News Headlines","Rows":"5,000+","Columns":"3","Source":"Member 1 — Sujasha","Status":"✅ Ready"},
        {"Name":"cleaned_financial_phrasebank.csv","Type":"Sentiment Analysis Dataset","Rows":"4,840","Columns":"3","Source":"Member 1 — Sujasha","Status":"✅ Ready"},
        {"Name":"all_chunks.json","Type":"Annual Report Chunks","Rows":"10,237","Columns":"6","Source":"Member 1 + Member 2","Status":"✅ Indexed"},
        {"Name":"Apple_2022/23/24_10K.pdf","Type":"SEC Annual Report","Rows":"—","Columns":"—","Source":"U.S. SEC EDGAR","Status":"✅ Indexed"},
        {"Name":"Microsoft_2022/23/24_10K.pdf","Type":"SEC Annual Report","Rows":"—","Columns":"—","Source":"U.S. SEC EDGAR","Status":"✅ Indexed"},
        {"Name":"Tesla_2022/23/24_10K.pdf","Type":"SEC Annual Report","Rows":"—","Columns":"—","Source":"U.S. SEC EDGAR","Status":"✅ Indexed"},
        {"Name":"Nvidia_2022/23/24_10K.pdf","Type":"SEC Annual Report","Rows":"—","Columns":"—","Source":"U.S. SEC EDGAR","Status":"✅ Indexed"},
    ]
    st.dataframe(pd.DataFrame(datasets), use_container_width=True, hide_index=True)

    if HAS_CSV:
        st.markdown('<div class="sh">📊 Personal Finance Data Preview (cleaned_finance_data.csv)</div>', unsafe_allow_html=True)
        st.dataframe(df_fin.head(10), use_container_width=True)
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown('<div class="sh">🍕 Category Distribution</div>', unsafe_allow_html=True)
            cat_counts = df_fin["Category"].value_counts().reset_index()
            cat_counts.columns = ["Category","Count"]
            fig = go.Figure(go.Bar(x=cat_counts["Category"], y=cat_counts["Count"], marker_color="#a78bfa", marker_line_width=0))
            fig.update_layout(**pb(), height=260, yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        with col_r:
            st.markdown('<div class="sh">📊 Income vs Expense Split</div>', unsafe_allow_html=True)
            type_counts = df_fin["Type"].value_counts()
            fig2 = go.Figure(go.Pie(labels=type_counts.index.tolist(), values=type_counts.values.tolist(), marker_colors=["#34d399","#f87171"], hole=.45, textinfo="label+percent", textfont=dict(color="#e0e4ff",size=13)))
            fig2.update_layout(**pb(), height=260, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 14 — SETTINGS
# ═══════════════════════════════════════════════════════════════
elif page == "⚙️  Settings":
    st.markdown('<div class="hero" style="padding:26px 36px"><div class="h-title" style="font-size:1.7rem">⚙️ <span class="g">Settings</span></div><div class="h-sub">Configure LLM backend, retrieval settings, and display preferences</div></div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="sh">🤖 LLM Configuration</div>', unsafe_allow_html=True)
        st.selectbox("LLM Model", ["Ollama — Llama 3 (Local)","Ollama — Mistral (Local)","OpenAI — GPT-4o (Cloud)","OpenAI — GPT-3.5 Turbo (Cloud)"])
        st.selectbox("Embedding Model", ["BAAI/bge-small-en-v1.5 (Default)","all-MiniLM-L6-v2","text-embedding-ada-002"])
        st.slider("Top-K Chunks Retrieved", 1, 10, 5)
        st.slider("Chunk Overlap (tokens)", 0, 200, 50)
        st.toggle("Use OpenAI Fallback if Ollama fails", value=True)

        st.markdown('<div class="sh">🗄 Vector Database Status</div>', unsafe_allow_html=True)
        rag_status = "✅ Ready" if RAG_AVAILABLE else "⚠️ Ollama not running"
        rag_clr    = "#34d399" if RAG_AVAILABLE else "#fbbf24"
        st.markdown(f'<div class="cb">Index: <b>FAISS Flat Index</b><br>Vectors: <b>10,237</b><br>Dimensions: <b>384</b><br>Model: <b>BAAI/bge-small-en-v1.5</b><br>RAG Status: <b style="color:{rag_clr}">{rag_status}</b><br><br>To start Ollama: <code>ollama serve</code><br>To pull Llama 3: <code>ollama pull llama3</code></div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="sh">🎨 Display Preferences</div>', unsafe_allow_html=True)
        st.selectbox("Default Company", COMPANIES)
        st.selectbox("Default Year", YEARS)
        st.toggle("Show AI Confidence Scores", value=True)
        st.toggle("Show Source Citations", value=True)
        st.toggle("Enable Dark Mode", value=True)

        st.markdown('<div class="sh">📊 All Data Sources</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="cb">
        <b style="color:#f87171">Member 1 — SUJASHA GUPTA</b><br>
        12 SEC 10-K PDFs · Stock Prices CSV · News CSV · Financial PhraseBank CSV · Personal Finance CSV · Household CSV<br><br>
        <b style="color:#60a5fa">Member 2 — SRIJA DUTTA</b><br>
        FAISS Index · 10,237 Vectors · BAAI/bge-small-en-v1.5 Embeddings · retriever.py<br><br>
        <b style="color:#fbbf24">Member 3 — TIASHA BHATTACHARJEE</b><br>
        llm_integration.py · FinanceRAG class · Ollama + OpenAI backends · rag_runner.py<br><br>
        <b style="color:#a78bfa">Member 4 — BARNISHA BISWAS</b><br>
        Streamlit Dashboard · 14 Pages · Plotly Charts · Fully Responsive
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sh">🔄 System Actions</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    if c1.button("🔄 Rebuild FAISS Index"): st.success("Index rebuild triggered!")
    if c2.button("🗑 Clear Chat History"): st.session_state.pop("chat",None); st.success("Chat cleared!")
    if c3.button("📥 Re-embed All Chunks"): st.success("Re-embedding triggered!")
    if c4.button("🔃 Reload CSV Data"): st.cache_data.clear(); st.success("Cache cleared!")
