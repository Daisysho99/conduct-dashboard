import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="BNM Conduct Risk Dashboard",
    page_icon="🏦",
    layout="wide",
)

# ---------- BNM CUSTOM CSS ----------
BNM_STYLE = """
<style>

/* BNM Header Banner */
.banner {
    background-color: #003B88;
    padding: 22px;
    border-radius: 8px;
    color: white;
    margin-bottom: 25px;
    text-align: center;
    font-family: 'Arial', sans-serif;
}
.banner h1 {
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 4px;
}

/* Force center alignment for ALL st.image containers */
div[data-testid="stImage"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
}

/* Center Streamlit images (logo) */
div[data-testid="stImage"] img {
    display: block;
    margin-left: auto;
    margin-right: auto;
    margin-top: 10px;
    margin-bottom: 5px;
}

/* Prototype tag under the title */
.prototype-tag {
    font-size: 18px;
    font-weight: 400;
    opacity: 0.9;
    margin-bottom: 6px;
}

/* Subtitle */
.banner p {
    font-size: 14px;
    margin-top: 0px;
}

/* KPI Card Style */
.metric-container {
    background-color: #E6EEF8;
    padding: 15px;
    border-radius: 10px;
    border-left: 6px solid #003B88;
    margin-bottom: 15px;
}

/* Sidebar Style */
section[data-testid="stSidebar"] {
    background-color: #003B88 !important;
}
.sidebar-text {
    color: white !important;
    font-size: 16px !important;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab"] {
    font-size: 16px;
    font-weight: 600;
    color: #003B88;
}

/* Headings */
h2, h3, h4 {
    color: #003B88 !important;
}

/* Make layout nicer on mobile */
@media (max-width: 768px) {
    .stImage img {
        width: 130px !important;  /* smaller logo on phone */
    }

    .banner h1 {
        font-size: 22px;
    }

    .prototype-tag {
        font-size: 14px;
    }
}

</style>
"""

st.markdown(BNM_STYLE, unsafe_allow_html=True)

# ---------- LOGO ABOVE BANNER ----------
slogo_col1, logo_col2, logo_col3 = st.columns([1, 2, 1])
with logo_col2:
    st.image("bnm_logo.png", width=200)

# ---------- HEADER ----------
st.markdown("""
<div class="banner">
    <h1>Bank Negara Malaysia — Conduct Risk Dashboard</h1>
    <div class="prototype-tag">(Prototype)</div>
    <p>Monitoring FTFC, Prohibited Conduct & Complaints Handling Across Financial Institutions</p>
</div>
""", unsafe_allow_html=True)

# ---------- DATA LOADING ----------
@st.cache_data
def load_data():
    df_ftfc = pd.read_csv("complaints_dummy.csv", parse_dates=["date"])
    df_pbc = pd.read_csv("complaints_pbc.csv", parse_dates=["date"])
    df_handling = pd.read_csv("complaints_handling.csv", parse_dates=["date"])
    bank_customers = pd.read_csv("bank_customers.csv")

    for df in [df_ftfc, df_pbc, df_handling]:
        df["month"] = df["date"].dt.to_period("M").astype(str)

    return df_ftfc, df_pbc, df_handling, bank_customers

df_ftfc, df_pbc, df_handling, bank_customers = load_data()
ALL_BANKS = sorted(df_ftfc["bank_name"].unique())

PBC_COLS = ["pbc_hidden_fee", "pbc_misleading_statement", "pbc_harassment", "pbc_unauthorised_txn"]


# ---------- FAIRNESS FUNCTIONS ----------
def compute_fairness_metrics(df, bank_name=None):
    if bank_name and bank_name != "All banks":
        data = df[df["bank_name"] == bank_name].copy()
    else:
        data = df.copy()

    total_complaints = len(data)
    if total_complaints == 0:
        return None

    metrics = {
        "Transparency": data["issue_transparency"].sum() / total_complaints * 100,
        "Suitability": data["issue_suitability"].sum() / total_complaints * 100,
        "Misleading Sales": data["issue_misleading"].sum() / total_complaints * 100,
        "Unfair Fees": data["issue_unfair_fee"].sum() / total_complaints * 100,
        "Vulnerable Consumers": data["issue_vulnerable"].sum() / total_complaints * 100,
    }
    return metrics


def make_fairness_radar(metrics, title="Fairness Indicators"):
    labels = list(metrics.keys())
    values = list(metrics.values())

    values_closed = values + values[:1]
    angles = np.linspace(0, 2 * np.pi, len(values_closed))

    fig, ax = plt.subplots(subplot_kw={"polar": True})
    ax.plot(angles, values_closed, linewidth=2)
    ax.fill(angles, values_closed, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylim(0, 100)

    return fig


def make_fairness_trend(df, bank_name=None):
    if bank_name and bank_name != "All banks":
        data = df[df["bank_name"] == bank_name].copy()
    else:
        data = df.copy()

    fairness_cols = [
        "issue_transparency",
        "issue_suitability",
        "issue_misleading",
        "issue_unfair_fee",
        "issue_vulnerable",
    ]
    data["any_fairness_issue"] = data[fairness_cols].sum(axis=1) > 0

    grouped = (
        data.groupby("month")
        .agg(
            total_complaints=("complaint_id", "count"),
            fairness_complaints=("any_fairness_issue", "sum"),
        )
        .reset_index()
    )
    if len(grouped) == 0:
        return None

    grouped["fairness_pct"] = (
        grouped["fairness_complaints"] / grouped["total_complaints"] * 100
    )

    fig, ax = plt.subplots()
    x = np.arange(len(grouped["month"]))
    ax.plot(x, grouped["fairness_pct"], marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["month"], rotation=45)
    ax.set_ylabel("% complaints with fairness issues")
    ax.set_title(f"Fairness Trend – {bank_name if bank_name else 'All banks'}")
    fig.tight_layout()
    return fig


# ---------- PBC FUNCTIONS ----------
def compute_pbc_counts(df, bank_name=None):
    if bank_name and bank_name != "All banks":
        data = df[df["bank_name"] == bank_name].copy()
    else:
        data = df.copy()

    counts = {col: int(data[col].sum()) for col in PBC_COLS}
    data["any_pbc"] = data[PBC_COLS].sum(axis=1) > 0
    total_pbc_cases = int(data["any_pbc"].sum())
    return counts, total_pbc_cases


def make_pbc_heatmap(df):
    banks = sorted(df["bank_name"].unique())
    matrix = []
    for bank in banks:
        data_bank = df[df["bank_name"] == bank]
        row = [data_bank[col].sum() for col in PBC_COLS]
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, aspect="auto")

    ax.set_xticks(np.arange(len(PBC_COLS)))
    ax.set_yticks(np.arange(len(banks)))
    ax.set_xticklabels(PBC_COLS, rotation=45, ha="right")
    ax.set_yticklabels(banks)

    for i in range(len(banks)):
        for j in range(len(PBC_COLS)):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")

    ax.set_title("Prohibited Conduct by Bank")
    fig.colorbar(im, ax=ax, label="Number of complaints")
    fig.tight_layout()
    return fig


def make_pbc_bar_for_bank(df, bank_name=None):
    counts, total_pbc = compute_pbc_counts(df, bank_name)
    labels = list(counts.keys())
    values = list(counts.values())

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Number of complaints")
    ax.set_title(
        f"PBC Categories – {bank_name if bank_name and bank_name != 'All banks' else 'All banks'}"
    )
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


# ---------- COMPLAINTS HANDLING FUNCTIONS ----------
def compute_handling_metrics(df, bank_name=None):
    if bank_name and bank_name != "All banks":
        data = df[df["bank_name"] == bank_name].copy()
    else:
        data = df.copy()

    total = len(data)
    if total == 0:
        return None

    metrics = {
        "avg_days_to_resolve": data["days_to_resolve"].mean(),
        "pct_resolved_within_timeline": data["resolved_within_timeline"].mean() * 100,
        "pct_reopened": data["reopened_case"].mean() * 100,
        "pct_escalated": data["escalated_case"].mean() * 100,
        "pct_repeat_same_issue": data["repeat_same_issue"].mean() * 100,
        "total_complaints": total,
    }
    return metrics


def compute_complaints_per_1000(df, bank_customers):
    complaints_count = (
        df.groupby("bank_name")["complaint_id"]
        .count()
        .reset_index()
        .rename(columns={"complaint_id": "complaints"})
    )
    merged = complaints_count.merge(bank_customers, on="bank_name", how="left")
    merged["complaints_per_1000"] = (
        merged["complaints"] / merged["customers"] * 1000
    )
    return merged


def make_resolution_timeline(df, bank_name=None):
    if bank_name and bank_name != "All banks":
        data = df[df["bank_name"] == bank_name].copy()
    else:
        data = df.copy()

    grouped = (
        data.groupby("month")["days_to_resolve"]
        .mean()
        .reset_index()
        .rename(columns={"days_to_resolve": "avg_days_to_resolve"})
    )
    if len(grouped) == 0:
        return None

    fig, ax = plt.subplots()
    x = np.arange(len(grouped["month"]))
    ax.plot(x, grouped["avg_days_to_resolve"], marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["month"], rotation=45)
    ax.set_ylabel("Average days to resolve")
    ax.set_title(
        f"Resolution Time Trend – {bank_name if bank_name else 'All banks'}"
    )
    fig.tight_layout()
    return fig


def make_recurrence_trend(df, bank_name=None):
    if bank_name and bank_name != "All banks":
        data = df[df["bank_name"] == bank_name].copy()
    else:
        data = df.copy()

    grouped = (
        data.groupby("month")
        .agg(
            total=("complaint_id", "count"),
            repeat=("repeat_same_issue", "sum"),
        )
        .reset_index()
    )
    if len(grouped) == 0:
        return None

    grouped["repeat_pct"] = grouped["repeat"] / grouped["total"] * 100

    fig, ax = plt.subplots()
    x = np.arange(len(grouped["month"]))
    ax.plot(x, grouped["repeat_pct"], marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["month"], rotation=45)
    ax.set_ylabel("% repeat complaints (same issue)")
    ax.set_title(
        f"Recurrence Trend – {bank_name if bank_name else 'All banks'}"
    )
    fig.tight_layout()
    return fig


# ---------- OVERALL RISK SCORE ----------
def build_risk_table():
    risk_rows = []
    for bank in ALL_BANKS:

        ftfc = compute_fairness_metrics(df_ftfc, bank)
        ftfc_score = np.mean(list(ftfc.values())) if ftfc else 0

        _, total_pbc = compute_pbc_counts(df_pbc, bank)
        pbc_score = total_pbc

        h = compute_handling_metrics(df_handling, bank)
        if h:
            handling_score = (
                h["avg_days_to_resolve"]
                + (100 - h["pct_resolved_within_timeline"])
                + h["pct_reopened"]
                + h["pct_escalated"]
                + h["pct_repeat_same_issue"]
            )
        else:
            handling_score = 0

        risk_rows.append(
            {
                "bank": bank,
                "ftfc_score_raw": ftfc_score,
                "pbc_score_raw": pbc_score,
                "handling_score_raw": handling_score,
            }
        )

    df_risk = pd.DataFrame(risk_rows)

    for col in ["ftfc_score_raw", "pbc_score_raw", "handling_score_raw"]:
        max_val = df_risk[col].max()
        norm_col = col.replace("_raw", "_norm")
        if max_val == 0:
            df_risk[norm_col] = 0
        else:
            df_risk[norm_col] = df_risk[col] / max_val

    df_risk["final_score"] = (
        0.30 * df_risk["ftfc_score_norm"]
        + 0.40 * df_risk["pbc_score_norm"]
        + 0.30 * df_risk["handling_score_norm"]
    )

    def classify(score):
        if score < 0.33:
            return "Low"
        elif score < 0.66:
            return "Medium"
        else:
            return "High"

    df_risk["risk_category"] = df_risk["final_score"].apply(classify)
    return df_risk


def make_risk_heatmap(df_risk):
    banks = df_risk["bank"].tolist()
    scores = df_risk["final_score"].to_numpy()

    data = scores.reshape(1, -1)

    fig, ax = plt.subplots()
    im = ax.imshow(
        data,
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="RdYlGn_r",
    )

    ax.set_yticks([0])
    ax.set_yticklabels(["Overall risk"])

    ax.set_xticks(np.arange(len(banks)))
    ax.set_xticklabels(banks, rotation=45, ha="right")

    for j, v in enumerate(scores):
        ax.text(j, 0, f"{v:.2f}", ha="center", va="center", color="black")

    ax.set_title("Overall Conduct Risk Heatmap")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Conduct risk score")

    fig.tight_layout()
    return fig


df_risk = build_risk_table()


# ---------- SIDEBAR BRANDING ----------
st.sidebar.markdown(
    """
    <h2 class="sidebar-text">BNM Conduct Analytics</h2>
    <p class="sidebar-text">Select a bank to drill down into indicators.</p>
    """,
    unsafe_allow_html=True
)

# ---------- BANK SELECTOR ----------
bank_choice = st.sidebar.selectbox(
    "Select bank",
    ["All banks"] + ALL_BANKS
)

# ---------- KPI CARDS (BNM STYLE) ----------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-container">
            <h3>{len(ALL_BANKS)}</h3>
            <p>Total Banks</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-container">
            <h3>{(df_risk['risk_category'] == 'High').sum()}</h3>
            <p>Banks in High Risk</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-container">
            <h3>{df_risk['final_score'].mean():.2f}</h3>
            <p>Industry Avg Risk</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    highest = df_risk.loc[df_risk["final_score"].idxmax()]
    st.markdown(
        f"""
        <div class="metric-container">
            <h3>{highest['bank']}</h3>
            <p>Highest Risk ({highest['final_score']:.2f})</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- METHODOLOGY (SUMMARY) ----------
with st.expander("Methodology & Interpretation of Scores"):
    st.markdown(
        """
### **1. Scope of Prototype**
This dashboard uses synthetic data to illustrate how market conduct risk indicators  
could be monitored across:

- Fair Treatment to Financial Consumers (**FTFC**)  
- **Prohibited Business Conduct (PBC)**  
- **Complaints Handling** performance  

It is intended as a *conceptual prototype* for analytics exploration.

---

### **2. Scoring Approach**

#### **FTFC Score**
Measures the share of complaints involving:
- Transparency issues  
- Suitability concerns  
- Misleading sales  
- Unfair fees  
- Vulnerable consumers  

FTFC Score = *Average % of complaints involving fairness-related issues*

---

#### **PBC Score**
Counts frequency of key prohibited conduct categories:
- Hidden or excessive fees  
- Misleading statements  
- Harassment  
- Unauthorised transactions  

Higher frequency → Higher risk.

---

#### **Complaints Handling Score**
Captures the bank’s behaviour during dispute resolution:
- Average days to resolve  
- % resolved within timeline  
- % reopened  
- % escalated  
- % repeat issues  

Longer resolution times and higher recurrence → Higher risk.

---

### **3. Overall Conduct Risk Score**

Final Score =  
- **30%** FTFC  
- **40%** PBC  
- **30%** Complaints Handling  

Risk Category Thresholds:
- **Low Risk**: < 0.33  
- **Medium Risk**: 0.33 – 0.66  
- **High Risk**: ≥ 0.66  

---

### **4. Important Disclaimer**
This dashboard is a **prototype for analytical exploration only**.  
It:

- does **not** represent actual BNM supervisory assessments,  
- does **not** reflect real bank performance,  
- should not be interpreted as an official BNM view or position.  

The purpose is strictly to test and demonstrate analytical concepts  
for market conduct supervision.
"""
    )


# ---------- TABS ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Overall Risk", "Fairness (FTFC)", "Prohibited Conduct", "Complaints Handling"]
)

# --- Tab 1: Overall Risk ---
with tab1:
    st.subheader("Overall Conduct Risk")

    st.write("Risk scoring summary (all banks):")
    st.dataframe(df_risk)

    st.write("Overall risk heatmap (green = low, red = high):")
    st.pyplot(make_risk_heatmap(df_risk))

    if bank_choice != "All banks":
        st.markdown("---")
        st.write(f"Currently selected bank: **{bank_choice}**")
        st.write(
            df_risk[df_risk["bank"] == bank_choice][
                ["bank", "final_score", "risk_category"]
            ]
        )

# --- Tab 2: FTFC ---
with tab2:
    st.subheader("Fairness Indicators (FTFC)")

    metrics = compute_fairness_metrics(df_ftfc, bank_choice)
    if not metrics:
        st.write("No data available.")
    else:
        st.write("Fairness indicators (% of complaints):")
        st.dataframe(pd.DataFrame(metrics, index=["%"]))

        st.pyplot(make_fairness_radar(metrics, f"FTFC Radar – {bank_choice}"))
        trend = make_fairness_trend(df_ftfc, bank_choice)
        if trend:
            st.pyplot(trend)

# --- Tab 3: PBC ---
with tab3:
    st.subheader("Prohibited Business Conduct")

    counts, total_pbc = compute_pbc_counts(df_pbc, bank_choice)
    st.write(f"Total PBC complaints: **{total_pbc}**")
    st.dataframe(pd.DataFrame.from_dict(counts, orient="index", columns=["count"]))

    st.pyplot(make_pbc_bar_for_bank(df_pbc, bank_choice))

    st.write("Industry heatmap:")
    st.pyplot(make_pbc_heatmap(df_pbc))

# --- Tab 4: Handling ---
with tab4:
    st.subheader("Complaints Handling")

    handling = compute_handling_metrics(df_handling, bank_choice)
    if not handling:
        st.write("No data.")
    else:
        st.dataframe(pd.DataFrame(handling, index=["value"]).T)

        cp1000 = compute_complaints_per_1000(df_handling, bank_customers)
        st.write("Complaints per 1,000 customers (all banks):")
        st.dataframe(cp1000)

        st.pyplot(make_resolution_timeline(df_handling, bank_choice))
        st.pyplot(make_recurrence_trend(df_handling, bank_choice))
