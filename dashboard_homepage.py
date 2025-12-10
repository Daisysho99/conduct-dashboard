import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---------- PAGE CONFIG ----------

st.set_page_config(
    page_title="Conduct Risk Dashboard (Prototype)",
    page_icon="🏦",
    layout="wide",
)


# ---------- DATA LOADING ----------

@st.cache_data
def load_data():
    df_ftfc = pd.read_csv("complaints_dummy.csv", parse_dates=["date"])
    df_pbc = pd.read_csv("complaints_pbc.csv", parse_dates=["date"])
    df_handling = pd.read_csv("complaints_handling.csv", parse_dates=["date"])
    bank_customers = pd.read_csv("bank_customers.csv")

    # Add month column for all
    for df in [df_ftfc, df_pbc, df_handling]:
        df["month"] = df["date"].dt.to_period("M").astype(str)

    return df_ftfc, df_pbc, df_handling, bank_customers


df_ftfc, df_pbc, df_handling, bank_customers = load_data()
ALL_BANKS = sorted(df_ftfc["bank_name"].unique())

PBC_COLS = [
    "pbc_hidden_fee",
    "pbc_misleading_statement",
    "pbc_harassment",
    "pbc_unauthorised_txn",
]


# ---------- FAIRNESS (FTFC) FUNCTIONS ----------

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

    # close loop
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


# ---------- OVERALL RISK SCORE FUNCTIONS ----------

def build_risk_table():
    risk_rows = []
    for bank in ALL_BANKS:
        # FTFC
        ftfc = compute_fairness_metrics(df_ftfc, bank)
        ftfc_score = np.mean(list(ftfc.values())) if ftfc else 0

        # PBC
        _, total_pbc = compute_pbc_counts(df_pbc, bank)
        pbc_score = total_pbc

        # Handling
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

    # normalise 0–1
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
        cmap="RdYlGn_r",  # green low, red high
    )

    ax.set_yticks([0])
    ax.set_yticklabels(["Overall risk"])

    ax.set_xticks(np.arange(len(banks)))
    ax.set_xticklabels(banks, rotation=45, ha="right")

    for j, v in enumerate(scores):
        ax.text(j, 0, f"{v:.2f}", ha="center", va="center", color="black")

    ax.set_title("Overall Conduct Risk Heatmap (0 = low, 1 = high)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Conduct risk score")

    fig.tight_layout()
    return fig


df_risk = build_risk_table()


# ---------- HOMEPAGE / HEADER ----------

st.title("🏦 Conduct Risk Dashboard (Prototype)")
st.markdown(
    """
This prototype dashboard illustrates **market conduct risk indicators** across banks,  
combining **FTFC**, **Prohibited Business Conduct (PBC)**, and **Complaints Handling** metrics.
"""
)

# Top KPI cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Number of banks", len(ALL_BANKS))
with col2:
    st.metric("Banks with HIGH risk", int((df_risk["risk_category"] == "High").sum()))
with col3:
    st.metric(
        "Industry average risk score",
        f"{df_risk['final_score'].mean():.2f}",
    )
with col4:
    highest_row = df_risk.loc[df_risk["final_score"].idxmax()]
    st.metric("Highest risk bank", highest_row["bank"], f"{highest_row['final_score']:.2f}")

with st.expander("Methodology (summary)"):
    st.markdown(
        """
**Scope**

- Prototype using synthetic data for:
  - Fair Treatment to Financial Consumers (FTFC)
  - Prohibited Business Conduct (PBC)
  - Complaints Handling performance

**Scoring approach**

- FTFC score: average share of complaints with fairness-related issues  
- PBC score: frequency of complaints with prohibited conduct flags  
- Complaints handling score: based on resolution time, timeliness, reopen/escalation rates and recurrence  
- Overall conduct risk score combines:
  - 30% FTFC
  - 40% PBC
  - 30% Complaints handling

Risk categories:
- **Low**: score < 0.33  
- **Medium**: 0.33 ≤ score < 0.66  
- **High**: score ≥ 0.66  
"""
    )


# ---------- STREAMLIT APP LAYOUT (TABS) ----------

# Bank selector (drill-down)
bank_choice = st.sidebar.selectbox(
    "Select bank to drill down",
    ["All banks"] + ALL_BANKS
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Fairness (FTFC)", "Prohibited Conduct (PBC)", "Complaints Handling", "Overall Risk"]
)

# --- Tab 1: Fairness ---
with tab1:
    st.subheader("Fair Treatment to Financial Consumers (FTFC)")

    metrics = compute_fairness_metrics(df_ftfc, bank_choice)
    if not metrics:
        st.write("No complaints for this selection.")
    else:
        st.write("Fairness indicators (% of complaints):")
        st.dataframe(pd.DataFrame(metrics, index=["%"]))

        radar_fig = make_fairness_radar(
            metrics, title=f"Fairness Radar – {bank_choice}"
        )
        st.pyplot(radar_fig)

        trend_fig = make_fairness_trend(df_ftfc, bank_choice)
        if trend_fig:
            st.pyplot(trend_fig)

# --- Tab 2: PBC ---
with tab2:
    st.subheader("Prohibited Business Conduct (PBC)")

    counts, total_pbc = compute_pbc_counts(df_pbc, bank_choice)
    st.write(f"Total PBC-related complaints: **{total_pbc}**")
    st.write("Breakdown by PBC category:")
    st.dataframe(pd.DataFrame.from_dict(counts, orient="index", columns=["count"]))

    pbc_bar_fig = make_pbc_bar_for_bank(df_pbc, bank_choice)
    st.pyplot(pbc_bar_fig)

    st.write("Industry view (all banks) – PBC heatmap:")
    pbc_heatmap_fig = make_pbc_heatmap(df_pbc)
    st.pyplot(pbc_heatmap_fig)

# --- Tab 3: Complaints Handling ---
with tab3:
    st.subheader("Complaints Handling Quality")

    handling_metrics = compute_handling_metrics(df_handling, bank_choice)
    if not handling_metrics:
        st.write("No complaints for this selection.")
    else:
        st.write("Key KPIs:")
        st.dataframe(
            pd.DataFrame(handling_metrics, index=["value"]).T.rename(
                columns={"value": bank_choice}
            )
        )

        cp1000 = compute_complaints_per_1000(df_handling, bank_customers)
        st.write("Complaints per 1,000 customers (all banks):")
        st.dataframe(cp1000)

        res_fig = make_resolution_timeline(df_handling, bank_choice)
        if res_fig:
            st.pyplot(res_fig)

        rec_fig = make_recurrence_trend(df_handling, bank_choice)
        if rec_fig:
            st.pyplot(rec_fig)

# --- Tab 4: Overall Risk ---
with tab4:
    st.subheader("Overall Conduct Risk Score")

    st.write("Risk scoring summary (all banks):")
    st.dataframe(df_risk)

    st.write("Overall risk heatmap (green = low, red = high):")
    heatmap_fig = make_risk_heatmap(df_risk)
    st.pyplot(heatmap_fig)

    st.markdown("---")
    st.write(
        "Use the bank selector on the left to drill down into a specific bank, "
        "then view details in the other tabs (FTFC, PBC, Complaints Handling)."
    )

    if bank_choice != "All banks":
        st.write(f"Currently selected bank: **{bank_choice}**")
        st.write(
            df_risk[df_risk["bank"] == bank_choice][
                ["bank", "final_score", "risk_category"]
            ]
        )
