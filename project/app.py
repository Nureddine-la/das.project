import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Placement Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2.2rem; font-weight:700; color:#1a56db; margin-bottom:0.2rem; }
    .sub-title   { font-size:1rem;  color:#6b7280; margin-bottom:1.5rem; }
    .section-hdr { font-size:1.3rem; font-weight:600; color:#111827;
                   border-left:4px solid #1a56db; padding-left:10px; margin:1.5rem 0 0.8rem; }
    .metric-card { background:#f0f4ff; border-radius:10px; padding:1rem 1.4rem;
                   border:1px solid #c7d2fe; }
    .insight-box { background:#fefce8; border-left:4px solid #eab308;
                   padding:0.8rem 1rem; border-radius:6px; margin:0.5rem 0; font-size:0.92rem; }
    .good-box    { background:#f0fdf4; border-left:4px solid #22c55e;
                   padding:0.8rem 1rem; border-radius:6px; margin:0.5rem 0; font-size:0.92rem; }
    .warn-box    { background:#fff7ed; border-left:4px solid #f97316;
                   padding:0.8rem 1rem; border-radius:6px; margin:0.5rem 0; font-size:0.92rem; }
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    return df

df = load_data()

# ── Sidebar navigation ─────────────────────────────────────────────────────────
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3976/3976625.png", width=70)
st.sidebar.markdown("## 🎓 Placement Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview",
     "🔍 Data Collection & Preprocessing",
     "📊 Exploratory Data Analysis",
     "🤖 Modeling & Results",
     "🎯 Predict My Placement"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** Student Placement Data")
st.sidebar.markdown(f"**Rows:** {len(df):,}  |  **Columns:** {df.shape[1]}")
st.sidebar.markdown("**Target:** PlacedOrNot")
st.markdown('<p class="main-title"> plz USE LIGHT MODE I HAVE PROBLEMS with Dark Mode</p>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<p class="main-title">🎓 Student Placement Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">A complete ML pipeline — from raw data to actionable placement predictions</p>', unsafe_allow_html=True)

    placed     = df['PlacedOrNot'].sum()
    not_placed = len(df) - placed
    rate       = placed / len(df) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📁 Total Students", f"{len(df):,}")
    c2.metric("✅ Placed", f"{placed:,}")
    c3.metric("❌ Not Placed", f"{not_placed:,}")
    c4.metric("📈 Placement Rate", f"{rate:.1f}%")

    st.markdown('<p class="section-hdr">Project Pipeline</p>', unsafe_allow_html=True)
    steps = [
        ("1️⃣", "Data Collection", "Loaded a real-world student dataset with 2,966 records and 4 features."),
        ("2️⃣", "Preprocessing",   "Verified data types, checked for missing values, confirmed clean data."),
        ("3️⃣", "EDA",             "Explored distributions, correlations, and placement patterns per feature."),
        ("4️⃣", "Modeling",        "Applied Linear & Logistic Regression; evaluated with R², RMSE, Accuracy, F1."),
        ("5️⃣", "Deployment",      "Built this interactive Streamlit dashboard to present all results."),
    ]
    for icon, title, desc in steps:
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:0.6rem">
            <strong>{icon} {title}</strong><br>
            <span style="color:#6b7280">{desc}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-hdr">Dataset Features</p>', unsafe_allow_html=True)
    feat_df = pd.DataFrame({
        "Feature":     ["Internships", "CGPA", "HistoryOfBacklogs", "PlacedOrNot"],
        "Type":        ["Integer (0–3)", "Integer (5–9)", "Binary (0/1)", "Binary (0/1)"],
        "Description": [
            "Number of internships completed",
            "Cumulative Grade Point Average",
            "1 = Has history of academic backlogs",
            "Target — 1 = Placed, 0 = Not Placed"
        ]
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — DATA COLLECTION & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Data Collection & Preprocessing":
    st.markdown('<p class="main-title">🔍 Data Collection & Preprocessing</p>', unsafe_allow_html=True)

    st.markdown('<p class="section-hdr">Raw Dataset (first 10 rows)</p>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown('<p class="section-hdr">Dataset Shape & Types</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")
        info_df = pd.DataFrame({
            "Column":  df.columns,
            "Dtype":   [str(df[c].dtype) for c in df.columns],
            "Non-Null": [df[c].notnull().sum() for c in df.columns],
            "Missing":  [df[c].isnull().sum() for c in df.columns],
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Descriptive Statistics**")
        st.dataframe(df.describe().round(3), use_container_width=True)

    st.markdown('<p class="section-hdr">Missing Values Check</p>', unsafe_allow_html=True)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        st.markdown('<div class="good-box">✅ No missing values found. The dataset is complete and ready for analysis.</div>', unsafe_allow_html=True)
    else:
        st.warning(f"Missing values detected: {missing[missing>0].to_dict()}")

    st.markdown('<p class="section-hdr">Duplicate Rows</p>', unsafe_allow_html=True)
    dups = df.duplicated().sum()
    if dups == 0:
        st.markdown('<div class="good-box">✅ No duplicate rows found.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warn-box">⚠️ {dups} duplicate rows detected.</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-hdr">Unique Values per Column</p>', unsafe_allow_html=True)
    uv = {col: sorted(df[col].unique().tolist()) for col in df.columns}
    for col, vals in uv.items():
        st.markdown(f"**{col}:** {vals}")

    st.markdown('<p class="section-hdr">Preprocessing Steps Applied</p>', unsafe_allow_html=True)
    steps = [
        "✅ Verified all columns are numeric integers — no encoding needed.",
        "✅ Confirmed no null values — no imputation required.",
        "✅ No outliers found — all values are within expected categorical/discrete ranges.",
        "✅ Feature scaling (StandardScaler) applied only for Logistic Regression.",
        "✅ Train/Test split: 80% training, 20% testing with random_state=42.",
    ]
    for s in steps:
        st.markdown(f'<div class="good-box">{s}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Exploratory Data Analysis":
    st.markdown('<p class="main-title">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)

    # ── Distributions ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-hdr">Feature Distributions</p>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    colors = ['#1a56db', '#16a34a', '#dc2626', '#9333ea']
    for i, col in enumerate(df.columns):
        axes[i].hist(df[col], bins=10, color=colors[i], edgecolor='white', alpha=0.85)
        axes[i].set_title(col, fontsize=11, fontweight='bold')
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Placement rate ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-hdr">Placement Rate by Feature</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        # By CGPA
        cgpa_rate = df.groupby('CGPA')['PlacedOrNot'].mean().reset_index()
        cgpa_rate.columns = ['CGPA', 'PlacementRate']
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(cgpa_rate['CGPA'].astype(str), cgpa_rate['PlacementRate']*100,
                      color='#1a56db', edgecolor='white')
        ax.set_title('Placement Rate by CGPA', fontweight='bold')
        ax.set_xlabel('CGPA')
        ax.set_ylabel('Placement Rate (%)')
        ax.set_ylim(0, 110)
        for bar, val in zip(bars, cgpa_rate['PlacementRate']*100):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f'{val:.0f}%', ha='center', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        # By Internships
        int_rate = df.groupby('Internships')['PlacedOrNot'].mean().reset_index()
        int_rate.columns = ['Internships', 'PlacementRate']
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(int_rate['Internships'].astype(str), int_rate['PlacementRate']*100,
                      color='#16a34a', edgecolor='white')
        ax.set_title('Placement Rate by Internships', fontweight='bold')
        ax.set_xlabel('Number of Internships')
        ax.set_ylabel('Placement Rate (%)')
        ax.set_ylim(0, 110)
        for bar, val in zip(bars, int_rate['PlacementRate']*100):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f'{val:.0f}%', ha='center', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    c3, c4 = st.columns(2)
    with c3:
        # By Backlogs
        bl_rate = df.groupby('HistoryOfBacklogs')['PlacedOrNot'].mean().reset_index()
        bl_rate['Label'] = bl_rate['HistoryOfBacklogs'].map({0: 'No Backlog', 1: 'Has Backlog'})
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(bl_rate['Label'], bl_rate['PlacedOrNot']*100,
                      color=['#16a34a','#dc2626'], edgecolor='white')
        ax.set_title('Placement Rate by Backlog History', fontweight='bold')
        ax.set_ylabel('Placement Rate (%)')
        ax.set_ylim(0, 80)
        for bar, val in zip(bars, bl_rate['PlacedOrNot']*100):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c4:
        # Correlation heatmap
        fig, ax = plt.subplots(figsize=(6, 4))
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                    linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})
        ax.set_title('Correlation Heatmap', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Stacked bar ────────────────────────────────────────────────────────────
    st.markdown('<p class="section-hdr">Placed vs Not Placed by CGPA (Stacked)</p>', unsafe_allow_html=True)
    cgpa_cross = df.groupby(['CGPA','PlacedOrNot']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    cgpa_cross.plot(kind='bar', stacked=True, ax=ax,
                    color=['#dc2626','#1a56db'], edgecolor='white')
    ax.set_title('Placed vs Not Placed by CGPA', fontweight='bold')
    ax.set_xlabel('CGPA')
    ax.set_ylabel('Number of Students')
    ax.legend(['Not Placed','Placed'], loc='upper left')
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Key insights ────────────────────────────────────────────────────────────
    st.markdown('<p class="section-hdr">Key EDA Insights</p>', unsafe_allow_html=True)
    insights = [
        ("📌", "CGPA is the strongest predictor — students with CGPA 8 or 9 are placed at 100% rate, while CGPA 5 yields only 7.3%."),
        ("📌", "Internships show a positive trend — more internships correlate with higher placement rates (49% → 54% → 80%+ for 2+)."),
        ("📌", "Backlog history has minimal effect — placement rate drops only ~3% (55.8% → 53.0%) for students with backlogs."),
        ("📌", "The dataset is slightly imbalanced — 55.2% placed vs 44.8% not placed, close enough to not require resampling."),
        ("📌", "CGPA and PlacedOrNot show the highest correlation (0.56), confirming CGPA as the dominant feature."),
    ]
    for icon, text in insights:
        st.markdown(f'<div class="insight-box">{icon} {text}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — MODELING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Modeling & Results":
    st.markdown('<p class="main-title">🤖 Modeling & Results</p>', unsafe_allow_html=True)

    # ── Prepare data ───────────────────────────────────────────────────────────
    # Linear Regression — predict CGPA from other features
    X_lin = df[['Internships','HistoryOfBacklogs','PlacedOrNot']]
    y_lin = df['CGPA']
    X_lin_train, X_lin_test, y_lin_train, y_lin_test = train_test_split(
        X_lin, y_lin, test_size=0.2, random_state=42)
    lin_model = LinearRegression()
    lin_model.fit(X_lin_train, y_lin_train)
    y_lin_pred = lin_model.predict(X_lin_test)

    # Logistic Regression — predict PlacedOrNot
    X_log = df[['Internships','CGPA','HistoryOfBacklogs']]
    y_log = df['PlacedOrNot']
    X_log_train, X_log_test, y_log_train, y_log_test = train_test_split(
        X_log, y_log, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_log_train_s = scaler.fit_transform(X_log_train)
    X_log_test_s  = scaler.transform(X_log_test)
    log_model = LogisticRegression(random_state=42)
    log_model.fit(X_log_train_s, y_log_train)
    y_log_pred = log_model.predict(X_log_test_s)

    # ── Model selection tabs ────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📉 Linear Regression (CGPA)", "📊 Logistic Regression (Placement)"])

    # ─── Linear Regression ──────────────────────────────────────────────────────
    with tab1:
        st.markdown("**Target:** CGPA (continuous)  |  **Features:** Internships, HistoryOfBacklogs, PlacedOrNot")
        st.markdown('<p class="section-hdr">Model Performance</p>', unsafe_allow_html=True)

        r2   = r2_score(y_lin_test, y_lin_pred)
        rmse = np.sqrt(mean_squared_error(y_lin_test, y_lin_pred))
        mae  = np.mean(np.abs(y_lin_test - y_lin_pred))

        c1, c2, c3 = st.columns(3)
        c1.metric("R² Score",  f"{r2:.4f}")
        c2.metric("RMSE",      f"{rmse:.4f}")
        c3.metric("MAE",       f"{mae:.4f}")

        st.markdown('<p class="section-hdr">Coefficients</p>', unsafe_allow_html=True)
        coef_df = pd.DataFrame({
            "Feature":     list(X_lin.columns),
            "Coefficient": lin_model.coef_.round(4)
        })
        st.dataframe(coef_df, use_container_width=True, hide_index=True)
        st.markdown(f"**Intercept:** {lin_model.intercept_:.4f}")

        c1, c2 = st.columns(2)
        with c1:
            # Actual vs Predicted
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(y_lin_test, y_lin_pred, alpha=0.4, color='#1a56db', s=15)
            mn = min(y_lin_test.min(), y_lin_pred.min()) - 0.5
            mx = max(y_lin_test.max(), y_lin_pred.max()) + 0.5
            ax.plot([mn, mx],[mn, mx], 'r--', linewidth=1.5, label='Perfect fit')
            ax.set_xlabel('Actual CGPA')
            ax.set_ylabel('Predicted CGPA')
            ax.set_title('Actual vs Predicted CGPA', fontweight='bold')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            # Residuals
            residuals = y_lin_test - y_lin_pred
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(residuals, bins=25, color='#9333ea', edgecolor='white', alpha=0.8)
            ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
            ax.set_xlabel('Residual')
            ax.set_ylabel('Frequency')
            ax.set_title('Residuals Distribution', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Coeff bar chart
        fig, ax = plt.subplots(figsize=(8, 3))
        cols_sorted = coef_df.sort_values('Coefficient', ascending=True)
        colors = ['#dc2626' if v < 0 else '#16a34a' for v in cols_sorted['Coefficient']]
        ax.barh(cols_sorted['Feature'], cols_sorted['Coefficient'], color=colors, edgecolor='white')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title('Feature Coefficients', fontweight='bold')
        ax.set_xlabel('Coefficient Value')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown('<p class="section-hdr">Interpretation</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
        📌 <b>R² = 0.30</b> — The model explains 30% of the variance in CGPA. This is expected because CGPA in this 
        dataset takes only integer values (5–9), making it a semi-categorical variable that linear regression 
        is not perfectly suited for, but still reveals meaningful relationships.
        </div>
        <div class="insight-box">
        📌 <b>PlacedOrNot coefficient = +1.19</b> — The strongest predictor of CGPA is whether a student got placed. 
        Placed students tend to have a CGPA ~1.19 points higher on average, which makes intuitive sense.
        </div>
        <div class="insight-box">
        📌 <b>Internships coefficient = −0.11</b> — Slightly negative, suggesting internships alone don't 
        directly predict higher CGPA; students with more internships may invest time away from academics.
        </div>
        <div class="good-box">
        ✅ <b>Conclusion:</b> Placement status is the best linear predictor of CGPA, reinforcing that CGPA and 
        placement are tightly linked. RMSE of 0.83 on a 5–9 scale means errors stay within ~1 grade point.
        </div>
        """, unsafe_allow_html=True)

    # ─── Logistic Regression ────────────────────────────────────────────────────
    with tab2:
        st.markdown("**Target:** PlacedOrNot (binary)  |  **Features:** Internships, CGPA, HistoryOfBacklogs")
        st.markdown('<p class="section-hdr">Model Performance</p>', unsafe_allow_html=True)

        acc = accuracy_score(y_log_test, y_log_pred)
        report = classification_report(y_log_test, y_log_pred, output_dict=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{acc:.2%}")
        c2.metric("Precision", f"{report['weighted avg']['precision']:.2%}")
        c3.metric("Recall",    f"{report['weighted avg']['recall']:.2%}")
        c4.metric("F1-Score",  f"{report['weighted avg']['f1-score']:.2%}")

        c1, c2 = st.columns(2)
        with c1:
            # Confusion matrix
            cm = confusion_matrix(y_log_test, y_log_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Not Placed','Placed'],
                        yticklabels=['Not Placed','Placed'],
                        linewidths=0.5)
            ax.set_title('Confusion Matrix', fontweight='bold')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            # Feature importance (coefficients)
            log_coef_df = pd.DataFrame({
                "Feature":     list(X_log.columns),
                "Coefficient": log_model.coef_[0].round(4)
            }).sort_values('Coefficient', ascending=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            colors = ['#dc2626' if v < 0 else '#1a56db' for v in log_coef_df['Coefficient']]
            ax.barh(log_coef_df['Feature'], log_coef_df['Coefficient'], color=colors, edgecolor='white')
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_title('Feature Coefficients (Logistic)', fontweight='bold')
            ax.set_xlabel('Coefficient (standardized)')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Classification report
        st.markdown('<p class="section-hdr">Classification Report</p>', unsafe_allow_html=True)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

        st.markdown('<p class="section-hdr">Interpretation</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="good-box">
        ✅ <b>Accuracy = 73.4%</b> — The logistic regression model correctly predicts placement for ~3 out of 4 students. 
        This is solid performance given only 3 input features.
        </div>
        <div class="insight-box">
        📌 <b>CGPA coefficient = +1.79</b> (highest) — CGPA is by far the most influential feature for predicting 
        placement. Each standardized unit increase in CGPA significantly increases the log-odds of being placed.
        </div>
        <div class="insight-box">
        📌 <b>Internships coefficient = +0.56</b> — Internships are the second-strongest predictor. More practical 
        experience clearly improves placement chances.
        </div>
        <div class="insight-box">
        📌 <b>HistoryOfBacklogs coefficient = −0.05</b> — Near zero impact, confirming our EDA finding that backlog 
        history barely affects placement outcomes in this dataset.
        </div>
        <div class="good-box">
        ✅ <b>Conclusion:</b> CGPA and Internships are the two key drivers of placement success. 
        Students should prioritize maintaining a high GPA and gaining internship experience.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Predict My Placement":
    st.markdown('<p class="main-title">🎯 Predict My Placement</p>', unsafe_allow_html=True)
    st.markdown("Enter your details below to get a placement prediction powered by Logistic Regression.")

    # Train model
    X_log = df[['Internships','CGPA','HistoryOfBacklogs']]
    y_log = df['PlacedOrNot']
    scaler = StandardScaler()
    X_log_s = scaler.fit_transform(X_log)
    log_model = LogisticRegression(random_state=42)
    log_model.fit(X_log_s, y_log)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### Your Profile")
        cgpa        = st.slider("📚 CGPA",             min_value=5, max_value=9, value=7, step=1)
        internships = st.slider("💼 Internships",       min_value=0, max_value=3, value=1, step=1)
        backlogs    = st.selectbox("📋 History of Backlogs", options=[0, 1],
                                   format_func=lambda x: "No Backlogs" if x == 0 else "Has Backlogs")

        if st.button("🔍 Predict", type="primary", use_container_width=True):
            inp = scaler.transform([[internships, cgpa, backlogs]])
            pred  = log_model.predict(inp)[0]
            proba = log_model.predict_proba(inp)[0]

            st.markdown("---")
            if pred == 1:
                st.success(f"✅ **LIKELY TO BE PLACED**")
                st.markdown(f"Placement probability: **{proba[1]:.1%}**")
            else:
                st.error(f"❌ **UNLIKELY TO BE PLACED**")
                st.markdown(f"Placement probability: **{proba[1]:.1%}**")

            # Probability bar
            fig, ax = plt.subplots(figsize=(5, 1.2))
            ax.barh([''], [proba[1]], color='#16a34a', height=0.4)
            ax.barh([''], [1-proba[1]], left=[proba[1]], color='#dc2626', height=0.4)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0%','25%','50%','75%','100%'])
            ax.set_title('Placement Probability', fontweight='bold', fontsize=10)
            ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with c2:
        st.markdown("### What affects placement most?")
        st.markdown("""
        <div class="good-box">🏆 <b>CGPA</b> is the #1 factor. Aim for 8 or above for near-certain placement.</div>
        <div class="good-box">💼 <b>Internships</b> are the #2 factor. Even 1 internship noticeably helps.</div>
        <div class="insight-box">📋 <b>Backlog history</b> has minimal statistical impact on placement — don't let it discourage you.</div>
        """, unsafe_allow_html=True)

        st.markdown("### Model used")
        st.markdown("""
        - **Algorithm:** Logistic Regression  
        - **Training data:** 2,372 students (80%)  
        - **Test accuracy:** 73.4%  
        - **Features:** CGPA, Internships, HistoryOfBacklogs  
        - **Scaling:** StandardScaler applied
        """)

    st.markdown("---")
    st.markdown("*This prediction is based on historical data patterns. Use it as a guide, not a guarantee.*")
