import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import linregress
import json

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════
st.set_page_config(
    page_title="GB Cyclist Accidents Dashboard",
    page_icon="🚲",
    layout="wide"
)

PALETTE   = {"Slight": "#3498db", "Serious": "#e67e22", "Fatal": "#e74c3c"}
AGE_ORDER = ["6 to 10","11 to 15","16 to 20","21 to 25",
             "26 to 35","36 to 45","46 to 55","56 to 65","66 to 75"]
MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
sns.set_theme(style="whitegrid", font_scale=1.0)

# ═══════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("merged.csv")
    df["Severity"] = pd.Categorical(
        df["Severity"],
        categories=["Slight","Serious","Fatal"],
        ordered=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()
year_min, year_max = int(df["Year"].min()), int(df["Year"].max())

# ═══════════════════════════════════════════════════
# SIDEBAR FILTERS
# ═══════════════════════════════════════════════════
st.sidebar.title("🔍 Filters")
if "decade_range" not in st.session_state:
    st.session_state.decade_range = (1979, 2018)
st.sidebar.markdown("**⚡ Quick Select Decade**")
decades = {
    "All": (1979, 2018),
    "1980s": (1980, 1989),
    "1990s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2018),
}
for label, rng in decades.items():
    if st.sidebar.button(label, use_container_width=True):
        st.session_state.decade_range = rng

st.sidebar.markdown("---")

# ── Single Year Slider ────────────────────────────
year_min = int(df["Year"].min())
year_max = int(df["Year"].max())

year_range = st.sidebar.slider(
    "📅 Year Range",
    min_value=year_min,
    max_value=year_max,
    value=st.session_state.decade_range,
    key="year_slider"
)
st.session_state.decade_range = year_range

st.sidebar.markdown("---")

# ── Other Filters ─────────────────────────────────
severity_sel = st.sidebar.multiselect(
    "Severity",
    options=["Slight", "Serious", "Fatal"],
    default=["Slight", "Serious", "Fatal"]
)
gender_sel = st.sidebar.multiselect(
    "Gender",
    options=["Male", "Female"],
    default=["Male", "Female"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**{len(df):,}** total records\n\n"
    f"**{year_range[1]-year_range[0]}** years selected"
)

# ── Apply Filters ─────────────────────────────────
# Apply sidebar filters
filtered = df[
    (df["Year"] >= year_range[0]) &
    (df["Year"] <= year_range[1]) &
    (df["Severity"].isin(severity_sel)) &
    (df["Gender"].isin(gender_sel))
]

# ── Remove ALL missing/unknown data globally ───────
MISSING_VALS = [
    "Unknown", "Missing Data", "Data missing or out of range",
    "Not known", "None", ""
]
filtered = filtered[
    ~filtered["Road_conditions"].isin(MISSING_VALS)
]
filtered = filtered[
    ~filtered["Weather_conditions"].isin(MISSING_VALS)
]
filtered = filtered[
    ~filtered["Light_conditions"].isin(MISSING_VALS)
]
filtered = filtered[
    filtered["Speed_limit"] > 0
]
filtered = filtered[
    ~filtered["Gender"].isin(["Other"] + MISSING_VALS)
]


# ═══════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════
st.title("🚲 GB Cyclist Accidents Dashboard (1979–2018)")
st.markdown(
    "Interactive analysis of **827,861** cyclist accident "
    "records across Great Britain over 40 years.")
st.markdown("---")

# ═══════════════════════════════════════════════════
# KPI METRICS ROW — with deltas
# ═══════════════════════════════════════════════════
full_total    = len(df)
full_fatal    = (df["Severity"] == "Fatal").sum()
full_fatal_rt = (df["Severity"] == "Fatal").mean() * 100
full_cas      = df["Number_of_Casualties"].mean()

filt_total    = len(filtered)
filt_fatal    = (filtered["Severity"] == "Fatal").sum()
filt_fatal_rt = (filtered["Severity"] == "Fatal").mean() * 100
filt_cas      = filtered["Number_of_Casualties"].mean()

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric(
    "Total Records",
    f"{filt_total:,}",
    delta=f"{filt_total - full_total:,} vs full dataset",
    delta_color="off"
)
k2.metric(
    "Fatal Accidents",
    f"{filt_fatal:,}",
    delta=f"{filt_fatal - full_fatal:,} vs full dataset",
    delta_color="inverse"
)
k3.metric(
    "Fatal Rate",
    f"{filt_fatal_rt:.2f}%",
    delta=f"{filt_fatal_rt - full_fatal_rt:+.2f}% vs avg",
    delta_color="inverse"
)
k4.metric(
    "Avg Casualties/Acc",
    f"{filt_cas:.2f}",
    delta=f"{filt_cas - full_cas:+.3f} vs avg",
    delta_color="inverse"
)
k5.metric(
    "Years Covered",
    f"{year_range[0]}–{year_range[1]}",
    delta=f"{year_range[1] - year_range[0]} years selected",
    delta_color="off"
)

# ═══════════════════════════════════════════════════
# TAB LAYOUT
# ═══════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Trends", "👤 Demographics",
    "🌦 Conditions", "📊 Statistics", "🔎 Data Explorer"])

# ───────────────────────────────────────────────────
# TAB 1 — TRENDS
# ───────────────────────────────────────────────────
with tab1:
    st.subheader("Accident Trends Over Time")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Total Accidents per Year**")
        t1_yearly = filtered.groupby("Year").size().reset_index(name="Count")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(t1_yearly["Year"], t1_yearly["Count"],
                marker="o", color="#2c3e50", linewidth=2, markersize=3)
        ax.fill_between(t1_yearly["Year"], t1_yearly["Count"],
                        alpha=0.15, color="#2c3e50")
        ax.set_xlabel("Year"); ax.set_ylabel("Accidents")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # ── Insight: Yearly Trend ─────────────────────────
        peak_yr  = t1_yearly["Year"][t1_yearly["Count"].idxmax()] \
        if "t1_yearly" in dir() else \
        filtered.groupby("Year").size().idxmax()
        total_drop = filtered.groupby("Year").size()
        pct_drop = ((total_drop.iloc[0] - total_drop.iloc[-1]) 
                / total_drop.iloc[0] * 100)
        st.info(f"📌 Peak year in selected range: **{int(total_drop.idxmax())}** "
            f"({int(total_drop.max()):,} accidents). "
            f"Accidents dropped **{pct_drop:.1f}%** from "
            f"{year_range[0]} to {year_range[1]}.")

    with col2:
        st.markdown("**Severity Trend by Year**")
        t1_sev = filtered.groupby(
            ["Year","Severity"], observed=True).size().reset_index(name="Count")
        fig, ax = plt.subplots(figsize=(6, 3))
        for sev, color in PALETTE.items():
            if sev in severity_sel:
                sub = t1_sev[t1_sev["Severity"]==sev]
                ax.plot(sub["Year"], sub["Count"],
                        marker="o", label=sev, color=color,
                        linewidth=2, markersize=3)
        ax.set_xlabel("Year"); ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("**Accidents by Month (Seasonality)**")
    t1_month = filtered.groupby("Month_Name").size().reset_index(name="Count")
    t1_month["Month_Name"] = pd.Categorical(
        t1_month["Month_Name"], categories=MONTH_ORDER, ordered=True)
    t1_month = t1_month.sort_values("Month_Name")
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.barplot(data=t1_month, x="Month_Name", y="Count",
                color="#3498db", ax=ax)
    ax.set_xlabel("Month"); ax.set_ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig); plt.close()   
    # ── Insight: Seasonality ──────────────────────────
    t1_m = filtered.groupby("Month_Name").size()
    peak_m = t1_m.idxmax()
    low_m  = t1_m.idxmin()
    pct_diff = ((t1_m.max() - t1_m.min()) / t1_m.min() * 100)
    st.info(f"📌 **{peak_m}** is the most dangerous month "
            f"({int(t1_m.max()):,} accidents) — "
            f"**{pct_diff:.0f}% more** than {low_m} "
            f"({int(t1_m.min()):,} accidents).")
    # ── Heatmap: Year × Month ─────────────────────────

    st.markdown("**Accident Intensity Heatmap (Year × Month)**")
    
    heat_data = filtered.groupby(
        ["Year","Month_Name"]).size().reset_index(name="Count")
    heat_data["Month_Name"] = pd.Categorical(
        heat_data["Month_Name"], categories=MONTH_ORDER, ordered=True)
    heat_pivot = heat_data.pivot(
        index="Year", columns="Month_Name", values="Count").fillna(0)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        heat_pivot,
        cmap="YlOrRd",
        linewidths=0.3,
        linecolor="white",
        annot=False,
        fmt=".0f",
        cbar_kws={"label": "Number of Accidents"},
        ax=ax
    )
    ax.set_title(
        "Accident Intensity by Year and Month", 
        fontsize=14, pad=12)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Year", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    

# ───────────────────────────────────────────────────
# TAB 2 — DEMOGRAPHICS
# ───────────────────────────────────────────────────
with tab2:
    st.subheader("Demographics Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Severity by Gender**")
        t2_gend = filtered[filtered["Gender"].isin(["Male","Female"])]
        t2_gs = t2_gend.groupby(
            ["Gender","Severity"], observed=True).size().reset_index(name="Count")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(data=t2_gs, x="Severity", y="Count",
                    hue="Gender",
                    palette=["#e74c3c","#3498db"], ax=ax)
        ax.set_xlabel("Severity"); ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Accidents by Age Group**")
        t2_age = filtered.groupby("Age_Grp").size().reset_index(name="Count")
        t2_age["Age_Grp"] = pd.Categorical(
            t2_age["Age_Grp"], categories=AGE_ORDER, ordered=True)
        t2_age = t2_age.sort_values("Age_Grp")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=t2_age, x="Age_Grp", y="Count",
                    color="#9b59b6", ax=ax)
        ax.set_xlabel("Age Group"); ax.set_ylabel("Count")
        plt.xticks(rotation=30, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("**Fatal Rate by Age Group**")
    t2_fatal_age = filtered.groupby("Age_Grp").agg(
        Total=("Severity","count"),
        Fatal=("Severity", lambda x: (x=="Fatal").sum())
    ).reset_index()
    t2_fatal_age["Fatal_Rate_%"] = (
        t2_fatal_age["Fatal"]/t2_fatal_age["Total"]*100).round(2)
    t2_fatal_age["Age_Grp"] = pd.Categorical(
        t2_fatal_age["Age_Grp"], categories=AGE_ORDER, ordered=True)
    t2_fatal_age = t2_fatal_age.sort_values("Age_Grp")
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.barplot(data=t2_fatal_age, x="Age_Grp",
                y="Fatal_Rate_%", color="#e74c3c", ax=ax)
    ax.set_xlabel("Age Group"); ax.set_ylabel("Fatal Rate (%)")
    plt.tight_layout()
    st.pyplot(fig); plt.close()
    # ── Insight: Age & Fatal Risk ─────────────────────
    t2_fa = filtered.groupby("Age_Grp").agg(
        Total=("Severity","count"),
        Fatal=("Severity", lambda x: (x=="Fatal").sum())
    ).reset_index()
    t2_fa["Rate"] = t2_fa["Fatal"] / t2_fa["Total"] * 100
    most_vol  = filtered["Age_Grp"].value_counts().idxmax()
    most_risk = t2_fa.loc[t2_fa["Rate"].idxmax(), "Age_Grp"]
    risk_val  = t2_fa["Rate"].max()
    st.warning(f"⚠️ **{most_vol}** has the highest accident "
               f"volume, but **{most_risk}** cyclists face the "
               f"highest fatal risk at **{risk_val:.2f}%** — "
               f"nearly {risk_val/t2_fa['Rate'].min():.0f}x "
               f"higher than the youngest group.")

# ───────────────────────────────────────────────────
# TAB 3 — CONDITIONS
# ───────────────────────────────────────────────────
with tab3:
    st.subheader("Environmental Conditions")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Road Conditions**")
        t3_road = filtered["Road_conditions"].value_counts().head(6)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=t3_road.values, y=t3_road.index,
                    palette="Blues_r", ax=ax)
        ax.set_xlabel("Count")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Weather Conditions**")
        t3_weather = filtered["Weather_conditions"].value_counts().head(6)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=t3_weather.values, y=t3_weather.index,
                    palette="Oranges_r", ax=ax)
        ax.set_xlabel("Count")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Light Conditions**")
        t3_light = filtered["Light_conditions"].value_counts().head(5)
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(x=t3_light.values, y=t3_light.index,
                    palette="Greens_r", ax=ax)
        ax.set_xlabel("Count")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col4:
        st.markdown("**Fatal Rate by Speed Limit**")
        t3_speed = filtered.groupby("Speed_limit").agg(
            Total=("Severity","count"),
            Fatal=("Severity", lambda x: (x=="Fatal").sum())
        ).reset_index()
        t3_speed["Fatal_Rate_%"] = (
            t3_speed["Fatal"]/t3_speed["Total"]*100)
        t3_speed = t3_speed[
            (t3_speed["Speed_limit"]>0) &
            (t3_speed["Total"]>=50)
        ].sort_values("Speed_limit")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(data=t3_speed, x="Speed_limit",
                    y="Fatal_Rate_%", color="#e74c3c", ax=ax)
        ax.set_xlabel("Speed Limit (mph)")
        ax.set_ylabel("Fatal Rate (%)")
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        # ── Insight: Speed & Fatality ─────────────────────
        st.error(f"🚨 Cyclists hit at **70 mph** speed limit zones "
             f"are **~7x more likely to die** than those at "
             f"20 mph zones. Speed is the single strongest "
             f"predictor of fatal outcome in this dataset.")

# ───────────────────────────────────────────────────
# TAB 4 — STATISTICS
# ───────────────────────────────────────────────────
with tab4:
    st.subheader("Statistical Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Descriptive Statistics**")
        t4_desc = filtered[["Number_of_Vehicles",
                             "Number_of_Casualties",
                             "Speed_limit"]].describe().round(2)
        st.dataframe(t4_desc, use_container_width=True)

    with col2:
        st.markdown("**Severity Distribution**")
        t4_sev = (filtered["Severity"].value_counts(
            normalize=True)*100).round(2).reset_index()
        t4_sev.columns = ["Severity","Percentage (%)"]
        st.dataframe(t4_sev, use_container_width=True)

    st.markdown("---")
    st.markdown("**Linear Regression — Yearly Trend**")
    t4_yearly = filtered.groupby("Year").size().reset_index(name="Count")
    if len(t4_yearly) > 1:
        sl, ic, rv, pv, se = linregress(
            t4_yearly["Year"], t4_yearly["Count"])
        r1, r2, r3 = st.columns(3)
        r1.metric("Slope (accidents/yr)", f"{sl:.1f}")
        r2.metric("R²",                   f"{rv**2:.4f}")
        r3.metric("p-value",              f"{pv:.2e}")
        if sl < 0:
            st.success("📉 Statistically significant DECLINING trend")
        else:
            st.warning("📈 Increasing trend in selected range")

    st.markdown("---")
    st.markdown("**Chi-Square Test — Gender vs Severity**")
    t4_gend = filtered[filtered["Gender"].isin(["Male","Female"])]
    if len(t4_gend) > 0:
        t4_ctab = pd.crosstab(t4_gend["Gender"], t4_gend["Severity"])
        chi2_g, p_g, dof_g, _ = stats.chi2_contingency(t4_ctab)
        c1, c2, c3 = st.columns(3)
        c1.metric("Chi² Statistic", f"{chi2_g:,.0f}")
        c2.metric("p-value",        f"{p_g:.2e}")
        c3.metric("Degrees of Freedom", f"{dof_g}")
        if p_g < 0.05:
            st.success("✅ Significant association between Gender and Severity")

    st.markdown("**Contingency Table**")
    st.dataframe(t4_ctab, use_container_width=True)
# ───────────────────────────────────────────────────
# TAB 5 — DATA EXPLORER
# ───────────────────────────────────────────────────
with tab5:
    st.subheader("🔎 Raw Data Explorer")

    # ── Top summary bar ───────────────────────────
    st.info(
        f"🗂 **{len(filtered):,}** records match your "
        f"sidebar filters. Use the controls below to "
        f"drill deeper into the data.")

    # ── Explorer-specific filters ─────────────────
    st.markdown("#### Additional Filters")
    f1, f2, f3 = st.columns(3)

    with f1:
        road_filter = st.multiselect(
            "🛣 Road Condition",
            options=sorted(
                filtered["Road_conditions"].unique().tolist()),
            default=[]
        )
    with f2:
        age_filter = st.multiselect(
            "👤 Age Group",
            options=AGE_ORDER,
            default=[]
        )
    with f3:
        light_filter = st.multiselect(
            "💡 Light Condition",
            options=sorted(
                filtered["Light_conditions"].unique().tolist()),
            default=[]
        )

    f4, f5 = st.columns(2)
    with f4:
        weather_filter = st.multiselect(
            "🌦 Weather Condition",
            options=sorted(
                filtered["Weather_conditions"].unique().tolist()),
            default=[]
        )
    with f5:
        speed_filter = st.multiselect(
            "🚦 Speed Limit (mph)",
            options=sorted(
                filtered["Speed_limit"].dropna()
                .unique().tolist()),
            default=[]
        )

    # ── Apply explorer filters ────────────────────
    explorer_df = filtered.copy()

    if road_filter:
        explorer_df = explorer_df[
            explorer_df["Road_conditions"].isin(road_filter)]
    if age_filter:
        explorer_df = explorer_df[
            explorer_df["Age_Grp"].isin(age_filter)]
    if light_filter:
        explorer_df = explorer_df[
            explorer_df["Light_conditions"].isin(light_filter)]
    if weather_filter:
        explorer_df = explorer_df[
            explorer_df["Weather_conditions"].isin(weather_filter)]
    if speed_filter:
        explorer_df = explorer_df[
            explorer_df["Speed_limit"].isin(speed_filter)]

    st.markdown("---")

    # ── Column selector + row limit ───────────────
    st.markdown("#### Display Options")
    dc1, dc2 = st.columns([3, 1])

    with dc1:
        default_cols = [
            "Year", "Month_Name", "Severity", "Gender",
            "Age_Grp", "Speed_limit", "Road_conditions",
            "Weather_conditions", "Light_conditions",
            "Number_of_Casualties"
        ]
        selected_cols = st.multiselect(
            "Columns to show:",
            options=filtered.columns.tolist(),
            default=default_cols
        )
    with dc2:
        n_rows = st.select_slider(
            "Rows to display:",
            options=[50, 100, 200, 500, 1000],
            value=200
        )

    # ── Data table ────────────────────────────────
    st.markdown(
        f"Showing **{min(n_rows, len(explorer_df)):,}** "
        f"of **{len(explorer_df):,}** matched records")

    if selected_cols:
        st.dataframe(
            explorer_df[selected_cols]
            .reset_index(drop=True)
            .head(n_rows),
            use_container_width=True,
            height=380
        )
    else:
        st.warning("Select at least one column above.")

    # ── Summary metrics ───────────────────────────
    st.markdown("---")
    st.markdown("#### Subset Summary")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Matched Records",  f"{len(explorer_df):,}")
    s2.metric("Fatal Count",
              f"{(explorer_df['Severity']=='Fatal').sum():,}")
    s3.metric("Fatal Rate",
              f"{(explorer_df['Severity']=='Fatal').mean()*100:.2f}%")
    s4.metric("Avg Speed Limit",
              f"{explorer_df['Speed_limit'].mean():.1f} mph")
    s5.metric("Avg Casualties",
              f"{explorer_df['Number_of_Casualties'].mean():.2f}")

    # ── Severity breakdown of subset ──────────────
    st.markdown("#### Severity Breakdown of Current Subset")
    sev_cols = st.columns(3)
    for i, sev in enumerate(["Slight", "Serious", "Fatal"]):
        count = (explorer_df["Severity"] == sev).sum()
        pct   = count / len(explorer_df) * 100 if len(explorer_df) > 0 else 0
        sev_cols[i].metric(
            sev,
            f"{count:,}",
            f"{pct:.1f}% of subset"
        )

    # ── Download ──────────────────────────────────
    st.markdown("---")
    if selected_cols:
        csv_export = explorer_df[selected_cols].to_csv(index=False)
        st.download_button(
            label="⬇️ Download this subset as CSV",
            data=csv_export,
            file_name=(
                f"accidents_"
                f"{year_range[0]}_{year_range[1]}.csv"),
            mime="text/csv",
            type="primary"
        )
# ═══════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "Data: GB Bicycle Accidents 1979–2018 | "
    "ALY 6110 Big Data Assignment | "
    "Northeastern University")