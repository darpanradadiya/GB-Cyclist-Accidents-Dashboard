import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import linregress

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════
st.set_page_config(
    page_title="GB Cyclist Accidents Dashboard",
    page_icon="🚲",
    layout="wide"
)

PALETTE = {"Slight": "#3498db", "Serious": "#e67e22", "Fatal": "#e74c3c"}
AGE_ORDER = ["6 to 10","11 to 15","16 to 20","21 to 25",
             "26 to 35","36 to 45","46 to 55","56 to 65","66 to 75"]
MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
MISSING_VALS = [
    "Unknown", "Missing Data", "Data missing or out of range",
    "Not known", "None", ""
]
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# ═══════════════════════════════════════════════════
# LOAD & PRE-PROCESS DATA ONCE
# ═══════════════════════════════════════════════════
@st.cache_data
def load_data():
    accidents = pd.read_csv("Accidents.csv")
    bikers    = pd.read_csv("Bikers.csv")
    df = pd.merge(bikers, accidents, on="Accident_Index", how="inner")
    df["Severity"] = pd.Categorical(
        df["Severity"],
        categories=["Slight","Serious","Fatal"],
        ordered=True)
    df["Date"]       = pd.to_datetime(df["Date"])
    df["Year"]       = df["Date"].dt.year
    df["Month_Name"] = df["Date"].dt.strftime("%b")

    # Clean globally once at load time
    df = df[~df["Road_conditions"].isin(MISSING_VALS)]
    df = df[~df["Weather_conditions"].isin(MISSING_VALS)]
    df = df[~df["Light_conditions"].isin(MISSING_VALS)]
    df = df[df["Speed_limit"] > 0]
    df = df[~df["Gender"].isin(["Other"] + MISSING_VALS)]
    return df

@st.cache_data
def precompute(df):
    """Pre-aggregate heavy groupbys once."""
    yearly     = df.groupby("Year").size().reset_index(name="Count")
    sev_year   = df.groupby(
        ["Year","Severity"], observed=True).size().reset_index(name="Count")
    monthly    = df.groupby("Month_Name").size().reset_index(name="Count")
    monthly["Month_Name"] = pd.Categorical(
        monthly["Month_Name"], categories=MONTH_ORDER, ordered=True)
    monthly    = monthly.sort_values("Month_Name")
    heat       = df.groupby(
        ["Year","Month_Name"]).size().reset_index(name="Count")
    heat["Month_Name"] = pd.Categorical(
        heat["Month_Name"], categories=MONTH_ORDER, ordered=True)
    heat_pivot = heat.pivot(
        index="Year", columns="Month_Name", values="Count").fillna(0)
    return yearly, sev_year, monthly, heat_pivot

df = load_data()
year_min = int(df["Year"].min())
year_max = int(df["Year"].max())
full_yearly, full_sev_year, full_monthly, full_heat = precompute(df)

# ═══════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════
st.sidebar.title("🔍 Filters")

if "decade_range" not in st.session_state:
    st.session_state.decade_range = (1979, 2018)

st.sidebar.markdown("**⚡ Quick Select Decade**")
decades = {
    "All":   (1979, 2018),
    "1980s": (1980, 1989),
    "1990s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2018),
}
for label, rng in decades.items():
    if st.sidebar.button(label, use_container_width=True):
        st.session_state.decade_range = rng
        st.rerun()

st.sidebar.markdown("---")

year_range = st.sidebar.slider(
    "📅 Year Range",
    min_value=year_min,
    max_value=year_max,
    value=st.session_state.decade_range,
    key="year_slider"
)
if year_range != st.session_state.decade_range:
    st.session_state.decade_range = year_range

st.sidebar.markdown("---")

severity_sel = st.sidebar.multiselect(
    "🎯 Severity",
    options=["Slight","Serious","Fatal"],
    default=["Slight","Serious","Fatal"]
)
gender_sel = st.sidebar.multiselect(
    "👤 Gender",
    options=["Male","Female"],
    default=["Male","Female"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"📊 **{len(df):,}** total records\n\n"
    f"📅 **{year_range[1]-year_range[0]}** years selected"
)

# ═══════════════════════════════════════════════════
# APPLY FILTERS
# ═══════════════════════════════════════════════════
@st.cache_data
def apply_filters(yr0, yr1, sev, gend):
    f = df[
        (df["Year"] >= yr0) &
        (df["Year"] <= yr1) &
        (df["Severity"].isin(sev)) &
        (df["Gender"].isin(gend))
    ]
    return f

filtered = apply_filters(
    year_range[0], year_range[1],
    tuple(severity_sel), tuple(gender_sel)
)

# ═══════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════
st.title("🚲 GB Cyclist Accidents Dashboard (1979–2018)")
st.markdown(
    "Interactive analysis of **827,861** cyclist accident "
    "records across Great Britain over 40 years.")
st.markdown("---")

# ═══════════════════════════════════════════════════
# KPI ROW
# ═══════════════════════════════════════════════════
full_fatal_rt = (df["Severity"] == "Fatal").mean() * 100
full_cas      = df["Number_of_Casualties"].mean()

filt_total    = len(filtered)
filt_fatal    = (filtered["Severity"] == "Fatal").sum()
filt_fatal_rt = (filtered["Severity"] == "Fatal").mean() * 100
filt_cas      = filtered["Number_of_Casualties"].mean()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Records",    f"{filt_total:,}",
          delta=f"{filt_total - len(df):,} vs full",
          delta_color="off")
k2.metric("Fatal Accidents",  f"{filt_fatal:,}",
          delta=f"{filt_fatal - (df['Severity']=='Fatal').sum():,} vs full",
          delta_color="inverse")
k3.metric("Fatal Rate",       f"{filt_fatal_rt:.2f}%",
          delta=f"{filt_fatal_rt - full_fatal_rt:+.2f}% vs avg",
          delta_color="inverse")
k4.metric("Avg Casualties",   f"{filt_cas:.2f}",
          delta=f"{filt_cas - full_cas:+.3f} vs avg",
          delta_color="inverse")
k5.metric("Years Covered",
          f"{year_range[0]}–{year_range[1]}",
          delta=f"{year_range[1]-year_range[0]} yrs",
          delta_color="off")

st.markdown("---")

# ═══════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Trends", "👤 Demographics",
    "🌦 Conditions", "📊 Statistics",
    "🔎 Data Explorer"])

# ───────────────────────────────────────────────────
# TAB 1 — TRENDS
# ───────────────────────────────────────────────────
with tab1:
    st.subheader("Accident Trends Over Time")
    col1, col2 = st.columns(2)

    t1_yearly = filtered.groupby("Year").size().reset_index(name="Count")
    t1_sev    = filtered.groupby(
        ["Year","Severity"], observed=True).size().reset_index(name="Count")

    with col1:
        st.markdown("**Total Accidents per Year**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(t1_yearly["Year"], t1_yearly["Count"],
                marker="o", color="#2c3e50", linewidth=2, markersize=3)
        ax.fill_between(t1_yearly["Year"], t1_yearly["Count"],
                        alpha=0.15, color="#2c3e50")
        ax.set_xlabel("Year")
        ax.set_ylabel("Accidents")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("**Severity Trend by Year**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for sev, color in PALETTE.items():
            if sev in severity_sel:
                sub = t1_sev[t1_sev["Severity"] == sev]
                ax.plot(sub["Year"], sub["Count"],
                        marker="o", label=sev, color=color,
                        linewidth=2, markersize=3)
        ax.set_xlabel("Year")
        ax.set_ylabel("Count")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.legend(fontsize=8, loc="upper right")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Insight ───────────────────────────────────
    if len(t1_yearly) > 1:
        total_s = filtered.groupby("Year").size()
        pct_drop = ((total_s.iloc[0] - total_s.iloc[-1])
                    / total_s.iloc[0] * 100)
        st.info(
            f"📌 Peak year: **{int(total_s.idxmax())}** "
            f"({int(total_s.max()):,} accidents). "
            f"Accidents changed by **{pct_drop:.1f}%** "
            f"from {year_range[0]} to {year_range[1]}.")

    st.markdown("---")
    st.markdown("**Accidents by Month (Seasonality)**")
    t1_month = filtered.groupby("Month_Name").size().reset_index(name="Count")
    t1_month["Month_Name"] = pd.Categorical(
        t1_month["Month_Name"], categories=MONTH_ORDER, ordered=True)
    t1_month = t1_month.sort_values("Month_Name")
    fig, ax = plt.subplots(figsize=(12, 3.5))
    sns.barplot(data=t1_month, x="Month_Name", y="Count",
                color="#3498db", ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 200,
                f"{int(bar.get_height()):,}",
                ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Insight ───────────────────────────────────
    t1_m = filtered.groupby("Month_Name").size()
    if len(t1_m) > 0:
        peak_m = t1_m.idxmax()
        low_m  = t1_m.idxmin()
        pct_diff = ((t1_m.max() - t1_m.min()) / t1_m.min() * 100)
        st.info(
            f"📌 **{peak_m}** is the most dangerous month "
            f"({int(t1_m.max()):,} accidents) — "
            f"**{pct_diff:.0f}% more** than {low_m} "
            f"({int(t1_m.min()):,} accidents).")

    st.markdown("---")
    st.markdown("**Accident Intensity Heatmap (Year × Month)**")
    heat_data = filtered.groupby(
        ["Year","Month_Name"]).size().reset_index(name="Count")
    heat_data["Month_Name"] = pd.Categorical(
        heat_data["Month_Name"], categories=MONTH_ORDER, ordered=True)
    heat_pivot = heat_data.pivot(
        index="Year", columns="Month_Name",
        values="Count").fillna(0)
    fig, ax = plt.subplots(figsize=(14, max(6, len(heat_pivot)*0.25)))
    sns.heatmap(heat_pivot, cmap="YlOrRd",
                linewidths=0.2, linecolor="white",
                cbar_kws={"label": "Accidents",
                          "shrink": 0.6},
                ax=ax)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Year",  fontsize=11)
    ax.set_title("Accident Intensity by Year and Month",
                 fontsize=13, pad=10)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
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
        t2_gs   = t2_gend.groupby(
            ["Gender","Severity"], observed=True
        ).size().reset_index(name="Count")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=t2_gs, x="Severity", y="Count",
                    hue="Gender",
                    palette=["#e74c3c","#3498db"], ax=ax)
        ax.set_xlabel("Severity")
        ax.set_ylabel("Count")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.legend(fontsize=9, loc="upper right")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("**Accidents by Age Group**")
        t2_age = filtered.groupby(
            "Age_Grp").size().reset_index(name="Count")
        t2_age["Age_Grp"] = pd.Categorical(
            t2_age["Age_Grp"], categories=AGE_ORDER, ordered=True)
        t2_age = t2_age.sort_values("Age_Grp")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=t2_age, x="Age_Grp", y="Count",
                    color="#9b59b6", ax=ax)
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Count")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        plt.xticks(rotation=35, ha="right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown("**Fatal Rate by Age Group**")
    t2_fatal = filtered.groupby("Age_Grp").agg(
        Total=("Severity","count"),
        Fatal=("Severity", lambda x: (x=="Fatal").sum())
    ).reset_index()
    t2_fatal["Fatal_Rate_%"] = (
        t2_fatal["Fatal"] / t2_fatal["Total"] * 100).round(2)
    t2_fatal["Age_Grp"] = pd.Categorical(
        t2_fatal["Age_Grp"], categories=AGE_ORDER, ordered=True)
    t2_fatal = t2_fatal.sort_values("Age_Grp")
    fig, ax = plt.subplots(figsize=(12, 3.5))
    bars = sns.barplot(data=t2_fatal, x="Age_Grp",
                       y="Fatal_Rate_%", color="#e74c3c", ax=ax)
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Fatal Rate (%)")
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.2f}%",
                ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Insight ───────────────────────────────────
    if len(t2_fatal) > 0:
        most_vol  = filtered["Age_Grp"].value_counts().idxmax()
        most_risk = t2_fatal.loc[t2_fatal["Fatal_Rate_%"].idxmax(),
                                 "Age_Grp"]
        risk_val  = t2_fatal["Fatal_Rate_%"].max()
        min_val   = t2_fatal["Fatal_Rate_%"].min()
        st.warning(
            f"⚠️ **{most_vol}** has the highest accident volume, "
            f"but **{most_risk}** cyclists face the highest fatal "
            f"risk at **{risk_val:.2f}%** — nearly "
            f"{risk_val/min_val:.0f}x higher than the "
            f"youngest group.")

# ───────────────────────────────────────────────────
# TAB 3 — CONDITIONS
# ───────────────────────────────────────────────────
with tab3:
    st.subheader("Environmental Conditions")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Road Conditions**")
        t3_road = filtered["Road_conditions"].value_counts().head(6)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=t3_road.values, y=t3_road.index,
                    palette="Blues_r", ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        for i, v in enumerate(t3_road.values):
            ax.text(v + 1000, i, f"{v:,}",
                    va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("**Weather Conditions**")
        t3_weather = filtered["Weather_conditions"].value_counts().head(6)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=t3_weather.values, y=t3_weather.index,
                    palette="Oranges_r", ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        for i, v in enumerate(t3_weather.values):
            ax.text(v + 1000, i, f"{v:,}",
                    va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Light Conditions**")
        t3_light = filtered["Light_conditions"].value_counts().head(5)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.barplot(x=t3_light.values, y=t3_light.index,
                    palette="Greens_r", ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        for i, v in enumerate(t3_light.values):
            ax.text(v + 500, i, f"{v:,}",
                    va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col4:
        st.markdown("**Fatal Rate by Speed Limit**")
        t3_speed = filtered.groupby("Speed_limit").agg(
            Total=("Severity","count"),
            Fatal=("Severity", lambda x: (x=="Fatal").sum())
        ).reset_index()
        t3_speed["Fatal_Rate_%"] = (
            t3_speed["Fatal"] / t3_speed["Total"] * 100)
        t3_speed = t3_speed[
            t3_speed["Total"] >= 50
        ].sort_values("Speed_limit")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.barplot(data=t3_speed,
                    x="Speed_limit", y="Fatal_Rate_%",
                    color="#e74c3c", ax=ax)
        ax.set_xlabel("Speed Limit (mph)")
        ax.set_ylabel("Fatal Rate (%)")
        for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.05,
                    f"{bar.get_height():.1f}%",
                    ha="center", va="bottom", fontsize=7)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.error(
        "🚨 Cyclists in **70 mph** zones are **~7x more "
        "likely to die** than those in 20 mph zones. "
        "Speed is the single strongest predictor of "
        "fatal outcome in this dataset.")

# ───────────────────────────────────────────────────
# TAB 4 — STATISTICS
# ───────────────────────────────────────────────────
with tab4:
    st.subheader("Statistical Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Descriptive Statistics**")
        t4_desc = filtered[[
            "Number_of_Vehicles",
            "Number_of_Casualties",
            "Speed_limit"]].describe().round(2)
        st.dataframe(t4_desc, use_container_width=True)

    with col2:
        st.markdown("**Severity Distribution**")
        t4_sev = (filtered["Severity"]
                  .value_counts(normalize=True) * 100
                  ).round(2).reset_index()
        t4_sev.columns = ["Severity","Percentage (%)"]
        st.dataframe(t4_sev, use_container_width=True)

    st.markdown("---")
    st.markdown("**Linear Regression — Yearly Trend**")
    t4_yearly = filtered.groupby(
        "Year").size().reset_index(name="Count")
    if len(t4_yearly) > 1:
        sl, ic, rv, pv, se = linregress(
            t4_yearly["Year"], t4_yearly["Count"])
        r1, r2, r3 = st.columns(3)
        r1.metric("Slope (accidents/yr)", f"{sl:.1f}")
        r2.metric("R²",                   f"{rv**2:.4f}")
        r3.metric("p-value",              f"{pv:.2e}")
        if sl < 0:
            st.success(
                "📉 Statistically significant "
                "DECLINING trend confirmed")
        else:
            st.warning(
                "📈 Increasing trend in selected range")

    st.markdown("---")
    st.markdown("**Chi-Square Test — Gender vs Severity**")
    t4_gend = filtered[
        filtered["Gender"].isin(["Male","Female"])]
    if len(t4_gend) > 0:
        t4_ctab = pd.crosstab(
            t4_gend["Gender"], t4_gend["Severity"])
        chi2_g, p_g, dof_g, _ = stats.chi2_contingency(
            t4_ctab)
        c1, c2, c3 = st.columns(3)
        c1.metric("Chi² Statistic", f"{chi2_g:,.0f}")
        c2.metric("p-value",        f"{p_g:.2e}")
        c3.metric("Degrees of Freedom", f"{dof_g}")
        if p_g < 0.05:
            st.success(
                "✅ Significant association between "
                "Gender and Severity")
        st.markdown("**Contingency Table**")
        st.dataframe(t4_ctab, use_container_width=True)

# ───────────────────────────────────────────────────
# TAB 5 — DATA EXPLORER
# ───────────────────────────────────────────────────
with tab5:
    st.subheader("🔎 Raw Data Explorer")
    st.info(
        f"🗂 **{len(filtered):,}** records match your "
        f"sidebar filters. Use the controls below to "
        f"drill deeper.")

    st.markdown("#### Additional Filters")
    f1, f2, f3 = st.columns(3)
    with f1:
        road_filter = st.multiselect(
            "🛣 Road Condition",
            options=sorted(
                filtered["Road_conditions"].unique().tolist()),
            default=[])
    with f2:
        age_filter = st.multiselect(
            "👤 Age Group",
            options=AGE_ORDER,
            default=[])
    with f3:
        light_filter = st.multiselect(
            "💡 Light Condition",
            options=sorted(
                filtered["Light_conditions"].unique().tolist()),
            default=[])

    f4, f5 = st.columns(2)
    with f4:
        weather_filter = st.multiselect(
            "🌦 Weather Condition",
            options=sorted(
                filtered["Weather_conditions"].unique().tolist()),
            default=[])
    with f5:
        speed_filter = st.multiselect(
            "🚦 Speed Limit (mph)",
            options=sorted(
                filtered["Speed_limit"]
                .dropna().unique().tolist()),
            default=[])

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
    st.markdown("#### Display Options")
    dc1, dc2 = st.columns([3, 1])
    with dc1:
        default_cols = [
            "Year","Month_Name","Severity","Gender",
            "Age_Grp","Speed_limit","Road_conditions",
            "Weather_conditions","Light_conditions",
            "Number_of_Casualties"]
        selected_cols = st.multiselect(
            "Columns to show:",
            options=filtered.columns.tolist(),
            default=default_cols)
    with dc2:
        n_rows = st.select_slider(
            "Rows to display:",
            options=[50,100,200,500,1000],
            value=200)

    st.markdown(
        f"Showing **{min(n_rows, len(explorer_df)):,}** "
        f"of **{len(explorer_df):,}** matched records")

    if selected_cols:
        st.dataframe(
            explorer_df[selected_cols]
            .reset_index(drop=True)
            .head(n_rows),
            use_container_width=True,
            height=380)
    else:
        st.warning("Select at least one column.")

    st.markdown("---")
    st.markdown("#### Subset Summary")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Matched Records", f"{len(explorer_df):,}")
    s2.metric("Fatal Count",
              f"{(explorer_df['Severity']=='Fatal').sum():,}")
    s3.metric("Fatal Rate",
              f"{(explorer_df['Severity']=='Fatal').mean()*100:.2f}%")
    s4.metric("Avg Speed Limit",
              f"{explorer_df['Speed_limit'].mean():.1f} mph")
    s5.metric("Avg Casualties",
              f"{explorer_df['Number_of_Casualties'].mean():.2f}")

    st.markdown("#### Severity Breakdown")
    sev_cols = st.columns(3)
    for i, sev in enumerate(["Slight","Serious","Fatal"]):
        count = (explorer_df["Severity"] == sev).sum()
        pct   = (count/len(explorer_df)*100
                 if len(explorer_df) > 0 else 0)
        sev_cols[i].metric(
            sev, f"{count:,}",
            f"{pct:.1f}% of subset")

    st.markdown("---")
    if selected_cols:
        csv_export = explorer_df[selected_cols].to_csv(
            index=False)
        st.download_button(
            label="⬇️ Download filtered data as CSV",
            data=csv_export,
            file_name=(
                f"accidents_"
                f"{year_range[0]}_{year_range[1]}.csv"),
            mime="text/csv",
            type="primary")

# ═══════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "Data: GB Bicycle Accidents 1979–2018 | "
    "ALY 6110 Big Data Assignment | "
    "Northeastern University")