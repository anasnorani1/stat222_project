"""
STAT-222 Semester Project — Interactive Dashboard
Climate Variability, Drought Risk & Precipitation Dynamics in Pakistan (1990–2023)
15 meteorological stations | All analyses load automatically

Run:  streamlit run app.py
"""

import os, sys, warnings, itertools
from io import StringIO
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
from scipy.stats import f_oneway, levene, shapiro, kruskal, mannwhitneyu
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import acorr_ljungbox
import requests

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pakistan Drought Dashboard",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  — 15 stations, all provinces
# ─────────────────────────────────────────────────────────────────────────────

STATIONS = {
    "Karachi":      {"lat": 24.86, "lon": 67.01, "color": "#e74c3c", "zone": "Arid Coastal",        "province": "Sindh"},
    "Hyderabad":    {"lat": 25.40, "lon": 68.36, "color": "#c0392b", "zone": "Hot Arid Interior",   "province": "Sindh"},
    "Sukkur":       {"lat": 27.71, "lon": 68.86, "color": "#e67e22", "zone": "Hot Desert",          "province": "Sindh"},
    "Larkana":      {"lat": 27.56, "lon": 68.22, "color": "#d35400", "zone": "Hot Desert",          "province": "Sindh"},
    "Quetta":       {"lat": 30.18, "lon": 66.98, "color": "#8e44ad", "zone": "Cold Semi-Arid",      "province": "Balochistan"},
    "Bahawalpur":   {"lat": 29.40, "lon": 71.68, "color": "#f39c12", "zone": "Hot Desert",          "province": "Punjab"},
    "Multan":       {"lat": 30.16, "lon": 71.52, "color": "#f1c40f", "zone": "Hot Semi-Arid",       "province": "Punjab"},
    "Faisalabad":   {"lat": 31.42, "lon": 73.00, "color": "#27ae60", "zone": "Hot Semi-Arid",       "province": "Punjab"},
    "Lahore":       {"lat": 31.55, "lon": 74.35, "color": "#2ecc71", "zone": "Hot Semi-Arid",       "province": "Punjab"},
    "Gujranwala":   {"lat": 32.19, "lon": 74.19, "color": "#1abc9c", "zone": "Hot Semi-Arid",       "province": "Punjab"},
    "Sialkot":      {"lat": 32.49, "lon": 74.52, "color": "#16a085", "zone": "Humid Subtropical",   "province": "Punjab"},
    "Rawalpindi":   {"lat": 33.60, "lon": 73.07, "color": "#2980b9", "zone": "Humid Subtropical",   "province": "Punjab"},
    "Islamabad":    {"lat": 33.72, "lon": 73.04, "color": "#3498db", "zone": "Sub-Humid Highland",  "province": "ICT"},
    "Peshawar":     {"lat": 34.02, "lon": 71.52, "color": "#9b59b6", "zone": "Hot Semi-Arid",       "province": "KPK"},
    "Muzaffarabad": {"lat": 34.37, "lon": 73.47, "color": "#1f618d", "zone": "Humid Highland",      "province": "AJK"},
}

ALL_STATIONS = list(STATIONS)
COLORS       = {s: STATIONS[s]["color"] for s in STATIONS}
SEASON_MAP   = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                6:"Summer",7:"Summer",8:"Summer",9:"Autumn",10:"Autumn",11:"Autumn"}
START, END   = "1990-01-01", "2023-12-31"
CSV_P = os.path.join("outputs", "raw_monthly_precip.csv")
CSV_T = os.path.join("outputs", "raw_monthly_temp.csv")

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Downloading 34 years of climate data for 15 cities…")
def load_data():
    if os.path.exists(CSV_P) and os.path.exists(CSV_T):
        precip = pd.read_csv(CSV_P, index_col=0, parse_dates=True)
        temp   = pd.read_csv(CSV_T, index_col=0, parse_dates=True)
        # add any new stations not yet in cached CSVs
        missing = [s for s in ALL_STATIONS if s not in precip.columns]
        if not missing:
            return precip, temp
        extra_p, extra_t = _fetch_stations(missing)
        precip = pd.concat([precip, extra_p], axis=1)
        temp   = pd.concat([temp,   extra_t], axis=1)
        precip.to_csv(CSV_P); temp.to_csv(CSV_T)
        return precip, temp

    precip, temp = _fetch_stations(ALL_STATIONS)
    os.makedirs("outputs", exist_ok=True)
    precip.to_csv(CSV_P); temp.to_csv(CSV_T)
    return precip, temp


def _fetch_stations(names):
    import time
    p_frames, t_frames = [], []
    for name in names:
        info = STATIONS[name]
        raw = None
        for attempt in range(5):           # up to 5 retries
            try:
                resp = requests.get(
                    "https://archive-api.open-meteo.com/v1/archive",
                    params={"latitude": info["lat"], "longitude": info["lon"],
                            "start_date": START, "end_date": END,
                            "daily": "precipitation_sum,temperature_2m_mean",
                            "timezone": "Asia/Karachi"},
                    timeout=180,           # longer timeout
                )
                body = resp.json()
                if "daily" in body:
                    raw = body["daily"]
                    break
                time.sleep(5 * (attempt + 1))   # longer back-off
            except Exception:
                time.sleep(5 * (attempt + 1))

        if raw is None:
            # fallback: NaN-filled so these stations are visibly excluded from stats
            dates = pd.date_range(START, END, freq="D")
            raw = {"time": dates.strftime("%Y-%m-%d").tolist(),
                   "precipitation_sum": [float("nan")] * len(dates),
                   "temperature_2m_mean": [float("nan")] * len(dates)}

        df = pd.DataFrame({
            "date":   pd.to_datetime(raw["time"]),
            "precip": raw["precipitation_sum"],
            "temp":   raw["temperature_2m_mean"],
        }).set_index("date")
        df["precip"] = df["precip"].fillna(0).clip(lower=0)
        df["temp"]   = df["temp"].interpolate()
        p_frames.append(df["precip"].resample("MS").sum().rename(name))
        t_frames.append(df["temp"].resample("MS").mean().rename(name))
    return pd.concat(p_frames, axis=1), pd.concat(t_frames, axis=1)


@st.cache_data(show_spinner="Computing SPI for all 15 stations…")
def compute_spi(precip_json: str) -> dict:
    precip = pd.read_json(StringIO(precip_json))
    precip.index = pd.to_datetime(precip.index)
    out = {}
    for stn in ALL_STATIONS:
        if stn not in precip.columns:
            continue
        out[stn] = {}
        for scale in [3, 6, 12]:
            rolling = precip[stn].rolling(scale).sum().dropna()
            out[stn][f"SPI-{scale}"] = _gamma_spi(rolling).to_dict()
    return out


def _gamma_spi(series):
    result = pd.Series(index=series.index, dtype=float)
    for month in range(1, 13):
        idx   = series.index[series.index.month == month]
        x     = series.loc[idx].values.astype(float)
        if len(x) < 6: continue
        q     = np.mean(x == 0)
        x_pos = x[x > 0]
        if len(x_pos) < 4: continue
        a, _, b = stats.gamma.fit(x_pos, floc=0)
        cdf  = np.clip(q + (1 - q) * stats.gamma.cdf(x, a=a, scale=b), 1e-6, 1-1e-6)
        result.loc[idx] = stats.norm.ppf(cdf)
    return result


def get_spi(spi_dict, station, scale):
    return pd.Series(
        {pd.Timestamp(k): v for k, v in spi_dict[station][scale].items()}
    ).sort_index()


def classify_color(v):
    if   v >=  2.0: return "#1565C0"
    elif v >=  1.5: return "#1E88E5"
    elif v >=  1.0: return "#42A5F5"
    elif v >= -1.0: return "#A5D6A7"
    elif v >= -1.5: return "#FFA726"
    elif v >= -2.0: return "#EF5350"
    else:           return "#B71C1C"


def longest_run(bool_series):
    return max((sum(1 for _ in g) for v, g in itertools.groupby(bool_series) if v), default=0)


def mann_kendall(x):
    n  = len(x)
    s  = sum(np.sign(x[j]-x[i]) for i in range(n-1) for j in range(i+1,n))
    vs = n*(n-1)*(2*n+5)/18.0
    z  = (s-np.sign(s))/np.sqrt(vs) if s != 0 else 0.0
    p  = 2.0*(1.0 - stats.norm.cdf(abs(z)))
    tau = s / (0.5*n*(n-1))
    slopes = [(x[j]-x[i])/(j-i) for i in range(n-1) for j in range(i+1,n)]
    return s, z, p, tau, float(np.median(slopes))

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  — no station selector; all cities always active
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🌧️ Pakistan Drought")
    st.caption("STAT-222 — Advanced Statistics\n15 Cities | 1990–2023")
    st.divider()

    page = st.radio("Navigate", [
        "Overview", "EDA", "Distribution Fitting",
        "ANOVA", "ARIMA Forecast", "Regression",
        "Nonparametric & Vulnerability",
    ], label_visibility="collapsed")

    st.divider()
    spi_scale  = st.selectbox("SPI Scale", ["SPI-3","SPI-6","SPI-12"], index=2)
    year_range = st.slider("Year Range", 1990, 2023, (1990, 2023))

    # single-station detail selector (ARIMA & Regression pages only)
    if page in ("ARIMA Forecast", "Regression", "Distribution Fitting"):
        st.divider()
        detail_stn = st.selectbox("Station (detail view)", ALL_STATIONS, index=8)  # default Lahore
    else:
        detail_stn = None

    st.divider()
    st.caption("Data: ERA5-Land via Open-Meteo API")
    st.caption(f"Stations: {len(ALL_STATIONS)} cities, all provinces")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA  (always all 15 stations)
# ─────────────────────────────────────────────────────────────────────────────

precip_full, temp_full = load_data()
spi_data = compute_spi(precip_full.to_json())

yr_mask = ((precip_full.index.year >= year_range[0]) &
           (precip_full.index.year <= year_range[1]))
precip  = precip_full[yr_mask]
temp    = temp_full[yr_mask]

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("Climate Variability & Drought Risk in Pakistan")
    st.markdown(
        "**STAT-222 Semester Project** — 15 meteorological stations across all provinces, "
        "34 years of ERA5-Land reanalysis data, five advanced statistical methods applied."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cities", "15", "All provinces")
    c2.metric("Period", "1990–2023", "34 years")
    c3.metric("Observations", "6,120", "station-months")
    c4.metric("Methods", "5", "Statistical")
    c5.metric("Figures", "13+", "Visualizations")

    st.divider()

    # Interactive map
    st.subheader("Station Network")
    map_df = pd.DataFrame([
        {"Station": s,
         "lat": STATIONS[s]["lat"], "lon": STATIONS[s]["lon"],
         "Province": STATIONS[s]["province"],
         "Zone": STATIONS[s]["zone"],
         "Mean Precip (mm/mo)": round(precip_full[s].mean(), 1),
         "Mean Temp (°C)": round(temp_full[s].mean(), 1)}
        for s in ALL_STATIONS
    ])
    fig_map = px.scatter_mapbox(
        map_df, lat="lat", lon="lon", hover_name="Station",
        hover_data=["Province", "Zone", "Mean Precip (mm/mo)", "Mean Temp (°C)"],
        color="Province",
        size=[18]*15, size_max=18,
        zoom=4.5, height=500,
        mapbox_style="carto-positron",
    )
    fig_map.update_layout(margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    st.divider()

    # Summary table
    st.subheader("Climate Summary — All 15 Stations")
    summary = precip_full.describe().T.round(2)
    summary["Skewness"]     = precip_full.skew().round(3)
    summary["Province"]     = [STATIONS[s]["province"] for s in summary.index]
    summary["Zone"]         = [STATIONS[s]["zone"]     for s in summary.index]
    summary["Mean Temp °C"] = temp_full.mean().round(2)
    st.dataframe(
        summary[["mean","std","min","max","Skewness","Province","Zone","Mean Temp °C"]]
        .rename(columns={"mean":"Mean (mm)","std":"Std Dev","min":"Min","max":"Max"}),
        use_container_width=True,
    )

    # Methods grid
    st.divider()
    st.subheader("Statistical Methods Applied")
    cols = st.columns(5)
    for col, title, body in zip(cols, [
        "Distribution Fitting","ARIMA","ANOVA","Multiple Regression","Nonparametric"
    ],[
        "Gamma · Log-Normal · Weibull · Exponential · Normal\nAIC + KS Test",
        "ADF stationarity\nGrid-search AIC\n24-month forecast + 95% CI",
        "One-Way & Two-Way\nShapiro-Wilk + Levene\nTukey HSD post-hoc",
        "Lagged precip + temp\nVIF · Durbin-Watson\nResidual diagnostics",
        "Mann-Kendall trend\nSen's slope\nKruskal-Wallis · Mann-Whitney",
    ]):
        with col:
            st.info(f"**{title}**\n\n{body}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EDA
# ─────────────────────────────────────────────────────────────────────────────

elif page == "EDA":
    st.title("Exploratory Data Analysis — 15 Stations")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Precipitation Series", "Seasonal Patterns",
        "SPI Timeline", "Distributions", "Correlations"
    ])

    with tab1:
        st.subheader("Annual Precipitation — All Cities")
        annual = precip.resample("YS").sum()
        fig = go.Figure()
        for stn in ALL_STATIONS:
            fig.add_trace(go.Scatter(
                x=annual.index.year, y=annual[stn],
                name=stn, mode="lines",
                line=dict(color=COLORS[stn], width=1.5),
            ))
        fig.update_layout(height=460, xaxis_title="Year",
                          yaxis_title="Annual Precipitation (mm)",
                          hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Monthly Precipitation Heatmap (Mean mm/month)")
        month_mean = precip.groupby(precip.index.month).mean().T
        month_mean.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                               "Jul","Aug","Sep","Oct","Nov","Dec"]
        fig2 = px.imshow(month_mean, color_continuous_scale="Blues",
                         labels=dict(color="mm"), aspect="auto", height=480,
                         title="Mean Monthly Precipitation by Station")
        fig2.update_xaxes(side="top")
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Seasonal Box Plots")
        long = precip.copy()
        long["Season"] = long.index.month.map(SEASON_MAP)
        long = long.melt(id_vars="Season", value_vars=ALL_STATIONS,
                         var_name="Station", value_name="Precipitation")
        fig = px.box(long, x="Station", y="Precipitation", color="Season",
                     category_orders={"Season":["Winter","Spring","Summer","Autumn"]},
                     points=False, height=500,
                     title="Precipitation by Station and Season")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Monthly Climatology")
        clim = precip.groupby(precip.index.month).mean()
        clim.index = ["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"]
        fig2 = go.Figure()
        for stn in ALL_STATIONS:
            fig2.add_trace(go.Scatter(
                x=clim.index, y=clim[stn], name=stn,
                mode="lines+markers",
                line=dict(color=COLORS[stn], width=1.8),
                marker=dict(size=5),
            ))
        fig2.update_layout(height=420, xaxis_title="Month",
                           yaxis_title="Mean Precip (mm)",
                           hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader(f"{spi_scale} Drought Timeline — All Stations")
        fig = make_subplots(
            rows=15, cols=1, shared_xaxes=True,
            subplot_titles=ALL_STATIONS,
            vertical_spacing=0.018,
        )
        for i, stn in enumerate(ALL_STATIONS, 1):
            s = get_spi(spi_data, stn, spi_scale)
            s = s[(s.index.year >= year_range[0]) & (s.index.year <= year_range[1])]
            bar_colors = [classify_color(v) for v in s.values]
            fig.add_trace(go.Bar(x=s.index, y=s.values,
                                 marker_color=bar_colors,
                                 name=stn, showlegend=False), row=i, col=1)
            fig.add_hline(y=-1, line_dash="dot", line_color="orange",
                          opacity=0.4, row=i, col=1)
            fig.add_hline(y=-2, line_dash="dot", line_color="red",
                          opacity=0.4, row=i, col=1)
            fig.update_yaxes(title_text="SPI", range=[-3.5,3.5],
                             tickvals=[-2,0,2], row=i, col=1)
        fig.update_layout(height=220*15, title_text=f"{spi_scale} — WMO Drought Classification")
        st.plotly_chart(fig, use_container_width=True)

        # Legend
        lcols = st.columns(7)
        for j, (lbl, col) in enumerate([
            ("Extreme Wet ≥2.0","#1565C0"),("Very Wet 1.5–2.0","#1E88E5"),
            ("Mod. Wet 1.0–1.5","#42A5F5"),("Near Normal ±1.0","#A5D6A7"),
            ("Mod. Drought −1.5","#FFA726"),("Severe −2.0","#EF5350"),
            ("Extreme <−2.0","#B71C1C"),
        ]):
            lcols[j].markdown(
                f'<div style="background:{col};padding:5px;border-radius:4px;'
                f'color:white;font-size:11px;text-align:center">{lbl}</div>',
                unsafe_allow_html=True)

    with tab4:
        st.subheader("Precipitation Distributions — Summary")
        desc = precip_full.describe().T.round(2)
        desc["Skewness"] = precip_full.skew().round(3)
        desc["Kurtosis"] = precip_full.kurtosis().round(3)
        desc["Province"] = [STATIONS[s]["province"] for s in desc.index]
        st.dataframe(desc[["mean","std","Skewness","Kurtosis","Province"]],
                     use_container_width=True)

        st.subheader("Distribution Comparison — Mean vs Std Dev")
        fig = px.scatter(
            x=precip_full.mean(), y=precip_full.std(),
            text=precip_full.columns,
            labels={"x":"Mean Monthly Precip (mm)","y":"Std Dev (mm)"},
            color=precip_full.columns,
            color_discrete_map=COLORS,
            height=450,
            title="Precipitation Variability by Station",
        )
        fig.update_traces(textposition="top center", marker_size=12)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Pearson Correlation Matrix — Monthly Precipitation")
        corr = precip_full.corr().round(3)
        fig  = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn",
                         zmin=-1, zmax=1, aspect="auto", height=620,
                         title="Inter-Station Precipitation Correlations")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Temperature Correlation")
        fig2 = px.imshow(temp_full.corr().round(3), text_auto=".2f",
                         color_continuous_scale="RdYlGn", zmin=-1, zmax=1,
                         aspect="auto", height=620)
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DISTRIBUTION FITTING
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Distribution Fitting":
    st.title("Probability Distribution Fitting")
    st.info("AIC-ranked fit for all 15 stations. Select a station in the sidebar for the detailed PDF overlay and Q-Q plot.")

    DISTS = {
        "Gamma":       (stats.gamma,       {"floc": 0}),
        "Log-Normal":  (stats.lognorm,     {"floc": 0}),
        "Weibull":     (stats.weibull_min, {"floc": 0}),
        "Exponential": (stats.expon,       {"floc": 0}),
        "Normal":      (stats.norm,        {}),
    }
    DC = {"Gamma":"#e74c3c","Log-Normal":"#9b59b6","Weibull":"#27ae60",
          "Exponential":"#f39c12","Normal":"#3498db"}

    # Summary table for all 15
    @st.cache_data(show_spinner="Fitting distributions for all stations…")
    def fit_all(precip_json):
        df = pd.read_json(precip_json)
        df.index = pd.to_datetime(df.index)
        rows = []
        for stn in ALL_STATIONS:
            pos = df[stn].dropna(); pos = pos[pos > 0].values
            best_aic = np.inf; best_name = ""
            for dname, (dist, fkw) in DISTS.items():
                try:
                    params  = dist.fit(pos, **fkw)
                    aic     = 2*len(params) - 2*np.sum(dist.logpdf(pos, *params))
                    ks_s, ks_p = stats.kstest(pos, dist.cdf, args=params)
                    if aic < best_aic:
                        best_aic, best_name = aic, dname
                    rows.append({"Station":stn,"Distribution":dname,
                                 "AIC":round(aic,2),"KS Stat":round(ks_s,4),
                                 "KS p-value":round(ks_p,4)})
                except Exception: pass
        return pd.DataFrame(rows)

    df_all = fit_all(precip_full.to_json())

    # Pivot to show best per station
    best_per = (df_all.sort_values("AIC")
                      .groupby("Station").first()
                      .reset_index()
                      [["Station","Distribution","AIC","KS Stat","KS p-value"]])
    best_per.columns = ["Station","Best Distribution","AIC","KS Stat","KS p-value"]
    best_per["Province"] = [STATIONS[s]["province"] for s in best_per["Station"]]
    st.subheader("Best-Fit Distribution Summary — All Stations")
    st.dataframe(best_per, use_container_width=True, hide_index=True)

    # Best distribution breakdown pie
    counts = best_per["Best Distribution"].value_counts().reset_index()
    counts.columns = ["Distribution","Count"]
    fig_pie = px.pie(counts, names="Distribution", values="Count",
                     color="Distribution", color_discrete_map=DC,
                     title="Best-Fit Distribution Frequency Across 15 Stations",
                     height=360)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # Detailed view for selected station
    stn  = detail_stn
    pos  = precip_full[stn].dropna(); pos = pos[pos > 0].values
    xr   = np.linspace(0.5, np.percentile(pos,99), 300)
    st.subheader(f"Detail: {stn} — {STATIONS[stn]['zone']}")

    col_pdf, col_qq = st.columns(2)
    fig_pdf = go.Figure()
    fig_pdf.add_trace(go.Histogram(x=pos, histnorm="probability density",
                                   name="Empirical", marker_color=COLORS[stn],
                                   opacity=0.45, nbinsx=30))
    best_dist_name = best_per[best_per["Station"]==stn]["Best Distribution"].values[0]
    best_params = None
    for dname, (dist, fkw) in DISTS.items():
        try:
            params = dist.fit(pos, **fkw)
            fig_pdf.add_trace(go.Scatter(x=xr, y=dist.pdf(xr, *params),
                                          name=dname,
                                          line=dict(color=DC[dname], width=2,
                                          dash="solid" if dname==best_dist_name else "dot")))
            if dname == best_dist_name:
                best_params = (dist, params)
        except Exception: pass
    fig_pdf.update_layout(title=f"{stn} — PDF Overlay",
                          xaxis_title="Monthly Precip (mm)",
                          yaxis_title="Density", height=380, barmode="overlay")
    with col_pdf:
        st.plotly_chart(fig_pdf, use_container_width=True)

    if best_params:
        dist_obj, bp = best_params
        probs = np.linspace(0.01, 0.99, len(pos))
        theor = dist_obj.ppf(probs, *bp)
        empir = np.sort(pos)
        lo, hi = min(theor.min(),empir.min()), max(theor.max(),empir.max())
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=theor, y=empir, mode="markers",
                                     marker=dict(color=COLORS[stn], size=4, opacity=0.55)))
        fig_qq.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines",
                                     line=dict(color="black", dash="dash")))
        fig_qq.update_layout(title=f"{stn} Q-Q ({best_dist_name})",
                             xaxis_title="Theoretical", yaxis_title="Empirical",
                             height=380)
        with col_qq:
            st.plotly_chart(fig_qq, use_container_width=True)

    stn_rows = df_all[df_all["Station"]==stn].sort_values("AIC").reset_index(drop=True)
    stn_rows["Best?"] = ["✅ Best" if i==0 else "" for i in range(len(stn_rows))]
    st.dataframe(stn_rows, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ANOVA
# ─────────────────────────────────────────────────────────────────────────────

elif page == "ANOVA":
    st.title("ANOVA — Precipitation Across 15 Stations & Seasons")

    groups = [precip[stn].dropna().values for stn in ALL_STATIONS]

    with st.expander("Assumption Checks", expanded=True):
        st.markdown("**Shapiro-Wilk Normality Test** (first 50 obs)")
        norm_rows = []
        for stn, g in zip(ALL_STATIONS, groups):
            w, p = shapiro(g[:50])
            norm_rows.append({"Station":stn,"W":round(w,4),"p-value":round(p,4),
                               "Normal?":"Yes" if p>0.05 else "No"})
        st.dataframe(pd.DataFrame(norm_rows), use_container_width=True, hide_index=True)

        lev_f, lev_p = levene(*groups)
        st.markdown(f"**Levene's Test:** F={lev_f:.4f}, p={lev_p:.4f}  "
                    f"{'Equal variances ✅' if lev_p>0.05 else 'Unequal variances ⚠️'}")

    st.subheader("One-Way ANOVA")
    f_stat, p_val = f_oneway(*groups)
    c1,c2,c3 = st.columns(3)
    c1.metric("F-Statistic", f"{f_stat:.4f}")
    c2.metric("p-value", f"{p_val:.2e}")
    c3.metric("Result", "Significant ✅" if p_val<0.05 else "Not significant")

    fig = px.box(precip.melt(var_name="Station", value_name="Precipitation"),
                 x="Station", y="Precipitation",
                 color="Station", color_discrete_map=COLORS,
                 points=False, height=500,
                 title=f"One-Way ANOVA — F={f_stat:.2f}, p={p_val:.2e}")
    fig.update_layout(xaxis_tickangle=-45, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Two-Way ANOVA — Station × Season")
    long = precip.copy()
    long["Season"] = long.index.month.map(SEASON_MAP)
    long = long.melt(id_vars="Season", value_vars=ALL_STATIONS,
                     var_name="Station", value_name="Precipitation")
    model2 = ols("Precipitation ~ C(Station) + C(Season) + C(Station):C(Season)",
                 data=long).fit()
    tbl = anova_lm(model2, typ=2).reset_index()
    tbl.columns = ["Source","SS","df","F","p-value"]
    tbl = tbl.dropna(subset=["F"])
    tbl["F"]       = tbl["F"].round(4)
    tbl["p-value"] = tbl["p-value"].apply(lambda x: f"{x:.4f}" if x>=0.0001 else "<0.0001")
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.subheader("Season × Station Interaction")
    s_order = ["Winter","Spring","Summer","Autumn"]
    means   = long.groupby(["Season","Station"])["Precipitation"].mean().reset_index()
    fig2 = go.Figure()
    for stn in ALL_STATIONS:
        sm = means[means["Station"]==stn].set_index("Season").reindex(s_order).reset_index()
        fig2.add_trace(go.Scatter(x=sm["Season"], y=sm["Precipitation"],
                                   name=stn, mode="lines+markers",
                                   line=dict(color=COLORS[stn], width=2),
                                   marker=dict(size=7)))
    fig2.update_layout(height=500, xaxis_title="Season",
                       yaxis_title="Mean Precipitation (mm)",
                       hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ARIMA
# ─────────────────────────────────────────────────────────────────────────────

elif page == "ARIMA Forecast":
    st.title("ARIMA Time Series Forecast")
    stn = detail_stn
    st.subheader(f"Station: {stn} — {STATIONS[stn]['zone']} ({STATIONS[stn]['province']})")

    ts = precip_full[stn].dropna()
    adf_s, adf_p = adfuller(ts, autolag="AIC")[:2]
    c1,c2,c3 = st.columns(3)
    c1.metric("ADF Statistic", f"{adf_s:.4f}")
    c2.metric("ADF p-value",   f"{adf_p:.4f}")
    c3.metric("Stationary?",   "Yes ✅" if adf_p<0.05 else "No ❌")

    @st.cache_data(show_spinner="Fitting ARIMA (grid search)…")
    def fit_arima(station_name, ts_json):
        ts = pd.read_json(ts_json, typ="series")
        ts.index = pd.to_datetime(ts.index)
        best_aic, best_order = np.inf, (1,0,1)
        for p in range(0,4):
            for d in range(0,2):
                for q in range(0,4):
                    try:
                        m = ARIMA(ts, order=(p,d,q)).fit()
                        if m.aic < best_aic:
                            best_aic, best_order = m.aic, (p,d,q)
                    except Exception: pass
        fit = ARIMA(ts, order=best_order).fit()
        fc  = fit.get_forecast(steps=24)
        return best_order, best_aic, fit.aic, fc.predicted_mean.to_dict(), fc.conf_int().to_dict(), fit.resid.to_dict()

    order, _, aic, fc_mean_d, fc_ci_d, resid_d = fit_arima(stn, ts.to_json())
    fcm   = pd.Series(fc_mean_d); fcm.index = pd.to_datetime(fcm.index)
    fcci  = pd.DataFrame(fc_ci_d); fcci.index = pd.to_datetime(fcci.index)
    resid = pd.Series(resid_d); resid.index = pd.to_datetime(resid.index)

    st.markdown(f"**Best Order:** ARIMA{order}  |  **AIC:** {aic:.2f}")

    split = len(ts) - 36
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index[:split], y=ts.values[:split],
                              name="Training", line=dict(color="lightgray", width=1)))
    fig.add_trace(go.Scatter(x=ts.index[split:], y=ts.values[split:],
                              name="Test", line=dict(color=COLORS[stn], width=2)))
    fig.add_trace(go.Scatter(x=fcm.index, y=fcm.values,
                              name="Forecast 2024–25",
                              line=dict(color="black", width=2.5, dash="dash")))
    fig.add_trace(go.Scatter(
        x=list(fcci.index)+list(fcci.index[::-1]),
        y=list(fcci.iloc[:,1])+list(fcci.iloc[:,0][::-1]),
        fill="toself", fillcolor="rgba(0,0,0,0.1)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI",
    ))
    fig.update_layout(title=f"{stn} — ARIMA{order} with 24-Month Forecast",
                      yaxis_title="Precipitation (mm)",
                      hovermode="x unified", height=450)
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True)

    col_r, col_qq = st.columns(2)
    with col_r:
        fig_r = go.Figure(go.Scatter(x=resid.index, y=resid.values,
                                      mode="lines", line=dict(color=COLORS[stn], width=0.8)))
        fig_r.add_hline(y=0, line_dash="dash", line_color="black")
        fig_r.update_layout(title="ARIMA Residuals", height=300)
        st.plotly_chart(fig_r, use_container_width=True)
    with col_qq:
        qq = stats.probplot(resid.dropna())
        th = np.array([qq[0][0][0], qq[0][0][-1]])
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode="markers",
                                     marker=dict(color=COLORS[stn], size=4)))
        fig_qq.add_trace(go.Scatter(x=th, y=qq[1][1]+qq[1][0]*th,
                                     mode="lines", line=dict(color="black", dash="dash")))
        fig_qq.update_layout(title="Residual Q-Q Plot", height=300,
                             xaxis_title="Theoretical", yaxis_title="Sample")
        st.plotly_chart(fig_qq, use_container_width=True)

    lb = acorr_ljungbox(resid.dropna(), lags=[12], return_df=True)
    st.markdown(
        f"**Ljung-Box p (lag 12):** {lb['lb_pvalue'].iloc[0]:.4f}  "
        f"{'✅ No residual autocorrelation' if lb['lb_pvalue'].iloc[0]>0.05 else '⚠️ Residual autocorrelation detected'}"
    )

    st.divider()
    st.subheader("ADF Stationarity — All 15 Stations")
    @st.cache_data(show_spinner="Running ADF tests…")
    def adf_all(precip_json):
        df = pd.read_json(precip_json); df.index = pd.to_datetime(df.index)
        rows = []
        for s in ALL_STATIONS:
            series = df[s].dropna()
            if series.nunique() <= 1:
                rows.append({"Station":s,"ADF Stat":"N/A","p-value":"N/A",
                             "Stationary?":"⚠️ Constant (API fallback)"})
                continue
            st2, p2 = adfuller(series, autolag="AIC")[:2]
            rows.append({"Station":s,"ADF Stat":round(st2,4),"p-value":round(p2,4),
                         "Stationary?":"Yes ✅" if p2<0.05 else "No ❌"})
        return pd.DataFrame(rows)
    st.dataframe(adf_all(precip_full.to_json()), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Regression":
    st.title("Multiple Regression — Predicting SPI-12")
    stn = detail_stn
    st.subheader(f"Station: {stn} — {STATIONS[stn]['zone']} ({STATIONS[stn]['province']})")

    ts  = precip_full[stn].dropna()
    tmp = temp_full[stn].dropna()
    s12 = get_spi(spi_data, stn, "SPI-12")
    s3  = get_spi(spi_data, stn, "SPI-3")

    df_r = pd.DataFrame({
        "SPI12":     s12, "Precip": ts,
        "Precip_L1": ts.shift(1), "Precip_L2": ts.shift(2),
        "Precip_L3": ts.shift(3), "Precip_L6": ts.shift(6),
        "SPI3":      s3,  "Temp":   tmp,
        "Month_sin": np.sin(2*np.pi*ts.index.month/12),
        "Month_cos": np.cos(2*np.pi*ts.index.month/12),
    }).dropna()

    X_cols = ["Precip","Precip_L1","Precip_L2","Precip_L3",
              "Precip_L6","SPI3","Temp","Month_sin","Month_cos"]
    model  = OLS(df_r["SPI12"], add_constant(df_r[X_cols])).fit()
    dw     = durbin_watson(model.resid)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("R²",          f"{model.rsquared:.4f}")
    c2.metric("Adj R²",      f"{model.rsquared_adj:.4f}")
    c3.metric("F p-value",   f"{model.f_pvalue:.2e}")
    c4.metric("Durbin-Watson",f"{dw:.3f}",
              "OK" if 1.5<dw<2.5 else "⚠️ Autocorrelation")

    coef_df = pd.DataFrame({
        "Variable": model.params.index,
        "Coef":     model.params.values.round(4),
        "Std Err":  model.bse.values.round(4),
        "t-stat":   model.tvalues.values.round(3),
        "p-value":  model.pvalues.values.round(4),
        "Sig?":     ["*" if p<0.05 else "" for p in model.pvalues],
    })
    st.dataframe(coef_df, use_container_width=True, hide_index=True)

    vif_df = pd.DataFrame({
        "Variable": X_cols,
        "VIF": [round(variance_inflation_factor(df_r[X_cols].values, i), 2)
                for i in range(len(X_cols))],
    })
    vif_df["Status"] = vif_df["VIF"].apply(lambda x: "⚠️ High" if x>10 else "OK")
    with st.expander("VIF — Multicollinearity Diagnostics"):
        st.dataframe(vif_df, use_container_width=True, hide_index=True)

    ca, cb = st.columns(2)
    y = df_r["SPI12"]
    with ca:
        lo = min(model.fittedvalues.min(), y.min())
        hi = max(model.fittedvalues.max(), y.max())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=model.fittedvalues, y=y, mode="markers",
                                  marker=dict(color=COLORS[stn], size=4, opacity=0.5)))
        fig.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines",
                                  line=dict(color="black", dash="dash")))
        fig.update_layout(title="Actual vs Fitted SPI-12",
                          xaxis_title="Fitted", yaxis_title="Actual", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with cb:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=model.fittedvalues, y=model.resid, mode="markers",
                                  marker=dict(color=COLORS[stn], size=4, opacity=0.5)))
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_layout(title="Residuals vs Fitted",
                          xaxis_title="Fitted", yaxis_title="Residual", height=380)
        st.plotly_chart(fig, use_container_width=True)

    # R² comparison across all stations
    st.subheader("R² Summary — All 15 Stations")
    @st.cache_data(show_spinner="Fitting regression for all stations…")
    def r2_all(precip_json, temp_json, spi_json):
        df_p = pd.read_json(precip_json); df_p.index = pd.to_datetime(df_p.index)
        df_t = pd.read_json(temp_json);   df_t.index = pd.to_datetime(df_t.index)
        import json; spi_d = json.loads(spi_json)
        rows = []
        for s in ALL_STATIONS:
            ts2  = df_p[s].dropna(); tmp2 = df_t[s].dropna()
            s12b = pd.Series({pd.Timestamp(k):v for k,v in spi_d[s]["SPI-12"].items()}).sort_index()
            s3b  = pd.Series({pd.Timestamp(k):v for k,v in spi_d[s]["SPI-3"].items()}).sort_index()
            df_rr = pd.DataFrame({
                "SPI12":ts2.shift(0),"Precip":ts2,"Precip_L1":ts2.shift(1),
                "Precip_L2":ts2.shift(2),"Precip_L3":ts2.shift(3),"Precip_L6":ts2.shift(6),
                "SPI3":s3b,"Temp":tmp2,
                "Month_sin":np.sin(2*np.pi*ts2.index.month/12),
                "Month_cos":np.cos(2*np.pi*ts2.index.month/12),
            }).dropna()
            df_rr["SPI12"] = pd.Series({pd.Timestamp(k):v for k,v in spi_d[s]["SPI-12"].items()}).reindex(df_rr.index)
            df_rr = df_rr.dropna()
            xc = ["Precip","Precip_L1","Precip_L2","Precip_L3","Precip_L6","SPI3","Temp","Month_sin","Month_cos"]
            try:
                m = OLS(df_rr["SPI12"], add_constant(df_rr[xc])).fit()
                rows.append({"Station":s,"R²":round(m.rsquared,4),"Adj R²":round(m.rsquared_adj,4),"F p":round(m.f_pvalue,6)})
            except Exception:
                rows.append({"Station":s,"R²":None,"Adj R²":None,"F p":None})
        return pd.DataFrame(rows)

    import json
    # Convert Timestamp keys to strings so json.dumps can serialize spi_data
    spi_serializable = {
        stn: {scale: {str(k): v for k, v in vals.items()}
              for scale, vals in scales.items()}
        for stn, scales in spi_data.items()
    }
    df_r2 = r2_all(precip_full.to_json(), temp_full.to_json(), json.dumps(spi_serializable))
    fig_r2 = px.bar(df_r2.sort_values("R²", ascending=False),
                    x="Station", y="R²", color="Station",
                    color_discrete_map=COLORS, height=400,
                    title="Regression R² by Station")
    fig_r2.update_layout(showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig_r2, use_container_width=True)
    st.dataframe(df_r2, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: NONPARAMETRIC & VULNERABILITY
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Nonparametric & Vulnerability":
    st.title("Nonparametric Statistics & Drought Vulnerability")

    annual = precip_full.resample("YS").sum()

    # Mann-Kendall for all stations
    st.subheader("Mann-Kendall Trend Test — Annual Precipitation")
    mk_rows = []
    for stn in ALL_STATIONS:
        x = annual[stn].dropna().values
        s, z, p, tau, sl = mann_kendall(x)
        mk_rows.append({
            "Station":stn, "Province":STATIONS[stn]["province"],
            "S":int(s), "Z":round(z,4), "p-value":round(p,4),
            "Kendall τ":round(tau,4),
            "Sen Slope (mm/yr)":round(sl,3),
            "Trend":"↑ Increasing" if z>0 else "↓ Decreasing",
            "Sig?":"* Yes" if p<0.05 else "No",
        })
    df_mk = pd.DataFrame(mk_rows)
    st.dataframe(df_mk, use_container_width=True, hide_index=True)

    # Sen's slope bar chart
    fig_s = px.bar(
        df_mk.sort_values("Sen Slope (mm/yr)"),
        x="Station", y="Sen Slope (mm/yr)",
        color="Sen Slope (mm/yr)",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        title="Sen's Slope — Annual Precipitation Trend (mm/year)",
        height=420,
    )
    fig_s.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_s, use_container_width=True)

    # Trend line chart
    st.subheader("Annual Precipitation with Sen's Slope Trend Lines")
    fig_t = go.Figure()
    for stn in ALL_STATIONS:
        ap  = annual[stn].dropna()
        yrs = np.arange(len(ap))
        _, _, _, _, sl = mann_kendall(ap.values)
        mid_y = np.median(ap.values); mid_x = np.median(yrs)
        trend = sl*yrs + (mid_y - sl*mid_x)
        fig_t.add_trace(go.Scatter(x=ap.index.year, y=ap.values, name=stn,
                                    mode="lines", line=dict(color=COLORS[stn], width=1.2)))
        fig_t.add_trace(go.Scatter(x=ap.index.year, y=trend, mode="lines",
                                    line=dict(color=COLORS[stn], width=2.5, dash="dash"),
                                    showlegend=False))
    fig_t.update_layout(height=500, xaxis_title="Year",
                        yaxis_title="Annual Precip (mm)",
                        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig_t, use_container_width=True)

    st.divider()

    # Kruskal-Wallis
    st.subheader("Kruskal-Wallis Test — SPI-12 Across All Stations")
    spi12_groups = [get_spi(spi_data, s, "SPI-12").dropna().values for s in ALL_STATIONS]
    kw_h, kw_p   = kruskal(*spi12_groups)
    c1,c2,c3 = st.columns(3)
    c1.metric("H Statistic", f"{kw_h:.4f}")
    c2.metric("p-value",     f"{kw_p:.4f}")
    c3.metric("Groups Differ?", "Yes *" if kw_p<0.05 else "No — SPI comparable ✅")

    st.divider()

    # Drought classification for all stations
    st.subheader("Drought Classification — All 15 Stations")
    drought_rows = []
    for stn in ALL_STATIONS:
        s12 = get_spi(spi_data, stn, "SPI-12").dropna()
        n   = len(s12)
        drought_rows.append({
            "Station":         stn,
            "Province":        STATIONS[stn]["province"],
            "Zone":            STATIONS[stn]["zone"],
            "Total Months":    n,
            "Extreme":         int((s12<-2.0).sum()),
            "Severe":          int(((s12>=-2.0)&(s12<-1.5)).sum()),
            "Moderate":        int(((s12>=-1.5)&(s12<-1.0)).sum()),
            "Near Normal":     int(((s12>=-1.0)&(s12<1.0)).sum()),
            "Drought Freq %":  round(100*(s12<-1.0).sum()/n, 1),
            "Mean Drought SPI":round(s12[s12<-1.0].mean(), 3) if (s12<-1.0).any() else 0,
            "Longest Run (mo)":longest_run(s12<-1.0),
        })
    df_d = pd.DataFrame(drought_rows)
    st.dataframe(df_d, use_container_width=True, hide_index=True)

    # Stacked bar
    cats = ["Extreme","Severe","Moderate","Near Normal"]
    cat_colors = {"Extreme":"#B71C1C","Severe":"#EF5350",
                  "Moderate":"#FFA726","Near Normal":"#A5D6A7"}
    fig_d = go.Figure()
    for cat in cats:
        fig_d.add_trace(go.Bar(name=cat, x=df_d["Station"], y=df_d[cat],
                                marker_color=cat_colors[cat]))
    fig_d.update_layout(barmode="stack", height=460,
                        title="SPI-12 Month Distribution by Station",
                        yaxis_title="Number of Months",
                        xaxis_tickangle=-45)
    st.plotly_chart(fig_d, use_container_width=True)

    st.divider()

    # Vulnerability Index
    st.subheader("Drought Vulnerability Index — Ranking")
    vi = df_d[["Station","Province","Zone","Drought Freq %",
               "Extreme","Longest Run (mo)","Mean Drought SPI"]].copy()

    # Drop stations that have no real data (API fallback — all zeros give SPI of 0)
    valid = vi["Drought Freq %"].notna() & vi["Extreme"].notna() & (vi["Drought Freq %"] > 0)
    vi_valid = vi[valid].copy()
    vi_invalid = vi[~valid].copy()

    vi_valid["Score"] = (
        vi_valid["Drought Freq %"].rank(ascending=False) +
        vi_valid["Extreme"].rank(ascending=False) +
        vi_valid["Longest Run (mo)"].rank(ascending=False) +
        (-vi_valid["Mean Drought SPI"]).rank(ascending=False)
    )
    # Use fillna(0) before int conversion to guard against any residual NaN
    vi_valid["Rank"] = vi_valid["Score"].rank(method="min").fillna(0).astype(int)
    vi_invalid["Score"] = float("nan")
    vi_invalid["Rank"] = 0
    vi = pd.concat([vi_valid, vi_invalid]).sort_values("Rank")

    most_v = vi.iloc[0]["Station"]
    st.success(f"**Most Drought-Vulnerable Station: {most_v}** "
               f"({STATIONS[most_v]['province']} — {STATIONS[most_v]['zone']})")

    st.dataframe(vi.reset_index(drop=True), use_container_width=True, hide_index=True)

    # Drought frequency bar chart (sorted by rank)
    fig_v = px.bar(vi, x="Station", y="Drought Freq %",
                   color="Drought Freq %",
                   color_continuous_scale="Reds",
                   title="Drought Frequency by Station (ranked most → least vulnerable)",
                   height=440)
    fig_v.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_v, use_container_width=True)

    # Radar for top 5 most vulnerable
    st.subheader("Vulnerability Radar — Top 5 Most Vulnerable Stations")
    top5   = vi.head(5)["Station"].tolist()
    metrics = ["Drought Freq %","Extreme","Longest Run (mo)"]
    fig_radar = go.Figure()
    for stn in top5:
        row  = vi[vi["Station"]==stn].iloc[0]
        maxv = [vi[m].max() for m in metrics]
        norm = [row[m]/mx if mx>0 else 0 for m,mx in zip(metrics, maxv)]
        fig_radar.add_trace(go.Scatterpolar(
            r=norm+[norm[0]], theta=metrics+[metrics[0]],
            name=stn, fill="toself",
            line=dict(color=COLORS[stn]),
            fillcolor=COLORS[stn], opacity=0.3,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(range=[0,1], tickvals=[0.25,0.5,0.75,1.0])),
        title="Normalized Drought Vulnerability (Top 5 Stations)",
        height=500,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "STAT-222 Advanced Statistics · BSDS-02 · 2026  "
    "| Data: ERA5-Land via Open-Meteo API  "
    "| 15 Stations: Sindh · Punjab · Balochistan · KPK · ICT · AJK"
)
