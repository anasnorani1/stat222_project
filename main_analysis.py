# =============================================================================
# STAT-222: Advanced Statistics — Semester Project
# =============================================================================
# Title  : Climate Variability, Drought Risk & Precipitation Dynamics in
#          Pakistan — A Multi-Station Statistical Assessment (1990–2023)
# Methods: Probability Distribution Fitting, ARIMA, ANOVA (One-Way & Two-Way),
#          Multiple Regression, Nonparametric Statistics (Mann-Kendall,
#          Kruskal-Wallis, Mann-Whitney, Drought Frequency Analysis)
# Data   : Open-Meteo Archive API — daily precipitation, 34-year record
# Stations: 15 cities across Pakistan (all provinces + AJK)
# =============================================================================

import os
import sys
import warnings
import numpy as np

# Fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import requests
from scipy import stats
from scipy.special import gammaln
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

# =============================================================================
# CONFIGURATION
# =============================================================================

STATIONS = {
    # Sindh
    "Karachi":      {"lat": 24.86, "lon": 67.01, "color": "#e74c3c", "zone": "Arid Coastal",         "province": "Sindh"},
    "Hyderabad":    {"lat": 25.40, "lon": 68.36, "color": "#c0392b", "zone": "Hot Arid Interior",    "province": "Sindh"},
    "Sukkur":       {"lat": 27.71, "lon": 68.86, "color": "#e67e22", "zone": "Hot Desert",           "province": "Sindh"},
    "Larkana":      {"lat": 27.56, "lon": 68.22, "color": "#d35400", "zone": "Hot Desert",           "province": "Sindh"},
    # Balochistan
    "Quetta":       {"lat": 30.18, "lon": 66.98, "color": "#8e44ad", "zone": "Cold Semi-Arid",       "province": "Balochistan"},
    # Punjab
    "Bahawalpur":   {"lat": 29.40, "lon": 71.68, "color": "#f39c12", "zone": "Hot Desert",           "province": "Punjab"},
    "Multan":       {"lat": 30.16, "lon": 71.52, "color": "#f1c40f", "zone": "Hot Semi-Arid",        "province": "Punjab"},
    "Faisalabad":   {"lat": 31.42, "lon": 73.00, "color": "#27ae60", "zone": "Hot Semi-Arid",        "province": "Punjab"},
    "Lahore":       {"lat": 31.55, "lon": 74.35, "color": "#2ecc71", "zone": "Hot Semi-Arid",        "province": "Punjab"},
    "Gujranwala":   {"lat": 32.19, "lon": 74.19, "color": "#1abc9c", "zone": "Hot Semi-Arid",        "province": "Punjab"},
    "Sialkot":      {"lat": 32.49, "lon": 74.52, "color": "#16a085", "zone": "Humid Subtropical",    "province": "Punjab"},
    # Islamabad / Rawalpindi
    "Rawalpindi":   {"lat": 33.60, "lon": 73.07, "color": "#2980b9", "zone": "Humid Subtropical",    "province": "Punjab"},
    "Islamabad":    {"lat": 33.72, "lon": 73.04, "color": "#3498db", "zone": "Sub-Humid Highland",   "province": "ICT"},
    # KPK
    "Peshawar":     {"lat": 34.02, "lon": 71.52, "color": "#9b59b6", "zone": "Hot Semi-Arid",        "province": "KPK"},
    # AJK
    "Muzaffarabad": {"lat": 34.37, "lon": 73.47, "color": "#1f618d", "zone": "Humid Highland",       "province": "AJK"},
}

START_DATE = "1990-01-01"
END_DATE   = "2023-12-31"
OUT        = "outputs"
os.makedirs(OUT, exist_ok=True)

SPI_THRESHOLDS = {
    "Extreme Wet":      (2.00,  9.99),
    "Very Wet":         (1.50,  2.00),
    "Moderately Wet":   (1.00,  1.50),
    "Near Normal":      (-1.00, 1.00),
    "Moderate Drought": (-1.50, -1.00),
    "Severe Drought":   (-2.00, -1.50),
    "Extreme Drought":  (-9.99, -2.00),
}

SEASON_MAP = {12: "Winter", 1: "Winter",  2: "Winter",
              3:  "Spring", 4: "Spring",  5: "Spring",
              6:  "Summer", 7: "Summer",  8: "Summer",
              9:  "Autumn", 10: "Autumn", 11: "Autumn"}

# =============================================================================
# SECTION 1 — DATA COLLECTION
# =============================================================================

def fetch_station(name: str, lat: float, lon: float) -> pd.DataFrame:
    """Download daily precipitation + temperature from Open-Meteo Archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": START_DATE, "end_date": END_DATE,
        "daily": "precipitation_sum,temperature_2m_mean",
        "timezone": "Asia/Karachi",
    }
    resp = requests.get(url, params=params, timeout=90)
    resp.raise_for_status()
    raw = resp.json()["daily"]
    df = pd.DataFrame({
        "date":   pd.to_datetime(raw["time"]),
        "precip": raw["precipitation_sum"],
        "temp":   raw["temperature_2m_mean"],
    }).set_index("date")
    df["precip"] = df["precip"].fillna(0).clip(lower=0)
    df["temp"]   = df["temp"].interpolate()
    return df


def load_monthly_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (monthly_precip_df, monthly_temp_df) for all three stations."""
    precip_frames, temp_frames = [], []
    for name, info in STATIONS.items():
        print(f"  Fetching {name}...")
        daily = fetch_station(name, info["lat"], info["lon"])
        monthly_p = daily["precip"].resample("MS").sum().rename(name)
        monthly_t = daily["temp"].resample("MS").mean().rename(name)
        precip_frames.append(monthly_p)
        temp_frames.append(monthly_t)
    precip = pd.concat(precip_frames, axis=1)
    temp   = pd.concat(temp_frames,   axis=1)
    return precip, temp

# =============================================================================
# SECTION 2 — SPI CALCULATION (WMO Standard)
# =============================================================================

def _fit_gamma_spi(series: pd.Series) -> pd.Series:
    """
    Fit a 2-parameter Gamma distribution to monthly rolling precipitation
    for each calendar month separately (WMO Technical Document 1090).
    Zero-precipitation months are handled via a mixed probability model.
    Returns the SPI series.
    """
    spi = pd.Series(index=series.index, dtype=float, name="SPI")
    for month in range(1, 13):
        idx = series.index[series.index.month == month]
        x   = series.loc[idx].values.astype(float)
        if len(x) < 6:
            continue
        q     = np.mean(x == 0)              # P(precipitation = 0)
        x_pos = x[x > 0]
        if len(x_pos) < 4:
            continue
        # MLE for Gamma (α=shape, β=scale)
        alpha, _, beta = stats.gamma.fit(x_pos, floc=0)
        # Mixed CDF: H(x) = q + (1-q)*G(x)
        cdf = q + (1.0 - q) * stats.gamma.cdf(x, a=alpha, scale=beta)
        cdf = np.clip(cdf, 1e-6, 1.0 - 1e-6)
        spi.loc[idx] = stats.norm.ppf(cdf)
    return spi


def compute_all_spi(precip: pd.DataFrame) -> dict:
    """
    Returns nested dict: spi[station][scale] = pd.Series of SPI values.
    Scales: 3-month, 6-month, 12-month.
    """
    spi = {}
    for station in STATIONS:
        spi[station] = {}
        for scale in [3, 6, 12]:
            rolling = precip[station].rolling(scale).sum().dropna()
            spi[station][f"SPI-{scale}"] = _fit_gamma_spi(rolling)
    return spi

# =============================================================================
# SECTION 3 — EXPLORATORY DATA ANALYSIS
# =============================================================================

def run_eda(precip: pd.DataFrame, temp: pd.DataFrame, spi: dict) -> None:
    print("\n" + "="*65)
    print("SECTION 3 — EXPLORATORY DATA ANALYSIS")
    print("="*65)

    # ── 3a. Descriptive statistics ──────────────────────────────────────────
    print("\n--- Precipitation Descriptive Statistics (mm/month) ---")
    desc = precip.describe().round(2)
    desc.loc["skewness"] = precip.skew().round(3)
    desc.loc["kurtosis"] = precip.kurtosis().round(3)
    print(desc.to_string())

    print("\n--- Temperature Descriptive Statistics (°C/month mean) ---")
    print(temp.describe().round(2).to_string())

    # ── 3b. Monthly precipitation time series ───────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    for i, (stn, info) in enumerate(STATIONS.items()):
        axes[i].bar(precip.index, precip[stn], color=info["color"],
                    alpha=0.7, width=25, label=stn)
        axes[i].set_ylabel("Precip (mm)")
        axes[i].set_title(f"{stn} — Monthly Precipitation 1990–2023"
                          f"  |  Climate Zone: {info['zone']}", fontweight="bold")
        # 5-year rolling mean
        roll = precip[stn].rolling(60).mean()
        axes[i].plot(roll.index, roll.values, color="black", lw=1.5,
                     linestyle="--", label="5-yr rolling mean")
        axes[i].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{OUT}/01_monthly_precip_timeseries.png", bbox_inches="tight")
    plt.close()
    print("\n[Saved] 01_monthly_precip_timeseries.png")

    # ── 3c. Histograms + fitted gamma overlay ───────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (stn, info) in enumerate(STATIONS.items()):
        data = precip[stn].dropna()
        pos  = data[data > 0]
        axes[i].hist(data, bins=30, density=True, color=info["color"],
                     alpha=0.6, edgecolor="white", label="Empirical")
        alpha_g, _, beta_g = stats.gamma.fit(pos, floc=0)
        xr = np.linspace(0, pos.quantile(0.99), 300)
        axes[i].plot(xr, stats.gamma.pdf(xr, alpha_g, scale=beta_g),
                     "k-", lw=2, label=f"Gamma(α={alpha_g:.2f})")
        axes[i].plot(xr, stats.lognorm.pdf(xr, *stats.lognorm.fit(pos, floc=0)),
                     "b--", lw=1.5, label="Log-Normal")
        axes[i].set_title(f"{stn}", fontweight="bold")
        axes[i].set_xlabel("Monthly Precipitation (mm)")
        axes[i].set_ylabel("Density")
        axes[i].legend()
    fig.suptitle("Precipitation Distributions with Fitted PDFs", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT}/02_precip_distributions.png", bbox_inches="tight")
    plt.close()
    print("[Saved] 02_precip_distributions.png")

    # ── 3d. Seasonal box plots ───────────────────────────────────────────────
    long = precip.copy()
    long["Season"] = long.index.month.map(SEASON_MAP)
    long = long.melt(id_vars="Season", value_vars=list(STATIONS),
                     var_name="Station", value_name="Precipitation")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=long, x="Season", y="Precipitation", hue="Station",
                palette={s: STATIONS[s]["color"] for s in STATIONS},
                order=["Winter", "Spring", "Summer", "Autumn"], ax=ax,
                fliersize=2)
    ax.set_title("Seasonal Precipitation Distribution by Station", fontweight="bold")
    ax.set_ylabel("Monthly Precipitation (mm)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/03_seasonal_boxplots.png", bbox_inches="tight")
    plt.close()
    print("[Saved] 03_seasonal_boxplots.png")

    # ── 3e. SPI-12 drought timeline ──────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    for i, (stn, info) in enumerate(STATIONS.items()):
        s12 = spi[stn]["SPI-12"].dropna()
        ax  = axes[i]
        ax.axhline(0,  color="black",  lw=0.6, ls="--")
        ax.axhline(-1, color="orange", lw=0.8, ls=":")
        ax.axhline(-2, color="red",    lw=0.8, ls=":")
        ax.axhline(1,  color="green",  lw=0.8, ls=":")
        ax.fill_between(s12.index, s12, 0, where=(s12 < 0),
                        color="#c0392b", alpha=0.45, label="Drought")
        ax.fill_between(s12.index, s12, 0, where=(s12 >= 0),
                        color="#27ae60", alpha=0.45, label="Wet")
        ax.plot(s12.index, s12, color=info["color"], lw=0.9)
        ax.set_ylim(-3.5, 3.5)
        ax.set_ylabel("SPI-12")
        ax.set_title(f"{stn} — SPI-12 (1990–2023)", fontweight="bold")
        ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{OUT}/04_spi12_timeline.png", bbox_inches="tight")
    plt.close()
    print("[Saved] 04_spi12_timeline.png")

    # ── 3f. Correlation heatmap ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, df, title in zip(axes, [precip, temp],
                              ["Precipitation Correlation", "Temperature Correlation"]):
        sns.heatmap(df.corr(), annot=True, fmt=".3f", cmap="RdYlGn",
                    center=0, vmin=-1, vmax=1, ax=ax, square=True,
                    linewidths=0.5)
        ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/05_correlation_heatmaps.png", bbox_inches="tight")
    plt.close()
    print("[Saved] 05_correlation_heatmaps.png")

    # ── 3g. Multi-scale SPI comparison (Lahore example) ─────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = {"SPI-3": "#e74c3c", "SPI-6": "#f39c12", "SPI-12": "#2980b9"}
    for scale, col in colors.items():
        s = spi["Lahore"][scale].dropna()
        ax.plot(s.index, s.values, color=col, lw=1.0, alpha=0.85, label=scale)
    ax.axhline(-1, color="orange", lw=0.8, ls="--")
    ax.axhline(-2, color="red",    lw=0.8, ls="--")
    ax.axhline(0,  color="black",  lw=0.5, ls="--")
    ax.set_ylim(-3.5, 3.5)
    ax.set_title("Lahore — SPI at Multiple Timescales (3, 6, 12 months)",
                 fontweight="bold")
    ax.set_ylabel("SPI Value")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/06_spi_multiscale_lahore.png", bbox_inches="tight")
    plt.close()
    print("[Saved] 06_spi_multiscale_lahore.png")

# =============================================================================
# SECTION 4 — PROBABILITY DISTRIBUTION FITTING
# =============================================================================

def run_distribution_fitting(precip: pd.DataFrame) -> dict:
    print("\n" + "="*65)
    print("SECTION 4 — PROBABILITY DISTRIBUTION FITTING")
    print("="*65)

    DISTS = {
        "Gamma":       (stats.gamma,    {"floc": 0}),
        "Log-Normal":  (stats.lognorm,  {"floc": 0}),
        "Weibull":     (stats.weibull_min, {"floc": 0}),
        "Exponential": (stats.expon,    {"floc": 0}),
        "Normal":      (stats.norm,     {}),
    }

    results = {}
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    for idx, (stn, info) in enumerate(STATIONS.items()):
        data = precip[stn].dropna()
        pos  = data[data > 0].values
        results[stn] = {}

        ax_pdf = axes[idx, 0]
        ax_qq  = axes[idx, 1]

        xr = np.linspace(0.5, np.percentile(pos, 99), 300)
        ax_pdf.hist(pos, bins=25, density=True, color=info["color"],
                    alpha=0.5, edgecolor="white", label="Empirical")

        aic_dict = {}
        for dname, (dist, fkw) in DISTS.items():
            try:
                params   = dist.fit(pos, **fkw)
                log_lik  = np.sum(dist.logpdf(pos, *params))
                aic      = 2 * len(params) - 2 * log_lik
                ks_s, ks_p = stats.kstest(pos, dist.cdf, args=params)
                results[stn][dname] = {
                    "params": params, "AIC": aic,
                    "KS_stat": ks_s, "KS_p": ks_p
                }
                aic_dict[dname] = aic
                ax_pdf.plot(xr, dist.pdf(xr, *params), lw=1.5, label=dname)
            except Exception:
                pass

        best = min(aic_dict, key=aic_dict.get)
        ax_pdf.set_title(f"{stn} — Distribution Fitting\n(Best fit: {best} by AIC)",
                         fontweight="bold")
        ax_pdf.set_xlabel("Monthly Precipitation (mm)")
        ax_pdf.set_ylabel("Density")
        ax_pdf.legend(fontsize=8)

        # Q-Q for best distribution
        bd, bfkw = DISTS[best]
        bp = bd.fit(pos, **bfkw)
        probs = np.linspace(0.01, 0.99, len(pos))
        theor = bd.ppf(probs, *bp)
        empir = np.sort(pos)
        ax_qq.scatter(theor, empir, color=info["color"], s=12, alpha=0.5)
        lo, hi = min(theor.min(), empir.min()), max(theor.max(), empir.max())
        ax_qq.plot([lo, hi], [lo, hi], "k--", lw=1.5)
        ax_qq.set_title(f"{stn} — Q-Q Plot ({best})", fontweight="bold")
        ax_qq.set_xlabel("Theoretical Quantiles")
        ax_qq.set_ylabel("Empirical Quantiles")

        # Print table
        print(f"\n{stn} — Goodness-of-Fit Summary:")
        print(f"  {'Distribution':<14} {'AIC':>9} {'KS Stat':>9} {'KS p-value':>12}  Best?")
        print("  " + "-"*55)
        for dn in DISTS:
            if dn in results[stn]:
                r = results[stn][dn]
                marker = " <-- BEST" if dn == best else ""
                print(f"  {dn:<14} {r['AIC']:>9.2f} {r['KS_stat']:>9.4f}"
                      f" {r['KS_p']:>12.4f}{marker}")

    plt.suptitle("Distribution Fitting: All Stations", fontsize=13,
                 fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.savefig(f"{OUT}/07_distribution_fitting.png", bbox_inches="tight")
    plt.close()
    print("\n[Saved] 07_distribution_fitting.png")
    return results

# =============================================================================
# SECTION 5 — ANOVA (ONE-WAY & TWO-WAY)
# =============================================================================

def run_anova(precip: pd.DataFrame, spi: dict) -> None:
    print("\n" + "="*65)
    print("SECTION 5 — ANOVA ANALYSIS")
    print("="*65)

    stations_list = list(STATIONS)

    # Build long-format dataframe
    long = precip.copy()
    long["Season"] = long.index.month.map(SEASON_MAP)
    long["Year"]   = long.index.year
    long = long.melt(id_vars=["Season", "Year"], value_vars=stations_list,
                     var_name="Station", value_name="Precipitation")

    groups = [long.loc[long["Station"] == s, "Precipitation"].dropna().values
              for s in stations_list]

    # ── 5.1 Assumption Checks ────────────────────────────────────────────────
    print("\n5.1 Assumption Validation")
    print("  a) Normality — Shapiro-Wilk Test (first 50 obs per station):")
    for stn, g in zip(stations_list, groups):
        sw_s, sw_p = stats.shapiro(g[:50])
        verdict = "normal" if sw_p > 0.05 else "NOT normal"
        print(f"     {stn}: W={sw_s:.4f}, p={sw_p:.4f}  [{verdict}]")

    lev_s, lev_p = stats.levene(*groups)
    print(f"\n  b) Homogeneity of Variance — Levene's Test:")
    print(f"     F={lev_s:.4f}, p={lev_p:.4f}  "
          f"[{'Equal variances' if lev_p > 0.05 else 'Unequal variances'}]")

    # ── 5.2 One-Way ANOVA ────────────────────────────────────────────────────
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"\n5.2 One-Way ANOVA — Precipitation Across Stations")
    print(f"  F = {f_stat:.4f},  p = {p_val:.6f}")
    print(f"  → {'Significant difference' if p_val < 0.05 else 'No significant difference'}"
          f" at α = 0.05")

    # Tukey HSD post-hoc
    from scipy.stats import tukey_hsd
    tukey = tukey_hsd(*groups)
    print("\n  Tukey HSD Post-Hoc (p-values):")
    for i in range(len(stations_list)):
        for j in range(i + 1, len(stations_list)):
            p = tukey.pvalue[i, j]
            print(f"    {stations_list[i]} vs {stations_list[j]}: "
                  f"p = {p:.4f}  {'*' if p < 0.05 else 'ns'}")

    # ── 5.3 Two-Way ANOVA: Station × Season ──────────────────────────────────
    print("\n5.3 Two-Way ANOVA — Station × Season Interaction")
    model2way = ols(
        "Precipitation ~ C(Station) + C(Season) + C(Station):C(Season)",
        data=long
    ).fit()
    tbl = anova_lm(model2way, typ=2)
    print(tbl.round(4).to_string())

    # ── Visualizations ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # One-way box plot
    sns.boxplot(data=long, x="Station", y="Precipitation",
                palette={s: STATIONS[s]["color"] for s in STATIONS},
                ax=axes[0], fliersize=2)
    axes[0].set_title(f"One-Way ANOVA: Precipitation by Station\n"
                      f"F={f_stat:.2f}, p={p_val:.4f}", fontweight="bold")
    axes[0].set_ylabel("Monthly Precipitation (mm)")

    # Interaction plot
    s_order = ["Winter", "Spring", "Summer", "Autumn"]
    means   = long.groupby(["Season", "Station"])["Precipitation"].mean().reset_index()
    for stn, info in STATIONS.items():
        sm = (means[means["Station"] == stn]
              .set_index("Season").reindex(s_order))
        axes[1].plot(s_order, sm["Precipitation"].values,
                     marker="o", color=info["color"], label=stn, lw=2, ms=7)
    axes[1].set_title("Two-Way ANOVA Interaction: Station × Season",
                      fontweight="bold")
    axes[1].set_xlabel("Season")
    axes[1].set_ylabel("Mean Precipitation (mm)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{OUT}/08_anova_analysis.png", bbox_inches="tight")
    plt.close()
    print("\n[Saved] 08_anova_analysis.png")

# =============================================================================
# SECTION 6 — TIME SERIES ANALYSIS (ARIMA)
# =============================================================================

def run_arima(precip: pd.DataFrame) -> dict:
    print("\n" + "="*65)
    print("SECTION 6 — TIME SERIES ANALYSIS (ARIMA)")
    print("="*65)

    arima_models = {}

    # ── 6a. ACF / PACF overview ───────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for idx, (stn, info) in enumerate(STATIONS.items()):
        ts = precip[stn].dropna()
        plot_acf(ts,  lags=36, ax=axes[idx, 0], color=info["color"])
        axes[idx, 0].set_title(f"{stn} — ACF", fontweight="bold")
        plot_pacf(ts, lags=36, ax=axes[idx, 1], color=info["color"])
        axes[idx, 1].set_title(f"{stn} — PACF", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/09_acf_pacf.png", bbox_inches="tight")
    plt.close()
    print("[Saved] 09_acf_pacf.png")

    # ── 6b. ADF test + ARIMA selection + forecast ─────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(15, 14))

    for idx, (stn, info) in enumerate(STATIONS.items()):
        print(f"\n--- {stn} ---")
        ts = precip[stn].dropna()

        # Stationarity
        adf_s, adf_p, *_ = adfuller(ts, autolag="AIC")
        print(f"  ADF Test: stat={adf_s:.4f}, p={adf_p:.4f}"
              f"  [{'Stationary' if adf_p < 0.05 else 'Non-stationary'}]")

        # Grid search ARIMA(p,d,q) — keep small to avoid long runtime
        best_aic   = np.inf
        best_order = (1, 0, 1)
        for p in range(0, 4):
            for d in range(0, 2):
                for q in range(0, 4):
                    try:
                        m = ARIMA(ts, order=(p, d, q)).fit()
                        if m.aic < best_aic:
                            best_aic, best_order = m.aic, (p, d, q)
                    except Exception:
                        pass

        print(f"  Best order: ARIMA{best_order}  AIC={best_aic:.2f}")
        fit = ARIMA(ts, order=best_order).fit()

        # Ljung-Box residual test
        lb = acorr_ljungbox(fit.resid, lags=[12, 24], return_df=True)
        print(f"  Ljung-Box (lag 12): Q={lb['lb_stat'].iloc[0]:.3f},"
              f" p={lb['lb_pvalue'].iloc[0]:.4f}")

        # Forecast 24 months
        fc   = fit.get_forecast(steps=24)
        fcm  = fc.predicted_mean
        fcci = fc.conf_int()

        arima_models[stn] = {"order": best_order, "aic": best_aic,
                              "model": fit, "forecast": fcm, "ci": fcci}

        # Plot
        ax = axes[idx]
        split = len(ts) - 36
        ax.plot(ts.index[:split],  ts.values[:split],
                color="lightgray", lw=0.8, label="Train")
        ax.plot(ts.index[split:],  ts.values[split:],
                color=info["color"], lw=1.4, label="Test (last 3 yr)")
        ax.plot(fcm.index, fcm.values,
                color="black", lw=2, ls="--", label="Forecast 2024–25")
        ax.fill_between(fcci.index, fcci.iloc[:, 0], fcci.iloc[:, 1],
                        color="black", alpha=0.12, label="95% CI")
        ax.set_title(f"{stn} — ARIMA{best_order} Forecast",
                     fontweight="bold")
        ax.set_ylabel("Precipitation (mm)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(f"{OUT}/10_arima_forecast.png", bbox_inches="tight")
    plt.close()
    print("[Saved] 10_arima_forecast.png")

    # ── 6c. Residual diagnostics for each model ───────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for idx, (stn, info) in enumerate(STATIONS.items()):
        fit = arima_models[stn]["model"]
        resid = fit.resid.dropna()
        ax_r, ax_q = axes[idx, 0], axes[idx, 1]
        ax_r.plot(resid.index, resid.values, color=info["color"], lw=0.8)
        ax_r.axhline(0, color="black", lw=0.6, ls="--")
        ax_r.set_title(f"{stn} — ARIMA Residuals", fontweight="bold")
        ax_r.set_ylabel("Residual (mm)")
        stats.probplot(resid, dist="norm", plot=ax_q)
        ax_q.set_title(f"{stn} — Residual Q-Q Plot", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/11_arima_residuals.png", bbox_inches="tight")
    plt.close()
    print("[Saved] 11_arima_residuals.png")

    return arima_models

# =============================================================================
# SECTION 7 — MULTIPLE REGRESSION
# =============================================================================

def run_regression(precip: pd.DataFrame, temp: pd.DataFrame, spi: dict) -> dict:
    print("\n" + "="*65)
    print("SECTION 7 — MULTIPLE REGRESSION")
    print("="*65)

    reg_results = {}
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    for idx, (stn, info) in enumerate(STATIONS.items()):
        print(f"\n--- {stn} ---")
        ts  = precip[stn].dropna()
        tmp = temp[stn].dropna()
        s12 = spi[stn]["SPI-12"].dropna()
        s3  = spi[stn]["SPI-3"].dropna()

        # Build feature matrix
        df_r = pd.DataFrame({
            "SPI12":       s12,
            "Precip":      ts,
            "Precip_L1":   ts.shift(1),
            "Precip_L2":   ts.shift(2),
            "Precip_L3":   ts.shift(3),
            "Precip_L6":   ts.shift(6),
            "SPI3":        s3,
            "Temp":        tmp,
            "Temp_L1":     tmp.shift(1),
            "Month_sin":   np.sin(2 * np.pi * ts.index.month / 12),
            "Month_cos":   np.cos(2 * np.pi * ts.index.month / 12),
        }).dropna()

        X_cols = ["Precip", "Precip_L1", "Precip_L2", "Precip_L3",
                  "Precip_L6", "SPI3", "Temp", "Temp_L1",
                  "Month_sin", "Month_cos"]

        X = add_constant(df_r[X_cols])
        y = df_r["SPI12"]
        model = OLS(y, X).fit()

        print(f"  R²={model.rsquared:.4f}  Adj-R²={model.rsquared_adj:.4f}"
              f"  F={model.fvalue:.2f}  p={model.f_pvalue:.2e}")
        print(f"  {'Variable':<14} {'Coef':>9} {'Std Err':>9} "
              f"{'t':>8} {'p':>10}  Sig?")
        print("  " + "-"*58)
        for v in model.params.index:
            sig = "*" if model.pvalues[v] < 0.05 else ""
            print(f"  {v:<14} {model.params[v]:>9.4f}"
                  f" {model.bse[v]:>9.4f}"
                  f" {model.tvalues[v]:>8.3f}"
                  f" {model.pvalues[v]:>10.4f}  {sig}")

        # VIF
        vif = pd.DataFrame({
            "Variable": X_cols,
            "VIF": [variance_inflation_factor(df_r[X_cols].values, i)
                    for i in range(len(X_cols))]
        })
        print(f"\n  VIF (>10 = multicollinearity concern):")
        print(vif.to_string(index=False))

        # Durbin-Watson
        dw = durbin_watson(model.resid)
        print(f"\n  Durbin-Watson: {dw:.4f}  "
              f"[{'No autocorrelation' if 1.5 < dw < 2.5 else 'Potential autocorrelation'}]")

        reg_results[stn] = model

        # Plots
        ax1, ax2 = axes[idx, 0], axes[idx, 1]
        ax1.scatter(model.fittedvalues, y, color=info["color"], s=12, alpha=0.4)
        lo = min(model.fittedvalues.min(), y.min())
        hi = max(model.fittedvalues.max(), y.max())
        ax1.plot([lo, hi], [lo, hi], "k--", lw=1.5)
        ax1.set_title(f"{stn} — Actual vs Fitted SPI-12\nR²={model.rsquared:.3f}",
                      fontweight="bold")
        ax1.set_xlabel("Fitted")
        ax1.set_ylabel("Actual SPI-12")

        ax2.scatter(model.fittedvalues, model.resid, color=info["color"],
                    s=12, alpha=0.4)
        ax2.axhline(0, color="black", lw=0.8, ls="--")
        ax2.set_title(f"{stn} — Residuals vs Fitted", fontweight="bold")
        ax2.set_xlabel("Fitted")
        ax2.set_ylabel("Residuals")

    plt.tight_layout()
    plt.savefig(f"{OUT}/12_regression_diagnostics.png", bbox_inches="tight")
    plt.close()
    print("\n[Saved] 12_regression_diagnostics.png")
    return reg_results

# =============================================================================
# SECTION 8 — NONPARAMETRIC STATISTICS
# =============================================================================

def _mann_kendall(x: np.ndarray):
    """
    Two-sided Mann-Kendall trend test.
    Returns (S, Z, p_value, Kendall_tau, Sen_slope).
    """
    n = len(x)
    s = sum(np.sign(x[j] - x[i]) for i in range(n - 1) for j in range(i + 1, n))
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    z = (s - np.sign(s)) / np.sqrt(var_s) if s != 0 else 0.0
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))
    slopes = [(x[j] - x[i]) / (j - i)
              for i in range(n - 1) for j in range(i + 1, n)]
    sen = np.median(slopes)
    return s, z, p, tau, sen


def run_nonparametric(precip: pd.DataFrame, spi: dict) -> dict:
    print("\n" + "="*65)
    print("SECTION 8 — NONPARAMETRIC STATISTICS")
    print("="*65)

    stations_list = list(STATIONS)
    annual = precip.resample("YS").sum()

    # ── 8.1 Mann-Kendall on annual precipitation ──────────────────────────────
    print("\n8.1 Mann-Kendall Trend Test — Annual Precipitation")
    print(f"  {'Station':<12} {'S':>6} {'Z':>8} {'p':>9}"
          f" {'τ':>8} {'Sen Slope (mm/yr)':>18}  Decision")
    print("  " + "-"*72)
    mk_res = {}
    for stn, info in STATIONS.items():
        x = annual[stn].dropna().values
        s, z, p, tau, sen = _mann_kendall(x)
        mk_res[stn] = dict(S=s, Z=z, p=p, tau=tau, sen=sen)
        trend   = "Increasing" if z > 0 else "Decreasing"
        sig     = "**SIG**" if p < 0.05 else "ns"
        print(f"  {stn:<12} {s:>6} {z:>8.4f} {p:>9.4f}"
              f" {tau:>8.4f} {sen:>18.3f}  {trend} ({sig})")

    # ── 8.2 Mann-Kendall on annual SPI-12 ────────────────────────────────────
    print("\n8.2 Mann-Kendall Trend Test — Annual Mean SPI-12")
    print(f"  {'Station':<12} {'Z':>8} {'p':>9}  Decision")
    print("  " + "-"*42)
    for stn in stations_list:
        annual_spi = spi[stn]["SPI-12"].resample("YS").mean().dropna().values
        _, z, p, _, _ = _mann_kendall(annual_spi)
        trend = "Drying" if z < 0 else "Wetting"
        sig   = "**SIG**" if p < 0.05 else "ns"
        print(f"  {stn:<12} {z:>8.4f} {p:>9.4f}  {trend} ({sig})")

    # ── 8.3 Kruskal-Wallis across stations (SPI-12) ──────────────────────────
    spi12_groups = [spi[s]["SPI-12"].dropna().values for s in stations_list]
    kw_h, kw_p = stats.kruskal(*spi12_groups)
    print(f"\n8.3 Kruskal-Wallis Test — SPI-12 Across Stations")
    print(f"  H = {kw_h:.4f},  p = {kw_p:.6f}")
    print(f"  → {'Significant' if kw_p < 0.05 else 'No significant'}"
          f" difference in drought severity distribution")

    # ── 8.4 Pairwise Mann-Whitney U ───────────────────────────────────────────
    print("\n8.4 Pairwise Mann-Whitney U Tests — SPI-12")
    for i in range(len(stations_list)):
        for j in range(i + 1, len(stations_list)):
            u, p = stats.mannwhitneyu(spi12_groups[i], spi12_groups[j],
                                      alternative="two-sided")
            print(f"  {stations_list[i]} vs {stations_list[j]}:"
                  f" U={u:.0f},  p={p:.4f}  {'*' if p < 0.05 else 'ns'}")

    # ── 8.5 Drought frequency & severity table ────────────────────────────────
    print("\n8.5 Drought Classification (SPI-12 Categories, WMO)")
    drought_summary = {}
    for stn in stations_list:
        s12 = spi[stn]["SPI-12"].dropna()
        n   = len(s12)
        d   = {
            "Total months":       n,
            "Extreme Drought":    int((s12 < -2.0).sum()),
            "Severe Drought":     int(((s12 >= -2.0) & (s12 < -1.5)).sum()),
            "Moderate Drought":   int(((s12 >= -1.5) & (s12 < -1.0)).sum()),
            "Near Normal":        int(((s12 >= -1.0) & (s12 < 1.0)).sum()),
            "Moderately Wet":     int(((s12 >= 1.0)  & (s12 < 1.5)).sum()),
            "Very Wet":           int(((s12 >= 1.5)  & (s12 < 2.0)).sum()),
            "Extreme Wet":        int((s12 >= 2.0).sum()),
            "Drought freq (%)":   round(100 * (s12 < -1.0).sum() / n, 1),
            "Mean drought SPI":   round(s12[s12 < -1.0].mean(), 3) if (s12 < -1.0).any() else np.nan,
        }
        drought_summary[stn] = d
    df_drought = pd.DataFrame(drought_summary).T
    print(df_drought.to_string())

    # ── Visualizations ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Annual precipitation + Sen's slope
    ax = axes[0, 0]
    for stn, info in STATIONS.items():
        ap   = annual[stn].dropna()
        yrs  = np.arange(len(ap))
        sen  = mk_res[stn]["sen"]
        mid_y = np.median(ap.values)
        mid_x = np.median(yrs)
        trend_line = sen * yrs + (mid_y - sen * mid_x)
        ax.plot(ap.index.year, ap.values, color=info["color"], lw=1.5,
                alpha=0.8, label=stn)
        ax.plot(ap.index.year, trend_line, color=info["color"], lw=2,
                ls="--")
    ax.set_title("Annual Precipitation + Sen's Slope (Mann-Kendall)",
                 fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Precipitation (mm)")
    ax.legend()

    # (b) Drought severity counts
    ax = axes[0, 1]
    cats = ["Moderate Drought", "Severe Drought", "Extreme Drought"]
    x_pos = np.arange(len(cats))
    w = 0.25
    for k, (stn, info) in enumerate(STATIONS.items()):
        vals = [drought_summary[stn][c] for c in cats]
        ax.bar(x_pos + k * w, vals, width=w, color=info["color"],
               edgecolor="black", lw=0.5, label=stn)
    ax.set_xticks(x_pos + w)
    ax.set_xticklabels(["Moderate\n(-1.5 to -1)", "Severe\n(-2 to -1.5)",
                         "Extreme\n(< -2)"], fontsize=8)
    ax.set_title("Drought Event Counts by Category (SPI-12)", fontweight="bold")
    ax.set_ylabel("Number of Months")
    ax.legend()

    # (c) Drought frequency comparison
    ax = axes[1, 0]
    freqs  = [drought_summary[s]["Drought freq (%)"] for s in stations_list]
    colors = [STATIONS[s]["color"] for s in stations_list]
    bars   = ax.bar(stations_list, freqs, color=colors, edgecolor="black", lw=0.8)
    for b, f in zip(bars, freqs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                f"{f:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Drought Frequency (% months SPI-12 < −1.0)", fontweight="bold")
    ax.set_ylabel("Drought Frequency (%)")
    ax.set_ylim(0, max(freqs) * 1.25)

    # (d) Vulnerability composite radar-style bar chart
    ax = axes[1, 1]
    metrics = {
        "Drought\nFreq (%)": {s: drought_summary[s]["Drought freq (%)"] for s in stations_list},
        "Extreme\nDrought":  {s: drought_summary[s]["Extreme Drought"]  for s in stations_list},
        "Severe\nDrought":   {s: drought_summary[s]["Severe Drought"]   for s in stations_list},
    }
    m_names = list(metrics)
    x2 = np.arange(len(m_names))
    for k, (stn, info) in enumerate(STATIONS.items()):
        vals = [metrics[m][stn] for m in m_names]
        # Normalize to 0–1 for comparison
        maxv = [max(metrics[m][s] for s in stations_list) for m in m_names]
        norm_vals = [v / mx if mx > 0 else 0 for v, mx in zip(vals, maxv)]
        ax.bar(x2 + k * 0.25, norm_vals, width=0.25,
               color=info["color"], edgecolor="black", lw=0.5, label=stn)
    ax.set_xticks(x2 + 0.25)
    ax.set_xticklabels(m_names, fontsize=9)
    ax.set_title("Normalized Drought Vulnerability Scores", fontweight="bold")
    ax.set_ylabel("Score (0 = least, 1 = most vulnerable)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{OUT}/13_nonparametric_analysis.png", bbox_inches="tight")
    plt.close()
    print("\n[Saved] 13_nonparametric_analysis.png")

    return drought_summary

# =============================================================================
# SECTION 9 — DROUGHT VULNERABILITY INDEX & FINAL RANKING
# =============================================================================

def compute_vulnerability_index(spi: dict, drought_summary: dict) -> None:
    print("\n" + "="*65)
    print("SECTION 9 — DROUGHT VULNERABILITY INDEX")
    print("="*65)

    stations_list = list(STATIONS)
    records = []

    for stn in stations_list:
        s12   = spi[stn]["SPI-12"].dropna()
        ds    = drought_summary[stn]
        rec = {
            "Station":         stn,
            "Zone":            STATIONS[stn]["zone"],
            "Drought Freq %":  ds["Drought freq (%)"],
            "Mean SPI (drought)": ds["Mean drought SPI"],
            "Extreme Events":  ds["Extreme Drought"],
            "Longest Drought": _longest_run(s12 < -1.0),
            "SPI Trend Z":     None,
        }
        # Mann-Kendall on SPI-12
        annual_spi = spi[stn]["SPI-12"].resample("YS").mean().dropna().values
        _, z, _, _, _ = _mann_kendall(annual_spi)
        rec["SPI Trend Z"] = round(z, 4)
        records.append(rec)

    df_vi = pd.DataFrame(records).set_index("Station")
    print(df_vi.to_string())

    # Rank: most negative SPI trend Z, highest drought freq, most extreme events
    df_vi["Rank Score"] = (
        df_vi["Drought Freq %"].rank(ascending=False) +
        (-df_vi["Mean SPI (drought)"].fillna(0)).rank(ascending=False) +
        df_vi["Extreme Events"].rank(ascending=False) +
        df_vi["Longest Drought"].rank(ascending=False)
    )
    df_vi["Vulnerability Rank"] = df_vi["Rank Score"].rank().astype(int)
    most_vulnerable = df_vi["Vulnerability Rank"].idxmin()
    print(f"\n  Most Drought-Vulnerable Station: {most_vulnerable}")
    print(df_vi[["Zone", "Drought Freq %", "Extreme Events",
                 "Longest Drought", "Vulnerability Rank"]].to_string())

    return df_vi


def _longest_run(bool_series: pd.Series) -> int:
    """Length of the longest consecutive True run."""
    max_run = curr = 0
    for val in bool_series:
        curr = curr + 1 if val else 0
        max_run = max(max_run, curr)
    return max_run

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("STAT-222 SEMESTER PROJECT")
    print("Climate Variability, Drought Risk & Precipitation Dynamics")
    print("Pakistan — Karachi | Lahore | Islamabad  (1990–2023)")
    print("=" * 65)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("\n[1/6] Downloading data from Open-Meteo Archive API...")
    precip, temp = load_monthly_data()
    precip.to_csv(f"{OUT}/raw_monthly_precip.csv")
    temp.to_csv(f"{OUT}/raw_monthly_temp.csv")
    print(f"  Records: {len(precip)} months  "
          f"({precip.index[0].year}–{precip.index[-1].year})")

    # ── 2. SPI ───────────────────────────────────────────────────────────────
    print("\n[2/6] Computing SPI (3-, 6-, 12-month scales)...")
    spi = compute_all_spi(precip)

    # ── 3. EDA ───────────────────────────────────────────────────────────────
    print("\n[3/6] Running Exploratory Data Analysis...")
    run_eda(precip, temp, spi)

    # ── 4. Distribution Fitting ───────────────────────────────────────────────
    print("\n[4/6] Fitting probability distributions...")
    dist_results = run_distribution_fitting(precip)

    # ── 5. ANOVA ─────────────────────────────────────────────────────────────
    print("\n[5/6] Running ANOVA...")
    run_anova(precip, spi)

    # ── 6. ARIMA ─────────────────────────────────────────────────────────────
    print("\n[6a/6] Fitting ARIMA models & forecasting...")
    arima_models = run_arima(precip)

    # ── 7. Regression ────────────────────────────────────────────────────────
    print("\n[6b/6] Multiple regression...")
    reg_results = run_regression(precip, temp, spi)

    # ── 8. Nonparametric ─────────────────────────────────────────────────────
    print("\n[6c/6] Nonparametric analysis...")
    drought_summary = run_nonparametric(precip, spi)

    # ── 9. Vulnerability Index ───────────────────────────────────────────────
    compute_vulnerability_index(spi, drought_summary)

    print("\n" + "=" * 65)
    print("COMPLETE — All outputs saved to:", OUT)
    print("=" * 65)


if __name__ == "__main__":
    main()
