# app.py
import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import altair as alt

st.set_page_config("Credit Risk Analytics Lab", layout="wide", page_icon="📊")
st.title("Interactive Credit Risk Analytics Lab — Fixed")

# ------------------------
# Helpers
# ------------------------
def rag_color(value, thresholds=(0.1, 0.25), reverse=False):
    if reverse:
        if value < thresholds[0]: return "normal"
        elif value < thresholds[1]: return "warning"
        else: return "critical"
    else:
        if value < thresholds[0]: return "critical"
        elif value < thresholds[1]: return "warning"
        else: return "normal"

def compute_vif(df):
    X = add_constant(df)
    vifs = [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]
    return pd.DataFrame({"feature": df.columns, "VIF": vifs})

def bin_edges(x, bins=10, method="quantile"):
    if method == "quantile":
        edges = np.unique(np.quantile(x, np.linspace(0, 1, bins)))
    else:
        edges = np.linspace(np.min(x), np.max(x), bins + 1)
    # ensure at least 2 edges
    if len(edges) < 2:
        edges = np.array([np.min(x), np.max(x)])
    return np.unique(edges)

def compute_woe_iv(series, target, bins=10, method="quantile", eps=1e-6):
    x = series.values
    y = target.values
    edges = bin_edges(x, bins=bins, method=method)
    cats = pd.cut(x, bins=edges, include_lowest=True)
    grouped = pd.DataFrame({"bin": cats, "y": y}).groupby("bin")["y"].agg(["sum","count"]).reset_index()
    grouped = grouped.rename(columns={"sum":"bad","count":"total"})
    grouped["good"] = grouped["total"] - grouped["bad"]
    total_bad = grouped["bad"].sum()
    total_good = grouped["good"].sum()
    # avoid divide by zero
    grouped["bad_rate"] = (grouped["bad"] + eps) / (total_bad + eps)
    grouped["good_rate"] = (grouped["good"] + eps) / (total_good + eps)
    grouped["WoE"] = np.log(grouped["good_rate"] / grouped["bad_rate"])
    grouped["IV"] = (grouped["good_rate"] - grouped["bad_rate"]) * grouped["WoE"]
    grouped["bin_str"] = grouped["bin"].astype(str)
    return grouped.reset_index(drop=True), edges

def compute_psi(expected, actual, bins=10, method="quantile", eps=1e-6):
    edges = bin_edges(expected, bins=bins, method=method)
    e_cat = pd.cut(expected, bins=edges, include_lowest=True)
    a_cat = pd.cut(actual, bins=edges, include_lowest=True)
    e_dist = e_cat.value_counts().sort_index()
    a_dist = a_cat.value_counts().sort_index()
    e_pct = (e_dist + eps) / (e_dist.sum() + eps)
    a_pct = (a_dist + eps) / (a_dist.sum() + eps)
    psi = ((a_pct - e_pct) * np.log(a_pct / e_pct)).sum()
    psi_df = pd.DataFrame({
        "bin": e_dist.index.astype(str),
        "expected_pct": e_pct.values,
        "actual_pct": a_pct.values,
        "contrib": np.abs(a_pct.values - e_pct.values)
    })
    return psi_df, psi, edges

# ------------------------
# Sidebar dataset loader (single upload)
# ------------------------
st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"], key="sidebar_csv")
if uploaded is not None:
    try:
        df_all = pd.read_csv(uploaded)
        st.sidebar.success(f"Loaded {df_all.shape[0]} rows × {df_all.shape[1]} cols")
    except Exception as e:
        st.sidebar.error("Failed to read CSV: " + str(e))
        st.stop()
else:
    rng = np.random.default_rng(42)
    n_demo = 1000
    df_all = pd.DataFrame({
        "income": rng.normal(5000, 1500, n_demo).clip(500, 20000),
        "age": rng.normal(40, 10, n_demo).clip(18, 90),
        "debt_to_income": rng.beta(2,5, n_demo),
        "utilization": rng.beta(2,5, n_demo),
        "num_delinq": rng.poisson(0.8, n_demo),
        "emp_length": rng.normal(5,3,n_demo).clip(0,40),
        "credit_score": rng.normal(700,50,n_demo).clip(300,850)
    })
    p = 1/(1 + np.exp(-5*(df_all["debt_to_income"] - 0.2)))
    df_all["default"] = (rng.uniform(size=n_demo) < p).astype(int)
    st.sidebar.info("Using synthetic demo dataset (upload CSV to analyze your own)")

with st.sidebar.expander("Preview dataset", expanded=True):
    st.write("Shape:", df_all.shape)
    st.dataframe(df_all.head(5))
    st.write("Column types:")
    st.write(df_all.dtypes.apply(lambda x: x.name))

numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    st.error("No numeric columns available in dataset. Upload a numeric CSV.")
    st.stop()

# ------------------------
# Tabs
# ------------------------
tabs = st.tabs(["Correlation", "Multicollinearity", "VIF", "WoE (multi)", "PSI (multi)"])

# ------------------------
# Tab 1: Correlation
# ------------------------
with tabs[0]:
    st.subheader("Correlation — quick diagnostics")
    presets = {"Debt-to-Income vs Default":"debt_to_income|default",
               "Utilization vs Default":"utilization|default",
               "Income vs Utilization":"income|utilization",
               "Credit score vs Default":"credit_score|default"}
    preset = st.selectbox("Preset (demo)", list(presets.keys()))
    x_default, y_default = presets[preset].split("|")
    x_col = st.selectbox("X column", numeric_cols, index=numeric_cols.index(x_default) if x_default in numeric_cols else 0)
    y_col = st.selectbox("Y column", numeric_cols, index=numeric_cols.index(y_default) if y_default in numeric_cols else 1)
    data = df_all[[x_col, y_col]].dropna()
    if len(data) < 3:
        st.warning("Not enough data for correlation.")
    else:
        r, pval = stats.pearsonr(data[x_col], data[y_col])
        rho, p_s = stats.spearmanr(data[x_col], data[y_col])
        c1, c2, c3 = st.columns([1,1,1])
        c1.metric("Pearson r", f"{r:.3f}", help=f"p={pval:.3g}")
        c2.metric("Spearman ρ", f"{rho:.3f}", help=f"p={p_s:.3g}")
        c3.write(f"N = {len(data)}")

        with st.expander("Formulas & notes"):
            st.latex(r"r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2}\sqrt{\sum (y_i - \bar{y})^2}}")
            st.write("Pearson measures linear association; Spearman measures monotonic rank association.")

        show_line = st.checkbox("Show OLS fit line", value=True)
        base = alt.Chart(data.reset_index()).mark_circle(size=40, opacity=0.6).encode(
            x=alt.X(x_col, title=x_col), y=alt.Y(y_col, title=y_col), tooltip=[x_col, y_col]
        ).interactive()
        if show_line:
            lr = np.polyfit(data[x_col], data[y_col], 1)
            data["_yhat"] = np.polyval(lr, data[x_col])
            base = base + alt.Chart(data).mark_line(color="red").encode(x=x_col, y="_yhat")
        st.altair_chart(base, use_container_width=True)

        # story
        if abs(r) > 0.7:
            st.success(f"Strong correlation (r={r:.2f}) — variables move together. Could indicate proxying in models.")
        elif abs(r) > 0.3:
            st.info(f"Moderate correlation (r={r:.2f}) — meaningful but check stability.")
        else:
            st.warning(f"Weak correlation (r={r:.2f}) — look for non-linear patterns or interactions.")

# ------------------------
# Tab 2: Multicollinearity
# ------------------------
with tabs[1]:
    st.subheader("Multicollinearity — correlation matrix")
    predictors = st.multiselect("Select predictors (2+)", numeric_cols, default=numeric_cols[:4])
    target_col = st.selectbox("Target (optional)", numeric_cols, index=numeric_cols.index("default") if "default" in numeric_cols else 0)
    if len(predictors) < 2:
        st.warning("Select at least two predictors.")
    else:
        dfm = df_all[[target_col] + predictors].dropna()
        X = dfm[predictors]; y = dfm[target_col]
        lin = LinearRegression().fit(X, y)
        coef_df = pd.DataFrame({"feature": predictors, "coef": lin.coef_})
        st.table(coef_df.style.format({"coef":"{:.4f}"}))

        corr = dfm[predictors].corr()
        corr_m = corr.reset_index().melt("index")
        heat = alt.Chart(corr_m).mark_rect().encode(
            x=alt.X("index:N", title=""), y=alt.Y("variable:N", title=""),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue"), title="corr"),
            tooltip=[alt.Tooltip("value:Q", format=".3f"), "index", "variable"]
        )
        st.altair_chart(heat, use_container_width=True)

        # story
        high_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > 0.8:
                    high_pairs.append((corr.index[i], corr.columns[j], corr.iloc[i, j]))
        if high_pairs:
            st.error("High multicollinearity detected:")
            for a,b,v in high_pairs:
                st.write(f"- {a} ↔ {b} (corr={v:.2f}): consider aggregation/removal.")
        else:
            st.success("No extreme multicollinearity detected (|corr| > 0.8).")

# ------------------------
# Tab 3: VIF
# ------------------------
with tabs[2]:
    st.subheader("VIF diagnostics")
    preds_vif = st.multiselect("Select predictors for VIF", numeric_cols, default=numeric_cols[:4])
    if len(preds_vif) < 2:
        st.warning("Pick at least two predictors.")
    else:
        vif_df = compute_vif(df_all[preds_vif].dropna())
        vif_df["RAG"] = vif_df["VIF"].apply(lambda v: rag_color(v, (5, 10)))
        st.dataframe(vif_df.style.format({"VIF":"{:.3f}"}), height=240)
        chart = alt.Chart(vif_df).mark_bar().encode(
            x="feature:N", y="VIF:Q",
            color=alt.Color("VIF:Q", scale=alt.Scale(domain=[0,5,10, max(12, vif_df['VIF'].max())],
                                                     range=["green","yellow","red","darkred"]))
        )
        st.altair_chart(chart, use_container_width=True)
        with st.expander("Formula"):
            st.latex(r"VIF_j = \frac{1}{1 - R_j^2}")
            st.write("VIF >5 moderate; >10 severe. Consider feature engineering if high.")

# ------------------------
# Tab 4: WoE (multi-feature)
# ------------------------
with tabs[3]:
    st.subheader("WoE & IV — multi-feature")
    bin_methods = st.radio("Binning method", ("quantile", "equal_width", "both"), horizontal=True)
    target = st.selectbox("Binary target (0=good,1=bad)", numeric_cols, index=numeric_cols.index("default") if "default" in numeric_cols else 0)
    features = st.multiselect("Select features", [c for c in numeric_cols if c != target], default=[c for c in numeric_cols if c != target][:2])
    bins = st.slider("Bins per feature", 3, 20, value=10)

    if not features:
        st.warning("Select at least one feature.")
    else:
        for feat in features:
            st.markdown(f"**Feature: {feat}**")
            df_sub = df_all[[feat, target]].dropna()

            def render_woe(method):
                wdf, edges = compute_woe_iv(df_sub[feat], df_sub[target], bins=bins, method=method)
                if wdf.empty:
                    st.write(f"No bins created for method {method}.")
                    return
                wdf["IV_contrib"] = np.abs(wdf["IV"])
                IV_total = wdf["IV"].sum()
                st.write(f"Method **{method}** — IV = **{IV_total:.3f}**")
                st.dataframe(wdf[["bin_str","bad","good","total","WoE","IV"]].rename(columns={"bin_str":"bin"}), height=220)
                c = alt.Chart(wdf.reset_index()).mark_bar().encode(
                    x=alt.X("index:N", title="bin"),
                    y=alt.Y("WoE:Q"),
                    color=alt.Color("IV_contrib:Q", scale=alt.Scale(scheme="reds"), title="IV contribution"),
                    tooltip=["bin_str","bad","good","total","WoE","IV"]
                ).properties(height=220)
                st.altair_chart(c, use_container_width=True)
                major = wdf[wdf["IV_contrib"] > 0.02]
                if not major.empty:
                    st.markdown("Major contributing bins (IV contribution > 0.02):")
                    for _, row in major.iterrows():
                        st.write(f"- {row['bin_str']}: N={int(row['total'])}, WoE={row['WoE']:.2f}, IV_contrib={row['IV']:.3f} — likely higher/lower risk segment.")
                else:
                    st.write("No single bin strongly dominates IV.")

            if bin_methods in ("quantile", "both"):
                render_woe("quantile")
            if bin_methods in ("equal_width", "both"):
                render_woe("equal_width")

        st.markdown("**Guidance:** IV < 0.02 weak, 0.02–0.1 medium, >0.1 strong. Investigate bins that drive IV for business meaning.")

# ------------------------
# Tab 5: PSI (multi-feature) — fixed scoping
# ------------------------
with tabs[4]:
    st.subheader("PSI — multi-feature population stability")
    psi_methods = st.radio("Binning method", ("quantile", "equal_width", "both"), horizontal=True)
    st.markdown("Pick paired columns: Expected (reference) and Actual (new/monitoring)")
    expected_cols = st.multiselect("Expected columns (reference)", numeric_cols, default=numeric_cols[:2])
    actual_cols = st.multiselect("Actual columns (monitoring)", numeric_cols, default=numeric_cols[:2])
    bins_psi = st.slider("Bins for PSI", 5, 20, value=10)

    if expected_cols and actual_cols:
        if len(expected_cols) != len(actual_cols):
            st.warning("Pick equal number of expected and actual columns (paired).")
        else:
            for ecol, acol in zip(expected_cols, actual_cols):
                st.markdown(f"**Pair: {ecol} (expected)  —  {acol} (actual)**")
                exp = df_all[ecol].dropna()
                act = df_all[acol].dropna()

                def render_psi(method):
                    psi_df, psi_val, edges = compute_psi(exp, act, bins=bins_psi, method=method)
                    st.write(f"Method **{method}** — PSI = **{psi_val:.3f}**")
                    st.dataframe(psi_df, height=220)
                    chart_df = pd.DataFrame({
                        "bin": list(psi_df["bin"]) + list(psi_df["bin"]),
                        "pct": list(psi_df["expected_pct"]) + list(psi_df["actual_pct"]),
                        "group": ["expected"]*len(psi_df) + ["actual"]*len(psi_df),
                        "contrib": list(psi_df["contrib"]) + list(psi_df["contrib"])
                    })
                    ch = alt.Chart(chart_df).mark_bar(opacity=0.75).encode(
                        x=alt.X("bin:N", sort=None),
                        y="pct:Q",
                        color=alt.Color("contrib:Q", scale=alt.Scale(scheme="reds"), title="PSI contribution"),
                        column=alt.Column("group:N", header=alt.Header(labelOrient="bottom")),
                        tooltip=["bin","group","pct","contrib"]
                    ).properties(height=200)
                    st.altair_chart(ch, use_container_width=True)

                    big = psi_df[psi_df["contrib"] > (psi_df["contrib"].mean() + psi_df["contrib"].std())]
                    if not big.empty:
                        st.markdown("Major contributing bins ( > 1σ ):")
                        for _, r in big.iterrows():
                            st.write(f"- {r['bin']}: expected {r['expected_pct']:.2f}, actual {r['actual_pct']:.2f}, contrib {r['contrib']:.3f}")
                    else:
                        st.write("No per-bin shifts exceeding 1σ from mean contribution.")

                    # story & guidance per method (moved inside render to avoid scope issues)
                    st.markdown("**Story & guidance (this method):**")
                    if psi_val > 0.25:
                        st.error("PSI > 0.25: Significant shift — investigate root causes and consider retraining.")
                    elif psi_val > 0.1:
                        st.warning("PSI 0.1–0.25: Moderate shift — monitor and diagnose drivers.")
                    else:
                        st.success("PSI < 0.1: Minimal shift — distribution stable for this feature.")

                if psi_methods in ("quantile", "both"):
                    render_psi("quantile")
                if psi_methods in ("equal_width", "both"):
                    render_psi("equal_width")
    else:
        st.info("Select paired expected and actual columns (same length).")

# ------------------------
# Regression diagnostics (diagnostics-only)
# ------------------------
st.markdown("---")
st.header("Sample Regression Diagnostics (diagnostics only)")
diag_target = st.selectbox("Diagnostics target (binary)", numeric_cols, index=numeric_cols.index("default") if "default" in numeric_cols else 0)
diag_preds = st.multiselect("Diagnostics predictors", [c for c in numeric_cols if c != diag_target], default=[c for c in numeric_cols if c != diag_target][:3])

if diag_preds:
    df_diag = df_all[[diag_target] + diag_preds].dropna()
    y = df_diag[diag_target].astype(int)
    X = add_constant(df_diag[diag_preds])
    try:
        model = Logit(y, X).fit(disp=False)
        summ = model.summary2().tables[1].reset_index().rename(columns={"index":"feature"})
        summ = summ[["feature","Coef.","Std.Err.","z","P>|z|","[0.025","0.975]"]]
        summ.columns = ["feature","coef","std_err","z_or_t","p_value","ci_lower","ci_upper"]
        st.dataframe(summ.style.format({
            "coef":"{:.4f}","std_err":"{:.4f}","z_or_t":"{:.3f}","p_value":"{:.3g}","ci_lower":"{:.4f}","ci_upper":"{:.4f}"
        }), height=260)
        st.markdown("**Diagnostics story:**")
        signif = summ[summ["p_value"] < 0.05]
        if not signif.empty:
            st.write(f"Significant predictors (p < 0.05): {', '.join(signif['feature'].tolist())}")
        else:
            st.write("No predictors significant at p < 0.05 in this sample.")
    except Exception as e:
        st.error("Regression diagnostics failed: " + str(e))
else:
    st.info("Select at least one predictor for diagnostics.")

st.caption("Educational tool — diagnostics only (no model saving).")
