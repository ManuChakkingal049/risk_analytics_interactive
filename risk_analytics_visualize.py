import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import altair as alt

st.set_page_config(page_title="Interactive Credit Risk Analytics Lab", layout="wide")

# -----------------------------
# Helper Functions
# -----------------------------
def rag_color(value, thresholds=(0.1, 0.25), reverse=False):
    if reverse:
        if value < thresholds[0]: return "green"
        elif value < thresholds[1]: return "amber"
        else: return "red"
    else:
        if value < thresholds[0]: return "red"
        elif value < thresholds[1]: return "amber"
        else: return "green"

def compute_vif_matrix(df):
    X = sm.add_constant(df)
    vif = [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]
    return pd.DataFrame({"feature": df.columns, "VIF": vif})

def woe_psi_binning(x, bins=10, method="quantile"):
    if method=="quantile":
        edges = np.unique(np.quantile(x, q=np.linspace(0,1,bins)))
    else:
        edges = np.linspace(np.min(x), np.max(x), bins+1)
    return np.unique(edges)

def compute_woe(df, feature, target, bins=10, method="quantile", eps=1e-6):
    x = df[feature].values
    y = df[target].values
    edges = woe_psi_binning(x, bins=bins, method=method)
    cats = pd.cut(x, bins=edges, include_lowest=True)
    grouped = df.groupby(cats)[target].agg(["sum","count"]).reset_index()
    grouped.rename(columns={"sum":"bad","count":"total"}, inplace=True)
    grouped["good"] = grouped["total"] - grouped["bad"]
    total_bad = grouped["bad"].sum()
    total_good = grouped["good"].sum()
    grouped["bad_rate"] = (grouped["bad"]+eps)/(total_bad+eps)
    grouped["good_rate"] = (grouped["good"]+eps)/(total_good+eps)
    grouped["WoE"] = np.log(grouped["good_rate"]/grouped["bad_rate"])
    grouped["IV"] = (grouped["good_rate"] - grouped["bad_rate"])*grouped["WoE"]
    return grouped, edges

def compute_psi(expected, actual, bins=10, method="quantile", eps=1e-6):
    edges = woe_psi_binning(expected, bins=bins, method=method)
    e = pd.cut(expected, bins=edges, include_lowest=True)
    a = pd.cut(actual, bins=edges, include_lowest=True)
    e_dist = pd.Series(e).value_counts().sort_index()
    a_dist = pd.Series(a).value_counts().sort_index()
    e_pct = (e_dist + eps)/(e_dist.sum()+eps)
    a_pct = (a_dist + eps)/(a_dist.sum()+eps)
    psi = ((a_pct - e_pct)*np.log(a_pct/e_pct)).sum()
    return pd.DataFrame({"bin":e_dist.index.astype(str),"expected_pct":e_pct.values,"actual_pct":a_pct.values}), psi, edges

def upload_csv():
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
    return None

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Correlation","Multicollinearity","VIF","WoE","PSI"])

# -----------------------------
# Tab 1: Correlation
# -----------------------------
with tab1:
    st.header("Correlation: Credit Risk Scenarios")
    st.markdown("**Purpose:** Measure linear relationship between two variables (e.g., Income vs Default Probability).")
    
    df_upload = upload_csv()
    
    if df_upload is not None:
        numeric_cols = df_upload.select_dtypes(include=np.number).columns.tolist()
        col_x = st.selectbox("Select X", numeric_cols)
        col_y = st.selectbox("Select Y", numeric_cols)
        data = df_upload[[col_x, col_y]].dropna()
        data.columns = ["X","Y"]
    else:
        n = st.slider("Sample size", 100, 5000, value=500, step=50)
        rng = np.random.default_rng(42)
        X = rng.normal(5000,1500,n)
        Y = np.clip(1 - 0.0001*X + rng.normal(0,0.05,n),0,1)
        data = pd.DataFrame({"X":X,"Y":Y})

    corr_pearson, pval = stats.pearsonr(data["X"], data["Y"])
    corr_spearman, pval_s = stats.spearmanr(data["X"], data["Y"])
    
    st.metric("Pearson r", f"{corr_pearson:.3f}", help=f"p-value: {pval:.3g}")
    st.metric("Spearman ρ", f"{corr_spearman:.3f}", help=f"p-value: {pval_s:.3g}")
    
    with st.expander("Formula"):
        st.latex(r"r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2} \sqrt{\sum (y_i - \bar{y})^2}}")
    
    chart = alt.Chart(data).mark_circle(size=50, opacity=0.6).encode(
        x="X", y="Y", tooltip=["X","Y"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.info("Scatterplot shows how X and Y co-vary. Strong positive or negative slope indicates correlation.")

# -----------------------------
# Tab 2: Multicollinearity
# -----------------------------
with tab2:
    st.header("Multicollinearity")
    st.markdown("**Purpose:** See how predictors correlate and affect regression stability.")
    df_upload = upload_csv()
    
    if df_upload is not None:
        numeric_cols = df_upload.select_dtypes(include=np.number).columns.tolist()
        target = st.selectbox("Select target (Y)", numeric_cols)
        predictors = st.multiselect("Select predictors", [c for c in numeric_cols if c!=target])
        dfM = df_upload[[target]+predictors].dropna()
    else:
        rng = np.random.default_rng(42)
        X1 = rng.normal(5000,1500,500)
        X2 = 0.3*X1 + rng.normal(0,500,500)
        X3 = rng.normal(10000,5000,500)
        X4 = rng.normal(40,10,500)
        Y = 0.001*X2 + 0.0001*X1 -0.0001*X3 + rng.normal(0,0.01,500)
        dfM = pd.DataFrame({"Y":Y,"Income":X1,"Loan":X2,"Savings":X3,"Age":X4})
    
    X = dfM.drop(columns=["Y"])
    y = dfM["Y"]
    model = LinearRegression().fit(X,y)
    st.subheader("Regression Coefficients")
    coef_df = pd.DataFrame({"feature": X.columns,"coef":model.coef_})
    st.dataframe(coef_df, use_container_width=True)
    
    st.subheader("Correlation Heatmap")
    corr_mat = dfM.corr()
    corr_chart = alt.Chart(corr_mat.reset_index().melt("index")).mark_rect().encode(
        x="index:N", y="variable:N", color="value:Q",
        tooltip=["index","variable", alt.Tooltip("value:Q", format=".2f")]
    )
    st.altair_chart(corr_chart, use_container_width=True)
    st.info("Red/Blue heatmap shows predictor correlations. High correlation indicates multicollinearity risk.")

# -----------------------------
# Tab 3: VIF
# -----------------------------
with tab3:
    st.header("Variance Inflation Factor (VIF)")
    st.markdown("**Purpose:** Quantify how multicollinearity inflates variance of regression coefficients.")
    df_upload = upload_csv()
    
    if df_upload is not None:
        numeric_cols = df_upload.select_dtypes(include=np.number).columns.tolist()
        predictors = st.multiselect("Select predictors", numeric_cols)
        dfV = df_upload[predictors].dropna()
    else:
        rng = np.random.default_rng(42)
        X1 = rng.normal(5000,1500,500)
        X2 = 0.95*X1 + rng.normal(0,500,500)
        X3 = rng.normal(10000,5000,500)
        dfV = pd.DataFrame({"X1":X1,"X2":X2,"X3":X3})
    
    if dfV.shape[1]>=2:
        vif_df = compute_vif_matrix(dfV)
        vif_df["RAG"] = vif_df["VIF"].apply(lambda x: rag_color(x,(5,10)))
        st.dataframe(vif_df,use_container_width=True)
    
    with st.expander("Formula"):
        st.latex(r"VIF_j = \frac{1}{1 - R_j^2}")
        st.markdown("Where R_j^2 is from regressing predictor X_j on all other predictors.")

# -----------------------------
# Tab 4: WoE (Multi-feature)
# -----------------------------
with tab4:
    st.header("Weight of Evidence (WoE) - Multi-feature")
    st.markdown("**Purpose:** Transform multiple features for logistic regression. Highlight which bins contribute most to IV.")

    df_upload = upload_csv()
    
    if df_upload is not None:
        numeric_cols = df_upload.select_dtypes(include=np.number).columns.tolist()
        target = st.selectbox("Select binary target (0=good,1=bad)", numeric_cols)
        features = st.multiselect("Select one or more features", [c for c in numeric_cols if c!=target], default=[numeric_cols[0]])
        dfW = df_upload[[target]+features].dropna()
    else:
        rng = np.random.default_rng(42)
        n = 500
        features = ["Debt_to_Income","Credit_Utilization"]
        x1 = rng.beta(2,5,n)
        x2 = rng.beta(2,5,n)
        p = 1/(1+np.exp(-5*(x1-0.2)))
        y = (rng.uniform(0,1,n)<p).astype(int)
        dfW = pd.DataFrame({"target":y,"Debt_to_Income":x1,"Credit_Utilization":x2})

    bins = st.slider("Number of bins", 3,20,value=10)
    
    for feature in features:
        st.subheader(f"Feature: {feature}")
        woe_table, edges = compute_woe(dfW, feature, "target", bins=bins)
        woe_table["IV_contrib"] = abs(woe_table["IV"])
        IV_total = woe_table["IV"].sum()
        st.metric("Information Value (IV)", f"{IV_total:.3f}", delta_color=rag_color(IV_total,(0.02,0.1)))
        st.dataframe(woe_table[["feature","bad","good","total","WoE","IV"]], use_container_width=True)
        
        chart = alt.Chart(woe_table.reset_index()).mark_bar().encode(
            x="index:N",
            y="WoE:Q",
            color=alt.Color("IV_contrib:Q", scale=alt.Scale(scheme='reds'), title="IV contribution"),
            tooltip=["feature","bad","good","total","WoE","IV"]
        )
        st.altair_chart(chart, use_container_width=True)
        
        with st.expander("Formula"):
            st.latex(r"\text{WoE}_k = \ln\left(\frac{\text{Good}_k/\text{Total Good}}{\text{Bad}_k/\text{Total Bad}}\right)")
            st.latex(r"\text{IV} = \sum_k (\frac{\text{Good}_k}{\text{Total Good}} - \frac{\text{Bad}_k}{\text{Total Bad}}) \cdot \text{WoE}_k")
        
        st.info("Redder bars indicate bins contributing more to IV. High IV means strong predictive power for default risk.")

# -----------------------------
# Tab 5: PSI (Multi-feature)
# -----------------------------
with tab5:
    st.header("Population Stability Index (PSI) - Multi-feature")
    st.markdown("**Purpose:** Detect population shifts for multiple features that may impact model performance.")

    df_upload = upload_csv()
    
    if df_upload is not None:
        numeric_cols = df_upload.select_dtypes(include=np.number).columns.tolist()
        expected_cols = st.multiselect("Select expected columns", numeric_cols, default=numeric_cols[:1])
        actual_cols = st.multiselect("Select actual columns", numeric_cols, default=numeric_cols[:1])
        if len(expected_cols)!=len(actual_cols):
            st.warning("Expected and actual columns must be same length")
        else:
            for exp_col, act_col in zip(expected_cols, actual_cols):
                expected = df_upload[exp_col].dropna()
                actual = df_upload[act_col].dropna()
                psi_table, psi_value, edges = compute_psi(expected, actual, bins=10)
                psi_table["contrib"] = abs(psi_table["actual_pct"] - psi_table["expected_pct"])
                
                st.subheader(f"Feature: {exp_col}")
                st.metric("PSI", f"{psi_value:.3f}", delta_color=rag_color(psi_value,(0.1,0.25)))
                st.dataframe(psi_table,use_container_width=True)
                
                chart_df = pd.DataFrame({"bin": np.tile(psi_table['bin'],2),
                                         "pct": np.concatenate([psi_table['expected_pct'], psi_table['actual_pct']]),
                                         "group": ["Expected"]*len(psi_table)+["Actual"]*len(psi_table),
                                         "contrib": np.concatenate([psi_table['contrib'], psi_table['contrib']])})
                chart = alt.Chart(chart_df).mark_bar(opacity=0.6).encode(
                    x="bin:N",
                    y="pct:Q",
                    color=alt.Color("contrib:Q", scale=alt.Scale(scheme='reds'), title="PSI contribution"),
                    tooltip=["bin","group","pct","contrib"]
                )
                st.altair_chart(chart, use_container_width=True)
                
                with st.expander("Formula"):
                    st.latex(r"\text{PSI} = \sum_k (a_k - e_k) \cdot \ln\left(\frac{a_k}{e_k}\right)")
                
                st.info("Redder bins contribute more to PSI, indicating significant shifts in this feature's distribution.")
    else:
        st.info("Upload CSV or use synthetic data to analyze PSI for multiple features.")

st.caption("This app simulates real-life credit risk scenarios. Upload your own CSV or use synthetic data to explore metrics interactively.")
