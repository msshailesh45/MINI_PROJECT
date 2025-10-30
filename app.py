import math
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state
import plotly.express as px


# -----------------------------
# Utility and core computations
# -----------------------------

def _safe_eps(arr_len: int) -> float:
    # Small smoothing value to avoid division by zero
    return 1.0 / max(arr_len, 10_000)

def _clean_series_numeric(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").dropna().to_numpy()

def _clean_series_categorical(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("<<MISSING>>")

def _hist_props(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(x, bins=bins)
    props = counts / max(len(x), 1)
    return props

def psi_numeric(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index (PSI) for numeric arrays using baseline quantile bins."""
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    # Use baseline (expected) quantiles as bin edges
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, qs))
    # Fallback: if too few unique edges, use linear edges between min/max
    if len(edges) < 3:
        lo = np.nanmin(expected)
        hi = np.nanmax(expected)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return np.nan
        edges = np.linspace(lo, hi, bins + 1)

    exp_p = _hist_props(expected, edges)
    act_p = _hist_props(actual, edges)

    eps = _safe_eps(len(expected) + len(actual))
    exp_p = np.clip(exp_p, eps, None)
    act_p = np.clip(act_p, eps, None)

    contrib = (act_p - exp_p) * np.log(act_p / exp_p)
    return float(np.sum(contrib))

def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    """PSI for categorical data over the union of categories."""
    if expected.empty or actual.empty:
        return np.nan
    exp = _clean_series_categorical(expected)
    act = _clean_series_categorical(actual)

    categories = pd.Index(sorted(set(exp.unique()).union(set(act.unique()))), dtype="string")
    exp_counts = exp.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
    act_counts = act.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)

    exp_p = exp_counts / max(exp_counts.sum(), 1.0)
    act_p = act_counts / max(act_counts.sum(), 1.0)

    eps = _safe_eps(len(expected) + len(actual))
    exp_p = np.clip(exp_p, eps, None)
    act_p = np.clip(act_p, eps, None)

    contrib = (act_p - exp_p) * np.log(act_p / exp_p)
    return float(np.sum(contrib))

def ks_and_wasserstein(expected: np.ndarray, actual: np.ndarray):
    """Two-sample KS test and Wasserstein distance for numeric arrays."""
    if len(expected) == 0 or len(actual) == 0:
        return np.nan, np.nan
    # KS
    try:
        ks_stat, ks_p = stats.ks_2samp(expected, actual, alternative="two-sided", method="auto")
    except TypeError:
        # Older SciPy versions have 'mode' instead of 'method'
        ks_stat, ks_p = stats.ks_2samp(expected, actual, alternative="two-sided", mode="auto")
    # Wasserstein
    wd = stats.wasserstein_distance(expected, actual)
    return float(ks_stat), float(ks_p), float(wd)

def _pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Efficient squared Euclidean distances
    # Returns matrix [len(A), len(B)]
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    A2 = np.sum(A*A, axis=1, keepdims=True)
    B2 = np.sum(B*B, axis=1, keepdims=True).T
    return A2 + B2 - 2 * A @ B.T

def _rbf_gamma_median_heuristic(X: np.ndarray, Y: np.ndarray, rng=None, max_samples: int = 1000) -> float:
    # Subsample for gamma based on median pairwise distance
    rng = check_random_state(rng)
    XY = np.vstack([X, Y])
    n = len(XY)
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        XY = XY[idx]
    D = _pairwise_sq_dists(XY, XY)
    # Take upper triangle excluding diagonal
    tri = D[np.triu_indices(len(XY), k=1)]
    med = np.median(np.sqrt(np.clip(tri, 0, None))) if len(tri) > 0 else 1.0
    if med <= 1e-12:
        med = 1.0
    gamma = 1.0 / (2.0 * (med ** 2))
    return float(gamma)

def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = None, rng=None) -> float:
    """Unbiased MMD with RBF kernel between two samples of vectors.
    Returns sqrt(MMD^2)."""
    if len(X) < 2 or len(Y) < 2:
        return np.nan
    if gamma is None:
        gamma = _rbf_gamma_median_heuristic(X, Y, rng=rng)
    Kxx = np.exp(-gamma * _pairwise_sq_dists(X, X))
    Kyy = np.exp(-gamma * _pairwise_sq_dists(Y, Y))
    Kxy = np.exp(-gamma * _pairwise_sq_dists(X, Y))

    m = len(X)
    n = len(Y)
    # Unbiased estimator
    sum_Kxx = (np.sum(Kxx) - np.trace(Kxx)) / (m * (m - 1))
    sum_Kyy = (np.sum(Kyy) - np.trace(Kyy)) / (n * (n - 1))
    sum_Kxy = np.sum(Kxy) / (m * n)

    mmd2 = sum_Kxx + sum_Kyy - 2.0 * sum_Kxy
    return float(np.sqrt(max(mmd2, 0.0)))

def mmd_permutation_pvalue(X: np.ndarray, Y: np.ndarray, num_permutations: int = 200, rng=None):
    """Optional: permutation p-value for MMD (computationally expensive)."""
    rng = check_random_state(rng)
    obs = mmd_rbf(X, Y, rng=rng)
    Z = np.vstack([X, Y])
    n = len(X)
    stats_perm = []
    for _ in range(num_permutations):
        idx = rng.permutation(len(Z))
        Xp = Z[idx[:n]]
        Yp = Z[idx[n:]]
        stats_perm.append(mmd_rbf(Xp, Yp, rng=rng))
    p = (np.sum(np.array(stats_perm) >= obs) + 1) / (num_permutations + 1)
    return obs, float(p)

# -----------------------------
# Text utilities
# -----------------------------

def tfidf_matrix(texts, max_features=1000, ngram_range=(1,2), stop_words="english"):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words=stop_words)
    X = vec.fit_transform(texts)
    return X.toarray(), vec

def token_distribution_from_tfidf(X: np.ndarray, feature_names, top_k: int = 30):
    # Aggregate TF-IDF scores across documents to get corpus-level weights
    token_scores = np.asarray(X).sum(axis=0)
    # Normalize to form a probability distribution
    if token_scores.sum() == 0:
        probs = np.zeros_like(token_scores)
    else:
        probs = token_scores / token_scores.sum()
    # Take top_k tokens
    idx = np.argsort(probs)[::-1][:top_k]
    return idx, probs[idx], [feature_names[i] for i in idx]

def text_numeric_features(texts: pd.Series):
    # Simple numeric proxies: length in chars and tokens, avg word length
    s = texts.fillna("")
    char_len = s.str.len().to_numpy()
    token_counts = s.str.split().map(len).to_numpy()
    avg_wlen = np.where(token_counts > 0, char_len / np.maximum(token_counts, 1), 0)
    return {
        "char_len": char_len,
        "token_count": token_counts,
        "avg_word_len": avg_wlen,
    }

# -----------------------------
# PSI thresholds and flags
# -----------------------------

def psi_severity(psi_value: float) -> str:
    if not np.isfinite(psi_value):
        return "N/A"
    if psi_value < 0.1:
        return "No drift"
    elif psi_value < 0.25:
        return "Moderate"
    else:
        return "Significant"

# -----------------------------
# Demo data (no internet needed)
# -----------------------------

def make_tabular_demo(seed=42, n_base=5000, n_curr=4000):
    rng = np.random.default_rng(seed)
    # Numeric features: normal distribution, slight shift in current
    base_num1 = rng.normal(0, 1, n_base)
    curr_num1 = rng.normal(0.4, 1.1, n_curr)  # drifted mean/variance

    base_num2 = rng.gamma(shape=2.0, scale=2.0, size=n_base)
    curr_num2 = rng.gamma(shape=2.0, scale=2.0, size=n_curr)  # similar

    # Categorical feature: probabilities change
    cats = np.array(["A", "B", "C", "D"])
    base_cat = rng.choice(cats, size=n_base, p=[0.50, 0.30, 0.15, 0.05])
    curr_cat = rng.choice(cats, size=n_curr, p=[0.30, 0.35, 0.25, 0.10])

    baseline = pd.DataFrame({
        "num_gauss": base_num1,
        "num_gamma": base_num2,
        "cat_grade": base_cat,
    })
    current = pd.DataFrame({
        "num_gauss": curr_num1,
        "num_gamma": curr_num2,
        "cat_grade": curr_cat,
    })
    return baseline, current

def make_text_demo(seed=24, n_base=1500, n_curr=1500):
    rng = np.random.default_rng(seed)
    vocab1 = ["movie", "story", "plot", "actor", "director", "film", "scene", "music", "drama", "comedy"]
    vocab2 = ["app", "ui", "bug", "feature", "code", "deploy", "server", "crash", "login", "update"]
    # Baseline: mostly movie words; Current: shift toward tech words
    def sample_doc(pivot="movies"):
        length = int(rng.integers(5, 20))
        words = []
        for _ in range(length):
            if pivot == "movies":
                if rng.random() < 0.8:
                    words.append(rng.choice(vocab1))
                else:
                    words.append(rng.choice(vocab2))
            else:
                if rng.random() < 0.8:
                    words.append(rng.choice(vocab2))
                else:
                    words.append(rng.choice(vocab1))
        return " ".join(words)
    baseline = pd.DataFrame({"review": [sample_doc("movies") for _ in range(n_base)]})
    current = pd.DataFrame({"review": [sample_doc("tech") for _ in range(n_curr)]})
    return baseline, current

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Data Drift Dashboard", layout="wide")

st.title("🔎 Data Drift Dashboard")
st.markdown("""
Compare a **Baseline** dataset vs a **Current** dataset and compute drift metrics:
- **PSI** (Population Stability Index)
- **KS-test** (Kolmogorov–Smirnov) for numeric features
- **Wasserstein distance** for numeric features
- **MMD** (Maximum Mean Discrepancy) for vector features (used here for text via TF-IDF)

Use the sidebar to upload CSVs (or try the demo), select columns, and tweak thresholds.
""")

with st.sidebar:
    st.header("Data & Settings")
    mode = st.radio("Data type", ["Tabular (CSV)", "Text (CSV with a text column)"])

    demo = st.checkbox("Use built-in demo data (no files needed)", value=True)

    if mode == "Tabular (CSV)":
        base_file = None
        curr_file = None
        if not demo:
            base_file = st.file_uploader("Baseline CSV", type=["csv"], key="base_tab")
            curr_file = st.file_uploader("Current CSV", type=["csv"], key="curr_tab")
        psi_bins = st.slider("PSI: number of bins (numeric)", min_value=5, max_value=50, value=10, step=1)
        ks_alpha = st.number_input("KS-test α (p-value cutoff)", min_value=0.0001, max_value=0.5, value=0.05, step=0.01, format="%.4f")
        wd_norm_thresh = st.number_input("Wasserstein drift threshold (normalized by IQR)", min_value=0.0, max_value=5.0, value=0.25, step=0.05)
    else:
        base_file = None
        curr_file = None
        if not demo:
            base_file = st.file_uploader("Baseline CSV (must contain the selected text column)", type=["csv"], key="base_text")
            curr_file = st.file_uploader("Current CSV (must contain the selected text column)", type=["csv"], key="curr_text")
        max_feats = st.slider("TF-IDF max features", min_value=200, max_value=5000, value=1000, step=100)
        top_k_tokens = st.slider("Top tokens for PSI", min_value=10, max_value=100, value=30, step=5)
        do_perm = st.checkbox("Compute MMD permutation p-value (slower)", value=False)
        num_perm = st.slider("Permutations", min_value=50, max_value=1000, value=200, step=50, disabled=not do_perm)

    st.divider()
    st.caption("Tip: PSI thresholds — < 0.1 no drift, 0.1–0.25 moderate, > 0.25 significant.")

# -----------------------------
# TABULAR MODE
# -----------------------------

if mode == "Tabular (CSV)":
    if demo:
        baseline, current = make_tabular_demo()
    else:
        if base_file is None or curr_file is None:
            st.info("Upload both Baseline and Current CSV files to proceed.")
            st.stop()
        baseline = pd.read_csv(base_file)
        current = pd.read_csv(curr_file)

    st.subheader("1) Select columns")
    with st.expander("Preview Baseline (first 5 rows)"):
        st.dataframe(baseline.head())
    with st.expander("Preview Current (first 5 rows)"):
        st.dataframe(current.head())

    num_cols = sorted(list(set(baseline.select_dtypes(include=[np.number]).columns) & set(current.select_dtypes(include=[np.number]).columns)))
    cat_cols = sorted(list(set(baseline.select_dtypes(exclude=[np.number]).columns) & set(current.select_dtypes(exclude=[np.number]).columns)))

    sel_num = st.multiselect("Numeric columns to analyze", num_cols, default=num_cols)
    sel_cat = st.multiselect("Categorical columns to analyze (PSI only)", cat_cols, default=cat_cols)

    results = []

    # Numeric analysis
    for col in sel_num:
        exp = _clean_series_numeric(baseline[col])
        act = _clean_series_numeric(current[col])
        psi_v = psi_numeric(exp, act, bins=psi_bins)
        ks_stat, ks_p, wd = ks_and_wasserstein(exp, act)

        # Normalize Wasserstein by IQR of baseline for a scale-free indicator
        try:
            iqr = stats.iqr(exp, nan_policy="omit")
        except Exception:
            iqr = np.subtract(*np.percentile(exp, [75, 25]))
        iqr = iqr if iqr > 0 else np.std(exp) if np.std(exp) > 0 else 1.0
        wd_norm = wd / iqr

        drift_flag = (psi_v >= 0.25) or ((ks_p is not np.nan) and (ks_p < ks_alpha and ks_stat > 0.1)) or (wd_norm > wd_norm_thresh)

        results.append({
            "feature": col,
            "type": "numeric",
            "PSI": psi_v,
            "PSI_severity": psi_severity(psi_v),
            "KS_stat": ks_stat,
            "KS_pvalue": ks_p,
            "Wasserstein": wd,
            "Wasserstein_norm_IQR": wd_norm,
            "Drift_flag": drift_flag,
        })

    # Categorical analysis (PSI only)
    for col in sel_cat:
        psi_v = psi_categorical(baseline[col], current[col])
        results.append({
            "feature": col,
            "type": "categorical",
            "PSI": psi_v,
            "PSI_severity": psi_severity(psi_v),
            "KS_stat": np.nan,
            "KS_pvalue": np.nan,
            "Wasserstein": np.nan,
            "Wasserstein_norm_IQR": np.nan,
            "Drift_flag": psi_v >= 0.25,
        })

    res_df = pd.DataFrame(results).set_index("feature")
    st.subheader("2) Results")
    st.dataframe(res_df.style.format({
        "PSI": "{:.3f}", "KS_stat": "{:.3f}", "KS_pvalue": "{:.3f}",
        "Wasserstein": "{:.3f}", "Wasserstein_norm_IQR": "{:.3f}"
    }))

    st.download_button("Download results as CSV", res_df.reset_index().to_csv(index=False).encode("utf-8"), file_name="drift_results_tabular.csv")

    st.subheader("3) Visualize a feature")
    sel_vis = st.selectbox("Pick a feature to visualize", sel_num + sel_cat, index=0 if (sel_num or sel_cat) else None)
    if sel_vis:
        col = sel_vis
        if col in sel_num:
            exp = _clean_series_numeric(baseline[col])
            act = _clean_series_numeric(current[col])
            st.write(f"**Histogram / Density — {col}**")
            df_hist = pd.DataFrame({
    col: np.concatenate([exp, act]),
    "dataset": ["baseline"] * len(exp) + ["current"] * len(act)
})
            fig = px.histogram(
                df_hist,
                x=col,
                color="dataset",
                nbins=30,
                marginal="rug",
                barmode="overlay",
                opacity=0.6,
                title=f"Histogram / Density — {col}"
                )
            st.plotly_chart(fig, use_container_width=True)
            # ECDF
            st.write("**Empirical CDF**")
            import plotly.graph_objects as go
            def ecdf(arr):
                x = np.sort(arr)
                y = np.arange(1, len(arr)+1) / len(arr)
                return x, y
            x1, y1 = ecdf(exp)
            x2, y2 = ecdf(act)
            fig = go.Figure()
            fig.add_scatter(x=x1, y=y1, mode="lines", name="baseline")
            fig.add_scatter(x=x2, y=y2, mode="lines", name="current")
            fig.update_layout(xaxis_title=col, yaxis_title="ECDF", template="simple_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Categorical bar chart
            exp = _clean_series_categorical(baseline[col])
            act = _clean_series_categorical(current[col])
            cats = sorted(set(exp.unique()).union(set(act.unique())))
            exp_counts = pd.Series(exp).value_counts().reindex(cats, fill_value=0)
            act_counts = pd.Series(act).value_counts().reindex(cats, fill_value=0)
            dfb = pd.DataFrame({
                "category": cats,
                "baseline": exp_counts.values / max(exp_counts.sum(), 1),
                "current": act_counts.values / max(act_counts.sum(), 1),
            })
            df_long = dfb.melt(id_vars="category", var_name="dataset", value_name="proportion")
            st.write(f"**Category distribution — {col}**")
            st.plotly_chart(__import__("plotly.express").__dict__["bar"](df_long, x="category", y="proportion", color="dataset", barmode="group"), use_container_width=True)

# -----------------------------
# TEXT MODE
# -----------------------------

else:
    if demo:
        baseline, current = make_text_demo()
    else:
        if base_file is None or curr_file is None:
            st.info("Upload both Baseline and Current CSV files to proceed.")
            st.stop()
        baseline = pd.read_csv(base_file)
        current = pd.read_csv(curr_file)

    st.subheader("1) Select the text column")
    common_cols = sorted(list(set(baseline.columns) & set(current.columns)))
    text_col = st.selectbox("Text column present in both files", common_cols, index=0)
    with st.expander("Preview Baseline (first 5 rows)"):
        st.dataframe(baseline[[text_col]].head())
    with st.expander("Preview Current (first 5 rows)"):
        st.dataframe(current[[text_col]].head())

    st.subheader("2) Vectorize text (TF-IDF)")
    base_texts = baseline[text_col].astype("string").fillna("").tolist()
    curr_texts = current[text_col].astype("string").fillna("").tolist()

    X_base, vec = tfidf_matrix(base_texts, max_features=max_feats)
    X_curr = vec.transform(curr_texts).toarray()
    feature_names = np.array(vec.get_feature_names_out())

    st.write(f"TF-IDF matrix shapes — baseline: {X_base.shape}, current: {X_curr.shape}")

    st.subheader("3) MMD on TF-IDF vectors")
    with st.spinner("Computing MMD (RBF kernel)…"):
        mmd_value = mmd_rbf(X_base, X_curr)
        st.metric("MMD (RBF)", f"{mmd_value:.4f}")
        if st.checkbox("Show permutation p-value settings", value=False):
            st.info("Permutation test shuffles documents between groups to estimate how unusual the observed MMD is.")
        if st.session_state.get("do_perm_global", None) is None:
            st.session_state["do_perm_global"] = False

        if st.checkbox("Compute permutation p-value (slow)", value=False):
            obs, p = mmd_permutation_pvalue(X_base, X_curr, num_permutations=200)
            st.write(f"Observed MMD: **{obs:.4f}**, permutation **p-value: {p:.4f}**")

    st.subheader("4) PSI on top TF-IDF tokens")
    idx_base, probs_base, toks_base = token_distribution_from_tfidf(X_base, feature_names, top_k=top_k_tokens)
    # Align top tokens across both corpora: use union of top tokens from each side
    idx_curr, probs_curr, toks_curr = token_distribution_from_tfidf(X_curr, feature_names, top_k=top_k_tokens)
    tokens_union = list(dict.fromkeys(toks_base + toks_curr))  # preserve order
    # Build aligned probability vectors for these tokens
    def probs_for_tokens(X, feature_names, tokens):
        scores = np.asarray(X).sum(axis=0)
        total = scores.sum()
        if total == 0:
            return np.zeros(len(tokens))
        feats = {f: i for i, f in enumerate(feature_names)}
        arr = np.zeros(len(tokens))
        for j, t in enumerate(tokens):
            i = feats.get(t, None)
            if i is not None:
                arr[j] = scores[i]
        arr = arr / (arr.sum() if arr.sum() > 0 else 1.0)
        return arr
    p_exp = probs_for_tokens(X_base, feature_names, tokens_union)
    p_act = probs_for_tokens(X_curr, feature_names, tokens_union)

    eps = _safe_eps(len(tokens_union))
    p_exp = np.clip(p_exp, eps, None)
    p_act = np.clip(p_act, eps, None)
    psi_text = float(np.sum((p_act - p_exp) * np.log(p_act / p_exp)))
    st.metric("PSI (top-token distribution)", f"{psi_text:.3f}", help="PSI computed over the union of top tokens from each corpus.")

    df_tokens = pd.DataFrame({
        "token": tokens_union,
        "baseline_prob": p_exp,
        "current_prob": p_act,
        "shift_contrib": (p_act - p_exp) * np.log(p_act / p_exp),
    }).sort_values("shift_contrib", ascending=False)
    st.write("Top token shifts (higher contribution => more drift):")
    st.dataframe(df_tokens.style.format({"baseline_prob": "{:.4f}", "current_prob": "{:.4f}", "shift_contrib": "{:.4f}"}))

    st.subheader("5) KS & Wasserstein on simple numeric proxies")
    feats_base = text_numeric_features(baseline[text_col])
    feats_curr = text_numeric_features(current[text_col])

    num_results = []
    for k in feats_base.keys():
        exp = feats_base[k]
        act = feats_curr[k]
        psi_v = psi_numeric(exp, act, bins=10)
        ks_stat, ks_p, wd = ks_and_wasserstein(exp, act)
        try:
            iqr = stats.iqr(exp, nan_policy="omit")
        except Exception:
            iqr = np.subtract(*np.percentile(exp, [75, 25]))
        iqr = iqr if iqr > 0 else np.std(exp) if np.std(exp) > 0 else 1.0
        wd_norm = wd / iqr
        num_results.append({
            "feature": f"text::{k}",
            "type": "numeric (proxy)",
            "PSI": psi_v,
            "KS_stat": ks_stat,
            "KS_pvalue": ks_p,
            "Wasserstein": wd,
            "Wasserstein_norm_IQR": wd_norm,
        })
    st.dataframe(pd.DataFrame(num_results).set_index("feature").style.format({
        "PSI": "{:.3f}", "KS_stat": "{:.3f}", "KS_pvalue": "{:.3f}", "Wasserstein": "{:.3f}", "Wasserstein_norm_IQR": "{:.3f}"
    }))

    st.subheader("6) Visualize token distribution")
    import plotly.express as px
    df_melt = df_tokens.melt(id_vars="token", value_vars=["baseline_prob", "current_prob"], var_name="dataset", value_name="prob")
    st.plotly_chart(px.bar(df_melt, x="token", y="prob", color="dataset", barmode="group"), use_container_width=True)
