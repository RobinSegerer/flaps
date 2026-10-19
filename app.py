
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

APP_VERSION = "1.7.0"

APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "flaps_logo.png"

st.set_page_config(
    page_title="FLAPs",
    page_icon="🐌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
.statspace-card {
    border: 1px solid rgba(120,120,120,.25);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    margin-bottom: .8rem;
}
.small-note {font-size: .92rem; opacity: .82;}
.success-box {
    border-left: 5px solid #2ca02c;
    padding: .65rem 1rem;
    background: rgba(44,160,44,.08);
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Base data
# -----------------------------
NAMES = ["Anna", "Ben", "Carla", "David", "Elena", "Farid", "Giulia", "Hugo", "Ida", "Jonas"]

BASE = pd.DataFrame({
    "Person": NAMES,
    "Wortschatz": [42, 47, 51, 56, 62, 49, 54, 44, 58, 52],
    "Sprachverständnis": [92, 101, 113, 121, 136, 96, 124, 84, 129, 104],
    "Aufmerksamkeit": [51, 46, 52, 54, 61, 57, 50, 48, 56, 53],
    "Satzverständnis": [44, 48, 53, 57, 63, 47, 58, 43, 60, 51],
})

NUMERIC_COLS = ["Wortschatz", "Sprachverständnis", "Aufmerksamkeit", "Satzverständnis"]
BASE[NUMERIC_COLS] = BASE[NUMERIC_COLS].astype(float)

def init_state():
    if "data" not in st.session_state:
        st.session_state.data = BASE.copy()
        st.session_state.data[NUMERIC_COLS] = st.session_state.data[NUMERIC_COLS].astype(float)
    if "module" not in st.session_state:
        st.session_state.module = "1 · Mittelwert & Varianz"
    if "show_formula" not in st.session_state:
        st.session_state.show_formula = False
    if "show_calc" not in st.session_state:
        st.session_state.show_calc = False
    if "ktt_rng_seed" not in st.session_state:
        st.session_state.ktt_rng_seed = 7
    if "ktt_results" not in st.session_state:
        st.session_state.ktt_results = []

init_state()

def reset_data():
    st.session_state.data = BASE.copy()
    st.session_state.data[NUMERIC_COLS] = st.session_state.data[NUMERIC_COLS].astype(float)

def clear_widget_keys(prefixes):
    """Remove stale widget values so transformed data are not overwritten on rerun."""
    for key in list(st.session_state.keys()):
        if any(key.startswith(prefix) for prefix in prefixes):
            del st.session_state[key]


def _sync_m1_selected_slider():
    person = st.session_state.get("m1_person_dropdown")
    if person is None:
        return
    data = st.session_state.data
    idxs = data.index[data["Person"] == person]
    if len(idxs):
        value = float(data.loc[idxs[0], "Wortschatz"])
        st.session_state[f"m1_single_{person}"] = int(round(value))

def transform_m1(operation):
    data = st.session_state.data.copy()
    before = data["Wortschatz"].astype(float).copy()

    if operation == "plus5":
        data["Wortschatz"] = (before + 5).astype(float)
        label = "Wortschatz: +5"
    elif operation == "minus5":
        data["Wortschatz"] = (before - 5).astype(float)
        label = "Wortschatz: −5"
    elif operation == "times2":
        data["Wortschatz"] = (before * 2).astype(float)
        label = "Wortschatz: ×2"
    elif operation == "half":
        data["Wortschatz"] = (before * 0.5).astype(float)
        label = "Wortschatz: ×0.5"
    elif operation == "reset":
        data["Wortschatz"] = BASE["Wortschatz"].to_numpy(dtype=float)
        label = "Wortschatz zurückgesetzt"
    else:
        return

    st.session_state.data = data
    changed = int(np.sum(~np.isclose(before.to_numpy(), data["Wortschatz"].astype(float).to_numpy())))
    st.session_state["m1_last_transform"] = (label, changed, len(data))
    _sync_m1_selected_slider()

def _sync_m2_selected_sliders():
    person = st.session_state.get("m2_person_dropdown")
    if person is None:
        return
    data = st.session_state.data
    idxs = data.index[data["Person"] == person]
    if not len(idxs):
        return
    idx = idxs[0]
    st.session_state[f"m2_single_x_{person}"] = int(round(float(data.loc[idx, "Wortschatz"])))
    st.session_state[f"m2_single_y_{person}"] = int(round(float(data.loc[idx, "Sprachverständnis"])))

def transform_m2(operation):
    data = st.session_state.data.copy()
    target = st.session_state.get("m2_transform_target", "Sprachverständnis (Y)")
    col = "Sprachverständnis" if str(target).startswith("Sprach") else "Wortschatz"

    before_col = data[col].astype(float).copy()
    r_before = corr(data["Wortschatz"], data["Sprachverständnis"])

    if operation == "plus5":
        data[col] = (before_col + 5).astype(float)
        label = f"{col}: +5"
    elif operation == "times2":
        data[col] = (before_col * 2).astype(float)
        label = f"{col}: ×2"
    elif operation == "reset":
        data["Wortschatz"] = BASE["Wortschatz"].to_numpy(dtype=float)
        data["Sprachverständnis"] = BASE["Sprachverständnis"].to_numpy(dtype=float)
        label = "Beide Variablen zurückgesetzt"
    else:
        return

    st.session_state.data = data
    r_after = corr(data["Wortschatz"], data["Sprachverständnis"])
    if operation == "reset":
        changed = len(data)
    else:
        changed = int(np.sum(~np.isclose(before_col.to_numpy(), data[col].astype(float).to_numpy())))

    st.session_state["m2_last_transform"] = (label, r_before, r_after, changed, len(data))
    _sync_m2_selected_sliders()

def sample_var(x):
    return float(np.var(np.asarray(x, dtype=float), ddof=1))

def sample_sd(x):
    return float(np.std(np.asarray(x, dtype=float), ddof=1))

def covariance(x, y):
    return float(np.cov(np.asarray(x, float), np.asarray(y, float), ddof=1)[0,1])

def corr(x, y):
    return float(np.corrcoef(np.asarray(x, float), np.asarray(y, float))[0,1])

def mean(x):
    return float(np.mean(np.asarray(x, dtype=float)))

def median(x):
    return float(np.median(np.asarray(x, dtype=float)))

def padded_range(values, min_pad=5, frac=.12):
    arr = np.asarray(values, dtype=float)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    span = max(hi - lo, 1.0)
    pad = max(min_pad, span * frac)
    return [lo - pad, hi + pad]


def teaching_range(values, min_pad=5, frac=.12, allow_negative=False):
    """Axis range for test-score plots; starts at 0 when the teaching option is active."""
    arr = np.asarray(values, dtype=float)
    lo, hi = padded_range(arr, min_pad=min_pad, frac=frac)
    if st.session_state.get("zero_based_axes", True) and not allow_negative:
        upper = max(hi, float(np.max(arr)) * 1.08, 1.0)
        return [0, upper]
    return [lo, hi]


def smooth_density(values, grid=None, bandwidth=None):
    """Simple Gaussian KDE implemented with NumPy for teaching visualisations."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([]), np.array([])
    sd = np.std(x, ddof=1) if n > 1 else 1.0
    if bandwidth is None:
        bandwidth = max(1.5, 1.06 * max(sd, 1e-6) * (n ** (-1/5)))
    if grid is None:
        lo, hi = padded_range(x, min_pad=max(5, 3*bandwidth), frac=.15)
        grid = np.linspace(lo, hi, 350)
    z = (grid[:, None] - x[None, :]) / bandwidth
    kernels = np.exp(-0.5 * z**2) / (np.sqrt(2*np.pi) * bandwidth)
    density = kernels.mean(axis=1)
    return grid, density

def zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std(ddof=1)


def simulate_population_from_sample(values, n=500, seed=123):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else 1.0
    sd = max(sd, 0.1)
    return rng.normal(mu, sd, size=n)

def simulate_bivariate_population(x, y, n=500, seed=123):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx, my = np.mean(x), np.mean(y)
    sx = max(np.std(x, ddof=1), 0.1)
    sy = max(np.std(y, ddof=1), 0.1)
    r = float(np.corrcoef(x, y)[0,1])
    r = max(min(r, 0.999), -0.999)
    cov = [[sx*sx, r*sx*sy], [r*sx*sy, sy*sy]]
    rng = np.random.default_rng(seed)
    sim = rng.multivariate_normal([mx, my], cov, size=n)
    return sim[:,0], sim[:,1]


def learning_question(text_):
    st.markdown("### Leitfrage")
    st.info(text_)

def observation(text_):
    st.markdown("#### Beobachtung")
    st.write(text_)

def explanation(text_):
    st.markdown("#### Erklärung")
    st.write(text_)

def formula_expander(title="Mathematische Erklärung"):
    return st.expander(title)

def quick_check(question, options, answer_index, explanation_text, key):
    st.markdown("#### Check")
    choice = st.radio(question, options, key=key)
    if st.button("Antwort prüfen", key=f"{key}_btn"):
        if options.index(choice) == answer_index:
            st.success("Richtig.")
        else:
            st.warning("Noch nicht ganz.")
        st.write(explanation_text)

def version_badge():
    st.caption(f"FLAPs · Version {APP_VERSION}")


def exam_questions(section_key, questions_and_answers):
    """Compact exam-prep block: questions first, answers only on demand."""
    st.markdown("#### Prüfungsfragen")
    st.caption("Erst ausprobieren und begründen – danach Antworten prüfen.")
    for i, (q, a) in enumerate(questions_and_answers, start=1):
        st.markdown(f"**{i}. {q}**")
    if st.button("Antworten prüfen", key=f"answers_{section_key}"):
        for i, (q, a) in enumerate(questions_and_answers, start=1):
            with st.expander(f"Antwort {i}", expanded=False):
                st.write(a)



def norm_ppf(p):
    """Approximate inverse standard normal CDF (Acklam approximation), scipy-free."""
    p = np.asarray(p, dtype=float)
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    out = np.empty_like(p)
    low = p < plow
    mid = (p >= plow) & (p <= phigh)
    high = p > phigh
    if np.any(low):
        q = np.sqrt(-2*np.log(p[low]))
        out[low] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if np.any(mid):
        q = p[mid]-0.5
        r = q*q
        out[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    if np.any(high):
        q = np.sqrt(-2*np.log(1-p[high]))
        out[high] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    return out.item() if out.ndim == 0 else out


def norm_pdf(x):
    return np.exp(-0.5 * np.asarray(x, dtype=float)**2) / np.sqrt(2*np.pi)

def theoretical_item_ability_corr(target_p, differentiation=1.0):
    """
    For the latent normal-threshold simulation:
    Corr(theta, X) = rho * phi(c) / sqrt(p(1-p))
    with c = Phi^{-1}(1-p).
    """
    p = float(np.clip(target_p, 1e-6, 1-1e-6))
    rho = float(np.clip(differentiation, -0.999, 0.999))
    c = float(norm_ppf(1-p))
    return rho * float(norm_pdf(c)) / np.sqrt(p*(1-p))

def simulate_ctt_item(ability, target_p, differentiation, rng):
    """Generate a dichotomous CTT item with target marginal p and chosen differentiation strength."""
    rho = float(np.clip(differentiation, 0.0, 0.95))
    eps = rng.normal(0, 1, len(ability))
    latent = rho * ability + np.sqrt(max(1e-9, 1-rho**2)) * eps
    threshold = norm_ppf(1-target_p)
    return (latent > threshold).astype(int)

def metric_row(vals):
    cols = st.columns(len(vals))
    for c, (label, value) in zip(cols, vals):
        c.metric(label, value)

def sidebar():
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.sidebar.title("FLAPs")
        st.sidebar.caption("Forschungs- und Lern-App für angewandte Psychometrie")
    st.sidebar.caption(f"Version {APP_VERSION}")

    module = st.sidebar.radio(
        "Modul wählen",
        [
            "1 · Mittelwert & Varianz",
            "2 · Zusammenhang zweier Variablen",
            "3 · Summenscores",
            "4 · Klassische Testtheorie",
            "5 · Itemanalyse",
            "6 · Reliabilitätsschätzung",
            "7 · Validität",
        ],
        index=[
            "1 · Mittelwert & Varianz",
            "2 · Zusammenhang zweier Variablen",
            "3 · Summenscores",
            "4 · Klassische Testtheorie",
            "5 · Itemanalyse",
            "6 · Reliabilitätsschätzung",
            "7 · Validität",
        ].index(
            st.session_state.module
            if st.session_state.module in [
                "1 · Mittelwert & Varianz",
                "2 · Zusammenhang zweier Variablen",
                "3 · Summenscores",
                "4 · Klassische Testtheorie",
                "5 · Itemanalyse",
                "6 · Reliabilitätsschätzung",
                "7 · Validität",
            ]
            else "2 · Zusammenhang zweier Variablen"
        )
    )
    st.session_state.module = module

    st.sidebar.divider()
    st.sidebar.toggle("Formeln anzeigen", key="show_formula")
    st.sidebar.toggle("Rechenweg anzeigen", key="show_calc")
    st.sidebar.toggle(
        "Testwert-Achsen bei 0 beginnen",
        value=True,
        key="zero_based_axes",
        help="Gilt für unstandardisierte Testwert-Grafiken. z-standardisierte Achsen müssen auch negative Werte zeigen und beginnen deshalb nicht bei 0."
    )

    st.sidebar.divider()
    if st.sidebar.button("Alle Beispieldaten zurücksetzen", use_container_width=True):
        reset_data()
        st.rerun()

    st.sidebar.caption("Erst vorhersagen, dann ausprobieren.")

sidebar()
df = st.session_state.data

# -----------------------------
# Module 1
# -----------------------------
def module1():
    version_badge()
    df = st.session_state.data
    st.title("1 · Mittelwert & Varianz")
    st.subheader("Wo liegen die Testwerte – und wie stark streuen sie?")
    st.write(
        "Sie arbeiten mit den **Wortschatzwerten** von zehn fiktiven Studierenden. "
        "Verändern Sie einzelne Werte oder transformieren Sie die ganze Verteilung."
    )

    left, right = st.columns([1.05, 1.65], gap="large")

    with left:
        st.markdown("### Testwert einer Person verändern")
        person = st.selectbox("Person", df["Person"], key="m1_person_dropdown")
        idx = df.index[df["Person"] == person][0]
        current = int(round(float(df.loc[idx, "Wortschatz"])))

        new_val = st.slider(
            f"Wortschatzwert von {person}",
            min_value=0.0, max_value=200.0, value=float(current), step=0.5,
            key=f"m1_single_{person}"
        )
        st.session_state.data.loc[idx, "Wortschatz"] = new_val
        df = st.session_state.data

        st.markdown("### Transformations-Labor")
        st.caption(
            "Diese Buttons transformieren immer **alle Personen gleichzeitig**. "
            "Danach werden die Personenregler neu aus den transformierten Daten aufgebaut."
        )

        c1, c2 = st.columns(2)
        c1.button(
            "Zu allen +5",
            use_container_width=True,
            on_click=transform_m1,
            args=("plus5",)
        )
        c2.button(
            "Alle ×2",
            use_container_width=True,
            on_click=transform_m1,
            args=("times2",)
        )

        c3, c4 = st.columns(2)
        c3.button(
            "Zu allen −5",
            use_container_width=True,
            on_click=transform_m1,
            args=("minus5",)
        )
        c4.button(
            "Alle ×0.5",
            use_container_width=True,
            on_click=transform_m1,
            args=("half",)
        )

        st.button(
            "Wortschatz zurücksetzen",
            use_container_width=True,
            on_click=transform_m1,
            args=("reset",)
        )

        if "m1_last_transform" in st.session_state:
            label, changed, total = st.session_state["m1_last_transform"]
            st.success(f"✓ {label}: {changed} von {total} Personen wurden verändert.")

    with right:
        x = df["Wortschatz"].astype(float).to_numpy()
        view_mode = st.radio(
            "Darstellung",
            ["Stichprobe (10 Personen)", "Simulierte Population (500 Personen)"],
            horizontal=True,
            key="m1_view_mode"
        )
        mu = mean(x)
        var = sample_var(x)
        sd = sample_sd(x)

        med = median(x)
        metric_row([
            ("Mittelwert", f"{mu:.2f}"),
            ("Median", f"{med:.2f}"),
            ("Varianz", f"{var:.2f}"),
            ("Standardabweichung", f"{sd:.2f}")
        ])

        plot_df = df[["Person", "Wortschatz"]].copy()
        plot_df["y"] = 0
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df["Wortschatz"], y=plot_df["y"],
            mode="markers+text", text=plot_df["Person"],
            textposition="top center",
            marker=dict(size=13),
            hovertemplate="%{text}: %{x}<extra></extra>"
        ))
        fig.add_vline(x=mu, line_dash="dash", annotation_text=f"Mittelwert = {mu:.2f}")
        xr = teaching_range(x)
        fig.update_layout(
            height=330,
            yaxis=dict(visible=False, fixedrange=True),
            xaxis=dict(title="Wortschatz-Testwert", range=xr, fixedrange=True),
            dragmode=False,
            uirevision="m1-fixed",
            margin=dict(l=20,r=20,t=30,b=30)
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})
        if st.session_state.get("zero_based_axes", True):
            st.caption("Die Testwert-Achse beginnt bewusst bei 0, damit Abstände nicht durch eine abgeschnittene Skala optisch übertrieben werden.")

        st.markdown("#### Wie entwickelt sich die Verteilung?")
        bw_default = max(1.5, 1.06 * max(sd, 1e-6) * (len(x) ** (-1/5)))
        grid = np.linspace(xr[0], xr[1], 350)
        grid, dens = smooth_density(x, grid=grid, bandwidth=bw_default)

        dens_fig = go.Figure()
        dens_fig.add_trace(go.Scatter(
            x=grid, y=dens, mode="lines", fill="tozeroy",
            name="Geglättete Stichprobe",
            hovertemplate="Testwert: %{x:.1f}<br>Dichte: %{y:.3f}<extra></extra>"
        ))
        if view_mode.startswith("Simulierte"):
            pop = simulate_population_from_sample(x, n=500, seed=123)
            _, pop_dens = smooth_density(pop, grid=grid)
            dens_fig.add_trace(go.Scatter(
                x=grid, y=pop_dens, mode="lines",
                name="Simulierte Population"
            ))
        dens_fig.add_trace(go.Scatter(
            x=x, y=np.zeros_like(x),
            mode="markers+text",
            text=df["Person"],
            textposition="bottom center",
            marker=dict(size=8),
            name="Personen",
            hovertemplate="%{text}: %{x:.1f}<extra></extra>"
        ))
        dens_fig.add_vline(
            x=mu, line_dash="dash",
            annotation_text=f"Mittelwert {mu:.1f}",
            annotation_position="top left"
        )
        dens_fig.add_vline(
            x=med, line_dash="dot",
            annotation_text=f"Median {med:.1f}",
            annotation_position="top right"
        )
        dens_fig.update_layout(
            height=300,
            xaxis=dict(title="Wortschatz-Testwert", range=xr, fixedrange=True),
            yaxis=dict(title="Geglättete Dichte", fixedrange=True, rangemode="tozero"),
            dragmode=False,
            uirevision="m1-density",
            margin=dict(l=20, r=20, t=25, b=30),
            showlegend=False
        )
        st.plotly_chart(
            dens_fig, use_container_width=True,
            config={"displayModeBar": False, "scrollZoom": False}
        )
        st.caption(
            "Die Kurve ist eine didaktische Glättung der wenigen Einzelwerte, keine angenommene "
            "Populationsverteilung. Wenn Sie nur eine Person verschieben, verändert sich die Kurve "
            "kontinuierlich mit."
        )

        with st.expander("Was sollte ich hier beobachten?", expanded=False):
            st.write(
                "Wenn Sie **zu allen Werten dieselbe Zahl addieren**, verschiebt sich die ganze Verteilung: "
                "Der Mittelwert ändert sich, die Abstände untereinander aber nicht. "
                "Beim **Multiplizieren** verändern sich dagegen auch die Abstände und damit die Streuung."
            )

    st.divider()
    st.markdown("### Challenge")
    st.write("**Erhöhen Sie den Mittelwert um genau 5 Punkte, ohne die Varianz zu verändern.**")
    base_mu = mean(BASE["Wortschatz"])
    base_var = sample_var(BASE["Wortschatz"])
    curr_mu = mean(df["Wortschatz"])
    curr_var = sample_var(df["Wortschatz"])
    if abs(curr_mu - (base_mu + 5)) < 1e-9 and abs(curr_var - base_var) < 1e-9:
        st.success("✓ Geschafft. Die Verteilung wurde verschoben, aber ihre Streuung blieb gleich.")

    if st.session_state.show_formula:
        st.markdown("#### Formeln")
        st.latex(r"\overline{X+c}=\bar X+c")
        st.latex(r"Var(X+c)=Var(X)")
        st.latex(r"Var(aX)=a^2 Var(X)")

    if st.session_state.show_calc:
        st.markdown("#### Rechenweg für die aktuelle Wortschatzvariable")
        calc = pd.DataFrame({
            "Person": df["Person"],
            "x": x,
            "x − Mittelwert": x-mu,
            "(x − Mittelwert)²": (x-mu)**2
        })
        st.dataframe(calc.round(2), use_container_width=True, hide_index=True)
        st.write(f"Summe der quadrierten Abweichungen: **{np.sum((x-mu)**2):.2f}**")
        st.write(f"Stichprobenvarianz: **{np.sum((x-mu)**2):.2f} / {len(x)-1} = {var:.2f}**")


# -----------------------------
# Module 2
# -----------------------------

    exam_questions("m1", [
        (
            "Was passiert mit Mittelwert, Varianz und Standardabweichung, wenn zu allen Testwerten 5 Punkte addiert werden?",
            "Der Mittelwert steigt um 5. Varianz und Standardabweichung bleiben unverändert, weil sich die Abstände der Personen untereinander nicht verändern."
        ),
        (
            "Was passiert mit Varianz und Standardabweichung, wenn alle Testwerte verdoppelt werden?",
            "Die Standardabweichung verdoppelt sich, die Varianz vervierfacht sich: Var(2X)=4·Var(X)."
        ),
        (
            "Warum kann eine Stichprobe denselben Mittelwert, aber eine deutlich andere Varianz haben?",
            "Der Mittelwert beschreibt das Zentrum, die Varianz die Streuung. Zwei Verteilungen können dasselbe Zentrum, aber sehr unterschiedliche Abstände der Einzelwerte zum Zentrum haben."
        ),
    ])




def module2():
    version_badge()
    st.title("2 · Zusammenhang zwischen zwei Variablen")
    st.caption(
        "Durchgehendes Beispiel: Wortschatz und Sprachverständnis werden bewusst auf deutlich unterschiedlichen "
        "Testskalen dargestellt. So wird sichtbar, was sich durch die Maßeinheit verändert – und was nicht."
    )

    st.markdown(
        "**Lernweg:** Kovarianz beschreibt gemeinsames Abweichen → Korrelation standardisiert diesen Zusammenhang → "
        "Regression nutzt den Zusammenhang zur Vorhersage."
    )

    step = st.radio(
        "Lernschritt",
        ["2A · Kovarianz", "2B · Von der Kovarianz zur Korrelation", "2C · Von der Korrelation zur Regression"],
        horizontal=True,
        key="m2_step"
    )
    st.divider()
    if step.startswith("2A"):
        module2_covariance()
    elif step.startswith("2B"):
        module2_correlation()
    else:
        module2_regression()


def module2_covariance():
    df = st.session_state.data
    learning_question(
        "Wie können wir beschreiben, ob zwei Testwerte bei denselben Personen gemeinsam über oder unter ihrem jeweiligen Mittelwert liegen?"
    )

    X = df["Wortschatz"].astype(float).to_numpy()
    Y = df["Sprachverständnis"].astype(float).to_numpy()
    mx, my = np.mean(X), np.mean(Y)
    cov_xy = covariance(X, Y)

    st.markdown("#### Experiment")
    person = st.selectbox("Person hervorheben", df["Person"].tolist(), key="m2a_person")
    idx = df.index[df["Person"] == person][0]
    line_mode = st.radio(
        "Abweichungslinien zeigen für",
        ["alle Personen", "nur ausgewählte Person"],
        horizontal=True,
        key="m2a_lines"
    )

    xr = teaching_range(X)
    yr = teaching_range(Y)

    fig = go.Figure()
    draw_idx = range(len(df)) if line_mode == "alle Personen" else [idx]
    for j in draw_idx:
        lw = 2.8 if j == idx else 1.0
        dash = "solid" if j == idx else "dot"
        fig.add_shape(type="line", x0=mx, x1=X[j], y0=Y[j], y1=Y[j], line=dict(width=lw, dash=dash))
        fig.add_shape(type="line", x0=X[j], x1=X[j], y0=my, y1=Y[j], line=dict(width=lw, dash=dash))

    fig.add_trace(go.Scatter(
        x=X, y=Y, mode="markers+text",
        text=df["Person"], textposition="top center",
        marker=dict(size=10), name="Personen"
    ))
    fig.add_vline(x=mx, line_dash="dash", annotation_text=f"Mittelwert X = {mx:.1f}")
    fig.add_hline(y=my, line_dash="dash", annotation_text=f"Mittelwert Y = {my:.1f}")
    fig.update_layout(
        height=520,
        xaxis=dict(title="Wortschatztest", range=xr, fixedrange=True),
        yaxis=dict(title="Sprachverständnistest", range=yr, fixedrange=True),
        dragmode=False,
        margin=dict(l=30, r=30, t=35, b=35)
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

    dx = X[idx] - mx
    dy = Y[idx] - my
    contribution = dx * dy
    metric_row([
        (f"{person}: Abweichung X", f"{dx:+.1f}"),
        (f"{person}: Abweichung Y", f"{dy:+.1f}"),
        ("Produkt der Abweichungen", f"{contribution:+.1f}"),
        ("Kovarianz insgesamt", f"{cov_xy:.2f}")
    ])

    observation(
        "Die beiden Linien zeigen die Abweichungen einer Person von den zwei Mittelwerten. "
        "Haben beide Abweichungen dasselbe Vorzeichen, ist ihr Produkt positiv. Haben sie unterschiedliche Vorzeichen, ist es negativ."
    )
    explanation(
        "Die Kovarianz fasst diese Produkte über alle Personen zusammen. Eine positive Kovarianz bedeutet: "
        "Personen mit überdurchschnittlichem Wortschatz haben tendenziell auch überdurchschnittliches Sprachverständnis – "
        "und umgekehrt. Die Größe der Kovarianz ist jedoch noch schwer direkt zu interpretieren, weil sie von beiden Testskalen abhängt."
    )

    with formula_expander():
        st.latex(r"Cov(X,Y)=\frac{1}{n-1}\sum_i (X_i-\bar X)(Y_i-\bar Y)")
        st.write(
            "Die Einheit der Kovarianz ist ein Produkt der beiden Einheiten – hier also sinngemäß "
            "‚Wortschatzpunkte × Sprachverständnispunkte‘. Deshalb ändert sich ihr Zahlenwert, wenn eine Testskala umgerechnet wird."
        )

    quick_check(
        "Eine Person liegt bei beiden Tests unter dem jeweiligen Mittelwert. Welchen Beitrag liefert sie zur Kovarianz?",
        ["einen positiven", "einen negativen", "immer genau 0"],
        0,
        "Beide Abweichungen sind negativ. Das Produkt zweier negativer Zahlen ist positiv.",
        "m2a_check"
    )



def module2_correlation():
    learning_question(
        "Wie wird aus der skalenabhängigen Kovarianz Schritt für Schritt ein standardisierter Zusammenhang zwischen −1 und +1?"
    )

    st.markdown("#### Experiment: Alle Testwerte einer Variablen verändern")
    st.selectbox(
        "Welche Variable verändern?",
        ["Sprachverständnis (Y)", "Wortschatz (X)"],
        key="m2_transform_target",
    )
    c1, c2, c3 = st.columns(3)
    c1.button("Zu allen +5", use_container_width=True, on_click=transform_m2, args=("plus5",))
    c2.button("Alle ×2", use_container_width=True, on_click=transform_m2, args=("times2",))
    c3.button("Beide zurücksetzen", use_container_width=True, on_click=transform_m2, args=("reset",))
    if "m2_last_transform" in st.session_state:
        label, r_before, r_after, changed, total = st.session_state["m2_last_transform"]
        st.success(
            f"{label}: {changed} von {total} Personen verändert. "
            f"Korrelation vorher: {r_before:.3f}; nachher: {r_after:.3f}."
        )
    st.caption("Die Umrechnung gilt für alle Personen der gewählten Variablen und kann mehrfach angewendet werden.")

    df = st.session_state.data
    X = df["Wortschatz"].astype(float).to_numpy()
    Y = df["Sprachverständnis"].astype(float).to_numpy()
    mx, my = np.mean(X), np.mean(Y)
    sx, sy = np.std(X, ddof=1), np.std(Y, ddof=1)
    cov_xy = covariance(X, Y)
    r_xy = corr(X, Y)

    st.markdown("#### Experiment: Standardisieren Sie die Punktwolke Schritt für Schritt")
    person = st.selectbox("Person hervorheben", df["Person"].tolist(), key="m2b_person")
    idx = df.index[df["Person"] == person][0]

    steps = [
        "1 · Originalwerte",
        "2 · Mittelwerte abziehen",
        "3 · X durch SD(X) teilen",
        "4 · Auch Y durch SD(Y) teilen",
    ]
    step = st.radio(
        "Transformationsschritt",
        options=steps,
        horizontal=True,
        key="m2b_transform_step"
    )
    step_no = steps.index(step) + 1

    # Transform the same data step by step.
    if step_no == 1:
        XP = X.copy()
        YP = Y.copy()
        x_title = "Wortschatztest"
        y_title = "Sprachverständnistest"
        step_text = (
            "Ausgangspunkt: Beide Variablen stehen auf ihren ursprünglichen Testskalen."
        )
    elif step_no == 2:
        XP = X - mx
        YP = Y - my
        x_title = "Wortschatz − Mittelwert"
        y_title = "Sprachverständnis − Mittelwert"
        step_text = (
            "Von jedem Wert wird der jeweilige Mittelwert abgezogen. "
            "Die Punktwolke wird nur zum Ursprung verschoben."
        )
    elif step_no == 3:
        XP = (X - mx) / sx
        YP = Y - my
        x_title = "Wortschatz (z)"
        y_title = "Sprachverständnis − Mittelwert"
        step_text = (
            "X ist jetzt standardisiert: Die x-Abstände werden in Standardabweichungen ausgedrückt. "
            "Y ist bisher nur zentriert."
        )
    else:
        XP = (X - mx) / sx
        YP = (Y - my) / sy
        x_title = "Wortschatz (z)"
        y_title = "Sprachverständnis (z)"
        step_text = (
            "Jetzt sind beide Variablen standardisiert. Beide Mittelwerte sind 0 und beide Standardabweichungen 1."
        )

    cur_mx, cur_my = float(np.mean(XP)), float(np.mean(YP))
    cur_sx = float(np.std(XP, ddof=1))
    cur_sy = float(np.std(YP, ddof=1))
    cur_cov = float(covariance(XP, YP))
    cur_r = float(corr(XP, YP))

    # For centered/standardized axes, negative values are necessary, so do not force zero.
    if step_no == 1:
        xr = teaching_range(XP)
        yr = teaching_range(YP)
    else:
        def centered_range(a):
            lo, hi = float(np.min(a)), float(np.max(a))
            pad = max((hi-lo)*0.18, 0.5)
            return [lo-pad, hi+pad]
        xr = centered_range(XP)
        yr = centered_range(YP)

    fig = go.Figure()

    # Show the same deviation-to-mean construction as in the covariance section.
    for j in range(len(df)):
        lw = 2.8 if j == idx else 0.8
        dash = "solid" if j == idx else "dot"
        fig.add_shape(
            type="line",
            x0=cur_mx, x1=XP[j], y0=YP[j], y1=YP[j],
            line=dict(width=lw, dash=dash)
        )
        fig.add_shape(
            type="line",
            x0=XP[j], x1=XP[j], y0=cur_my, y1=YP[j],
            line=dict(width=lw, dash=dash)
        )

    fig.add_trace(go.Scatter(
        x=XP, y=YP, mode="markers+text",
        text=df["Person"], textposition="top center",
        marker=dict(size=[14 if j == idx else 9 for j in range(len(df))]),
        name="Personen"
    ))
    fig.add_vline(x=cur_mx, line_dash="dash")
    fig.add_hline(y=cur_my, line_dash="dash")
    fig.update_layout(
        height=520,
        xaxis=dict(title=x_title, range=xr, fixedrange=True),
        yaxis=dict(title=y_title, range=yr, fixedrange=True),
        dragmode=False,
        showlegend=False,
        margin=dict(l=30, r=30, t=35, b=35)
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

    st.caption(step_text)

    metric_row([
        ("Mittelwert X", f"{cur_mx:.2f}"),
        ("SD X", f"{cur_sx:.2f}"),
        ("Mittelwert Y", f"{cur_my:.2f}"),
        ("SD Y", f"{cur_sy:.2f}"),
        ("Kovarianz jetzt", f"{cur_cov:.3f}"),
        ("Korrelation r", f"{cur_r:.3f}")
    ])

    # A compact running calculation makes the mechanism explicit.
    st.markdown("#### Was ist mit der Kovarianz passiert?")
    if step_no == 1:
        st.write(
            f"Ausgangswert: **Cov(X,Y) = {cov_xy:.3f}**. "
            "Die Korrelation ist bereits berechenbar, aber die Kovarianz trägt noch beide Testskalen mit."
        )
    elif step_no == 2:
        st.write(
            f"Nach dem Zentrieren bleibt die Kovarianz **unverändert: {cur_cov:.3f}**. "
            "Das Verschieben des Ursprungs verändert die gemeinsamen Abstände nicht."
        )
    elif step_no == 3:
        st.write(
            f"Nach der Standardisierung von X wird die Kovarianz durch SD(X) geteilt: "
            f"**{cov_xy:.3f} / {sx:.3f} = {cur_cov:.3f}**."
        )
    else:
        st.success(
            f"Jetzt wurde zusätzlich durch SD(Y) geteilt: "
            f"**{cov_xy:.3f} / ({sx:.3f} × {sy:.3f}) = {cur_cov:.3f}**. "
            f"Dieser Wert ist genau die **Korrelation r = {r_xy:.3f}**."
        )

    observation(
        "Die Personen behalten bei jedem Schritt ihre relative Lage zueinander. "
        "Zuerst wird die Punktwolke nur verschoben, danach werden ihre horizontalen und vertikalen Abstände neu skaliert."
    )
    explanation(
        "Zentrieren entfernt nur den Ursprung der beiden Skalen. Standardisieren geht einen Schritt weiter: "
        "Die Abstände werden durch die jeweilige Standardabweichung geteilt. "
        "Sind beide Variablen standardisiert, ist die Kovarianz nicht mehr von den ursprünglichen Maßeinheiten abhängig – "
        "und entspricht der Korrelation."
    )

    with formula_expander("Mathematisch: Schritt für Schritt von Kovarianz zu Korrelation"):
        st.latex(r"X_c=X-\bar X,\qquad Y_c=Y-\bar Y")
        st.latex(r"Cov(X_c,Y_c)=Cov(X,Y)")
        st.latex(r"Cov(z_X,Y_c)=\frac{Cov(X,Y)}{SD_X}")
        st.latex(r"Cov(z_X,z_Y)=\frac{Cov(X,Y)}{SD_X\,SD_Y}=r_{XY}")
        st.write(
            "Der erste Schritt verschiebt nur die Skalen. Die beiden folgenden Schritte entfernen nacheinander "
            "die Skalierung von X und Y."
        )

    quick_check(
        "Welcher Schritt verändert die Kovarianz noch nicht?",
        [
            "Nur das Abziehen der Mittelwerte",
            "Das Teilen von X durch SD(X)",
            "Das Teilen beider Variablen durch ihre Standardabweichung"
        ],
        0,
        "Das Zentrieren verschiebt nur den Ursprung. Erst das Teilen durch Standardabweichungen verändert die Größe der Kovarianz.",
        "m2b_check"
    )

def module2_regression():
    df = st.session_state.data
    learning_question(
        "Wie können wir denselben Zusammenhang nutzen, um aus einem Wortschatzwert einen Sprachverständniswert vorherzusagen?"
    )

    X = df["Wortschatz"].astype(float).to_numpy()
    Y = df["Sprachverständnis"].astype(float).to_numpy()
    mx, my = np.mean(X), np.mean(Y)
    sx, sy = np.std(X, ddof=1), np.std(Y, ddof=1)
    cov_xy = covariance(X, Y)
    var_x = sample_var(X)
    r_xy = corr(X, Y)

    best_slope = cov_xy / var_x
    best_intercept = my - best_slope*mx
    best_delta10 = best_slope*10
    pred_best = best_intercept + best_slope*X
    sse_best = float(np.sum((Y-pred_best)**2))

    # Stable, intuitive defaults
    if "m2c_center" not in st.session_state:
        st.session_state["m2c_center"] = float(round(my - 8, 1))
    if "m2c_delta10" not in st.session_state:
        st.session_state["m2c_delta10"] = 5.0

    st.markdown("#### Experiment: Legen Sie die Gerade so, wie Sie sie im Diagramm haben möchten")
    bc1, bc2 = st.columns(2)
    if bc1.button("Beste Gerade einsetzen", use_container_width=True, key="m2c_best_btn"):
        st.session_state["m2c_center"] = float(my)
        st.session_state["m2c_delta10"] = float(best_delta10)
        st.rerun()
    if bc2.button("Mit flacher Startgerade beginnen", use_container_width=True, key="m2c_reset_btn"):
        st.session_state["m2c_center"] = float(round(my - 8, 1))
        st.session_state["m2c_delta10"] = 5.0
        st.rerun()

    y_low = 0.0 if st.session_state.get("zero_based_axes", True) else max(0.0, float(np.min(Y))-40)
    y_high = max(200.0, float(np.max(Y))+40)
    # clamp stale session values
    st.session_state["m2c_center"] = min(max(float(st.session_state["m2c_center"]), y_low), y_high)
    st.session_state["m2c_delta10"] = min(max(float(st.session_state["m2c_delta10"]), -60.0), 60.0)

    c1, c2 = st.columns(2)
    with c1:
        center_y = st.slider(
            f"Vorhersage bei durchschnittlichem Wortschatz (X = {mx:.1f})",
            min_value=float(y_low), max_value=float(y_high), step=1.0,
            key="m2c_center"
        )
    with c2:
        delta10 = st.slider(
            "Änderung im Sprachverständnis bei +10 Wortschatzpunkten",
            min_value=-60.0, max_value=60.0, step=1.0,
            key="m2c_delta10"
        )

    slope = delta10 / 10.0
    intercept = center_y - slope*mx
    pred = intercept + slope*X
    resid = Y-pred
    sse = float(np.sum(resid**2))

    xr = teaching_range(X)
    yr_values = np.concatenate([Y, pred, pred_best])
    yr = teaching_range(yr_values)
    grid = np.linspace(xr[0], xr[1], 150)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X, y=Y, mode="markers+text", text=df["Person"], textposition="top center",
        marker=dict(size=10), name="Beobachtete Werte"
    ))
    fig.add_trace(go.Scatter(x=grid, y=intercept+slope*grid, mode="lines", name="Ihre Gerade"))
    for x, y, yp in zip(X, Y, pred):
        fig.add_shape(type="line", x0=x, x1=x, y0=yp, y1=y, line=dict(width=1, dash="dot"))
    fig.add_trace(go.Scatter(
        x=[mx], y=[center_y], mode="markers", marker=dict(size=13, symbol="diamond"),
        name="Ihre Vorhersage bei mittlerem X"
    ))
    fig.update_layout(
        height=520,
        xaxis=dict(title="Wortschatztest", range=xr, fixedrange=True),
        yaxis=dict(title="Sprachverständnistest", range=yr, fixedrange=True),
        dragmode=False
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

    distance_pct = ((sse/sse_best)-1)*100 if sse_best > 0 else np.nan
    metric_row([
        ("Ihre SSE", f"{sse:.1f}"),
        ("Kleinstmögliche SSE", f"{sse_best:.1f}"),
        ("Noch über dem Minimum", f"{distance_pct:.1f} %" if np.isfinite(distance_pct) else "–"),
        ("Steigung b₁", f"{slope:.3f}")
    ])

    observation(
        "Die senkrechten Linien zeigen die Vorhersagefehler. Mit dem ersten Regler verschieben Sie die Gerade bei einem "
        "durchschnittlichen Wortschatzwert nach oben oder unten. Mit dem zweiten Regler machen Sie sie steiler oder flacher."
    )
    explanation(
        "Die beste Regressionsgerade minimiert die Summe der quadrierten Vorhersagefehler (SSE). "
        "Sie verläuft immer durch den Punkt (Mittelwert X, Mittelwert Y). Ihre Steigung hängt direkt von Kovarianz und Korrelation ab."
    )

    with formula_expander("Mathematisch: Wie hängen Regression, Kovarianz und Korrelation zusammen?"):
        st.latex(r"b_1=\frac{Cov(X,Y)}{Var(X)}")
        st.latex(r"b_1=r_{XY}\frac{SD_Y}{SD_X}")
        st.latex(r"b_0=\bar Y-b_1\bar X")
        st.write(
            f"Aktuell: b₁ = {cov_xy:.2f} / {var_x:.2f} = {best_slope:.3f}. "
            f"Dasselbe ergibt sich aus r × SD(Y)/SD(X) = {r_xy:.3f} × {sy:.2f}/{sx:.2f} = {best_slope:.3f}."
        )
        st.write(
            "Die unstandardisierte Regressionssteigung ist deshalb normalerweise **nicht** gleich der Korrelation: "
            "Sie enthält zusätzlich das Verhältnis der beiden Standardabweichungen."
        )

    st.markdown("#### Was passiert nach der z-Standardisierung?")
    show_std = st.toggle("Standardisierte Regression anzeigen", value=False, key="m2c_std")
    if show_std:
        ZX = (X-mx)/sx
        ZY = (Y-my)/sy
        slope_z = corr(X, Y)
        zgrid = np.linspace(min(-2.5, ZX.min()-.3), max(2.5, ZX.max()+.3), 100)
        zfig = go.Figure()
        zfig.add_trace(go.Scatter(
            x=ZX, y=ZY, mode="markers+text", text=df["Person"], textposition="top center",
            marker=dict(size=10), name="z-Werte"
        ))
        zfig.add_trace(go.Scatter(x=zgrid, y=slope_z*zgrid, mode="lines", name="standardisierte Regressionsgerade"))
        zfig.add_vline(x=0, line_dash="dash")
        zfig.add_hline(y=0, line_dash="dash")
        zr = [min(-2.5, ZX.min()-.3, ZY.min()-.3), max(2.5, ZX.max()+.3, ZY.max()+.3)]
        zfig.update_layout(
            height=450,
            xaxis=dict(title="Wortschatz (z)", range=zr, fixedrange=True),
            yaxis=dict(title="Sprachverständnis (z)", range=zr, fixedrange=True),
            dragmode=False
        )
        st.plotly_chart(zfig, use_container_width=True, config={"displayModeBar": False})
        st.success(
            f"Jetzt sind SD(X)=SD(Y)=1. Deshalb wird b₁ = r × 1/1 = r. "
            f"Die standardisierte Steigung beträgt {slope_z:.3f} und entspricht genau der Korrelation."
        )

    quick_check(
        "Warum ist die Steigung der unstandardisierten Regressionsgeraden nicht automatisch gleich der Korrelation?",
        [
            "Weil die Steigung zusätzlich vom Verhältnis SD(Y)/SD(X) abhängt.",
            "Weil Regression keine Kovarianz verwendet.",
            "Weil eine Korrelation keine Richtung haben kann."
        ],
        0,
        "Es gilt b₁ = r · SD(Y)/SD(X). Erst wenn beide Variablen z-standardisiert sind, sind beide SDs gleich 1 und damit b₁ = r.",
        "m2c_check"
    )

def module4():
    version_badge()
    df = st.session_state.data
    st.title("3 · Summenscores")
    st.subheader("Warum ist die Varianz eines Summenscores nicht einfach die Summe der Einzelvarianzen?")
    st.write(
        "Wir kombinieren **Wortschatz (X)** und **Satzverständnis (Y)** zum Summenscore "
        "**Sprachkompetenz (S = X + Y)**."
    )

    mode = st.radio("Modus", ["Datenmodus", "Simulationsmodus"], horizontal=True)

    if mode == "Datenmodus":
        x = df["Wortschatz"].astype(float).to_numpy()
        y = df["Satzverständnis"].astype(float).to_numpy()
        s = x+y
        vx, vy, covxy, vs = sample_var(x), sample_var(y), covariance(x,y), sample_var(s)

        metric_row([
            ("Var(X)", f"{vx:.2f}"),
            ("Var(Y)", f"{vy:.2f}"),
            ("Cov(X,Y)", f"{covxy:.2f}"),
            ("Var(X+Y)", f"{vs:.2f}")
        ])

        show = pd.DataFrame({
            "Person": df["Person"],
            "Wortschatz X": x,
            "Satzverständnis Y": y,
            "Summenscore S": s
        })
        st.dataframe(show, use_container_width=True, hide_index=True)

        parts = pd.DataFrame({
            "Beitrag": ["Var(X)", "Var(Y)", "2 × Cov(X,Y)"],
            "Wert": [vx, vy, 2*covxy]
        })
        fig = px.bar(parts, x="Beitrag", y="Wert", text_auto=".2f")
        fig.update_layout(height=360, yaxis_title="Beitrag zur Varianz des Summenscores")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

        st.info(f"{vx:.2f} + {vy:.2f} + 2 × {covxy:.2f} = **{vs:.2f}**")

    else:
        left, right = st.columns([1,1.5], gap="large")
        with left:
            vx = st.slider("Varianz von X", 0.5, 25.0, 4.0, 0.5)
            vy = st.slider("Varianz von Y", 0.5, 25.0, 4.0, 0.5)
            rxy = st.slider("Korrelation r(X,Y)", -1.0, 1.0, 0.0, 0.05)
            covxy = rxy*np.sqrt(vx)*np.sqrt(vy)
            vs = vx+vy+2*covxy
            metric_row([
                ("Cov(X,Y)", f"{covxy:.2f}"),
                ("Var(X+Y)", f"{vs:.2f}")
            ])
        with right:
            parts = pd.DataFrame({
                "Beitrag": ["Var(X)", "Var(Y)", "2 × Cov(X,Y)"],
                "Wert": [vx, vy, 2*covxy]
            })
            fig = px.bar(parts, x="Beitrag", y="Wert", text_auto=".2f")
            fig.update_layout(height=420, yaxis_title="Beitrag")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})
            st.write(f"**Gesamt:** {vx:.2f} + {vy:.2f} + ({2*covxy:.2f}) = **{vs:.2f}**")

        st.markdown("### Challenge")
        st.write("Setzen Sie **Var(X)=4**, **Var(Y)=4** und verändern Sie nur die Korrelation.")
        if abs(vx-4)<1e-9 and abs(vy-4)<1e-9:
            st.success("✓ Die Einzelvarianzen sind fixiert. Beobachten Sie nun, wie stark r die Summenvarianz verändert.")

    if st.session_state.show_formula:
        st.latex(r"Var(X+Y)=Var(X)+Var(Y)+2Cov(X,Y)")
        st.latex(r"Cov(X,Y)=r_{XY}s_Xs_Y")


# -----------------------------
# Module 5
# -----------------------------

    exam_questions("m4", [
        (
            "Warum ist Var(X+Y) im Allgemeinen nicht einfach Var(X)+Var(Y)?",
            "Weil die gemeinsame Variation beider Variablen mitberücksichtigt werden muss: Var(X+Y)=Var(X)+Var(Y)+2Cov(X,Y)."
        ),
        (
            "Was passiert mit der Varianz eines Summenscores, wenn die Kovarianz zwischen den Bestandteilen steigt?",
            "Sie steigt, weil beide Bestandteile dann stärker gemeinsam variieren."
        ),
        (
            "Kann ein Summenscore trotz stark streuender Einzelvariablen relativ wenig streuen?",
            "Ja. Bei stark negativer Kovarianz können sich die Schwankungen der Einzelvariablen teilweise gegenseitig aufheben."
        ),
    ])


def module5():
    version_badge()
    st.title("4 · Klassische Testtheorie")
    st.subheader("Vom beobachteten Wert zu Reliabilität und Messmodellen")
    st.caption(
        "Durchgehendes Beispiel: Ein Sprachtest soll die sprachliche Fähigkeit eines Kindes messen. "
        "Der beobachtete Testwert kann jedoch zusätzlich durch zufällige Fehler, Sprachverständnis der Instruktion, "
        "Müdigkeit oder Eigenschaften der Testform beeinflusst werden."
    )

    section = st.radio(
        "Lernschritt",
        [
            "4A · Wahrer Wert und Messfehler",
            "4B · Zwei parallele Tests",
            "4C · Reliabilität und Testverlängerung",
            "4D · Wann sind Tests parallel?",
        ],
        key="m5_section"
    )

    with st.expander("Begriffe kurz erklärt"):
        st.markdown(
            "- **Wahrer Wert (T):** stabiler erwarteter Testwert einer Person über sehr viele vergleichbare Messungen.\n"
            "- **Messfehler (E):** Anteil des beobachteten Werts, der nicht zum wahren Wert gehört.\n"
            "- **Wahre Varianz:** Unterschiede zwischen Personen in ihren wahren Werten.\n"
            "- **Fehlervarianz:** Unterschiede in den Fehleranteilen.\n"
            "- **Kovarianz:** gemeinsames Variieren zweier Messungen.\n"
            "- **Reliabilität:** Anteil wahrer Varianz an der beobachteten Varianz.\n"
            "- **Residuum:** beobachteter Wert minus vorhergesagter Wert in einer Regression.\n"
            "- **Fehlerquadratsumme (SSE):** Summe der quadrierten Residuen."
        )

    # =================================================
    # 5A
    # =================================================

    if section.startswith("4A"):
        st.markdown("### 4A · Wahrer Wert und Messfehler")
        st.write(
            "Beispiel Sprachtest: **T** steht für die stabile sprachliche Fähigkeit des Kindes, **E** für Einflüsse, "
            "die den beobachteten Sprachtestwert zusätzlich verändern. In der Klassischen Testtheorie gilt **X = T + E**. "
            "Entscheidend ist die Annahme, dass der Messfehler nicht systematisch mit dem wahren Wert zusammenhängt."
        )
        st.info(
            "**Was bedeutet Messfehler?** Nicht unbedingt ein Fehler des Tests oder der Testleitung. "
            "Gemeint ist der Anteil des beobachteten Testwerts, der nicht zum wahren Wert T gehört – "
            "zum Beispiel momentane Müdigkeit, Raten, Missverständnisse einzelner Aufgaben oder zufällige Schwankungen."
        )

        scenario = st.radio(
            "Szenario",
            [
                "Keine systematische Beziehung",
                "Deckeneffekt im Sprachtest",
                "Komplexe Instruktionen belasten sprachschwächere Kinder zusätzlich",
                "Testleitung hilft sprachschwächeren Kindern stärker",
                "Manuell einstellen",
            ],
            key="m5a_scenario"
        )

        c1, c2 = st.columns(2)
        var_t = c1.slider("Wahre Varianz Var(T)", 1.0, 100.0, 25.0, 1.0, key="m5a_vart")
        var_e = c2.slider("Grundrauschen Var(E)", 0.0, 100.0, 16.0, 1.0, key="m5a_vare")

        rho_manual = 0.0
        if scenario == "Manuell einstellen":
            rho_manual = st.slider(
                "Korrelation zwischen T und E",
                -0.8, 0.8, 0.0, 0.05,
                key="m5a_rho_manual"
            )

        n = st.slider("Simulierte Personen", 100, 1000, 500, 50, key="m5a_n")
        rng = np.random.default_rng(11)
        T = rng.normal(50, np.sqrt(var_t), size=n)
        noise = rng.normal(0, np.sqrt(var_e), size=n)
        ceiling = None

        if scenario == "Keine systematische Beziehung":
            E = noise
            X = T + E
            st.info("Die Fehler sind zufällig um 0 verteilt und hängen nicht systematisch von T ab.")

        elif scenario.startswith("Deckeneffekt"):
            raw_X = T + noise
            ceiling = 58.0
            X = np.minimum(raw_X, ceiling)
            E = X - T
            st.info(
                "Beispiel Sprachtest: Der Test hat maximal 58 Punkte. Kinder mit sehr hoher sprachlicher Fähigkeit "
                "können deshalb nicht entsprechend hohe beobachtete Werte erzielen. Rechts stauen sich die Werte am Maximum; "
                "bei hohen T-Werten werden die Fehler systematisch negativer."
            )

        elif scenario.startswith("Komplexe"):
            z = (T - T.mean()) / T.std(ddof=1)
            E = noise + 3.5 * z
            X = T + E
            st.info(
                "Beispiel Sprachtest: Der Test soll Wortschatz oder Sprachverständnis messen, aber die Aufgabeninstruktionen "
                "sind selbst sprachlich sehr komplex. Kinder mit niedrigerem T verstehen dadurch zusätzlich die Aufgaben schlechter. "
                "Dieser zusätzliche Nachteil wird Teil des Messfehlers und hängt systematisch mit T zusammen."
            )

        elif scenario.startswith("Testleitung"):
            z = (T - T.mean()) / T.std(ddof=1)
            E = noise - 3.5 * z
            X = T + E
            st.info(
                "Beispiel Sprachtest: Die Testleitung erklärt schwierige Aufgaben bei sprachschwächeren Kindern häufiger nach. "
                "Deren beobachtete Werte werden dadurch relativ zu T angehoben, während sprachstarke Kinder kaum Hilfe erhalten. "
                "T und E werden negativ korreliert."
            )

        else:
            if var_e > 0:
                zt = (T - T.mean()) / T.std(ddof=1)
                zi = rng.normal(0, 1, size=n)
                E = np.sqrt(var_e) * (
                    rho_manual * zt + np.sqrt(max(0, 1-rho_manual**2)) * zi
                )
            else:
                E = np.zeros(n)
            X = T + E

        cov_te = covariance(T, E)
        r_te = corr(T, E)

        metric_row([
            ("Var(T)", f"{sample_var(T):.2f}"),
            ("Var(E)", f"{sample_var(E):.2f}"),
            ("Cov(T,E)", f"{cov_te:.2f}"),
            ("r(T,E)", f"{r_te:.3f}"),
            ("Var(X)", f"{sample_var(X):.2f}")
        ])

        left, right = st.columns(2, gap="large")

        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=T, y=E, mode="markers",
                marker=dict(size=5, opacity=.30)
            ))
            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(
                height=430,
                xaxis=dict(title="Wahrer Wert T", range=teaching_range(T), fixedrange=True),
                yaxis=dict(title="Messfehler E", fixedrange=True),
                dragmode=False
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with right:
            hfig = go.Figure()
            hfig.add_trace(go.Histogram(x=T, nbinsx=28, opacity=.48, name="Wahre Werte T"))
            hfig.add_trace(go.Histogram(x=X, nbinsx=28, opacity=.38, name="Beobachtete Werte X"))
            if ceiling is not None:
                hfig.add_vline(x=ceiling, line_dash="dash", annotation_text="Testmaximum")
            hfig.update_layout(
                barmode="overlay",
                height=430,
                xaxis=dict(
                    title="Wert",
                    range=teaching_range(np.concatenate([T, X])),
                    fixedrange=True
                ),
                yaxis_title="Häufigkeit"
            )
            st.plotly_chart(hfig, use_container_width=True, config={"displayModeBar": False})

        if scenario.startswith("Deckeneffekt"):
            st.markdown(
                "**Was sollte auffallen?** Rechts stauen sich die beobachteten Werte am Testmaximum. "
                "Gleichzeitig fallen die Messfehler bei hohen T-Werten systematisch negativer aus."
            )
        elif scenario.startswith("Komplexe"):
            st.markdown(
                "**Was sollte auffallen?** Die Fehlerwolke zeigt nun eine systematische Steigung statt rein zufälligem Rauschen."
            )
        elif scenario.startswith("Testleitung"):
            st.markdown(
                "**Was sollte auffallen?** Niedrige T-Werte erhalten eher positive Fehler, hohe T-Werte eher negative – "
                "die beobachtete Verteilung wird zusammengedrückt."
            )

        if abs(r_te) < 0.05:
            st.success("✓ T und E sind praktisch unkorreliert.")
        else:
            st.warning("T und E hängen systematisch zusammen: Die KTT-Annahme Cov(T,E)=0 ist verletzt.")

        if st.session_state.show_formula:
            st.latex(r"X=T+E")
            st.latex(r"Var(X)=Var(T)+Var(E)+2Cov(T,E)")
            st.write("Nur bei Cov(T,E)=0 gilt:")
            st.latex(r"Var(X)=Var(T)+Var(E)")

        exam_questions("m5a", [
            (
                "Was bedeutet X=T+E bei einem Sprachtest?",
                "Der beobachtete Sprachtestwert X setzt sich aus dem wahren Wert T und einem Messfehler E zusammen."
            ),
            (
                "Warum nimmt die KTT Cov(T,E)=0 an?",
                "Der Messfehler soll nicht systematisch von der tatsächlichen Fähigkeit abhängen. Sonst wäre der Fehler nicht mehr rein zufällig."
            ),
            (
                "Was kann bei einem Deckeneffekt mit Cov(T,E) passieren?",
                "Bei hohen wahren Werten werden Fehler systematisch negativer, weil Werte am Testmaximum abgeschnitten werden. T und E können dadurch korrelieren."
            ),
        ])

    elif section.startswith("4B"):
        st.markdown("### 4B · Zwei parallele Tests: Wann stimmt die Reliabilitätsschätzung?")
        learning_question(
            "Wann entspricht die Korrelation zweier paralleler Testformen tatsächlich der Reliabilität?"
        )

        st.write(
            "Wir betrachten zwei parallele Formen desselben Sprachtests. Beide messen denselben wahren Wert T "
            "und besitzen dieselbe Fehlervarianz. Entscheidend ist nun: **Sind die Messfehler beider Formen unabhängig?**"
        )

        st.markdown("#### Experiment")
        c1, c2 = st.columns(2)
        with c1:
            var_t = st.slider(
                "Varianz der wahren Werte",
                1.0, 30.0, 10.0, 1.0,
                key="ktt4b_vart"
            )
        with c2:
            var_e = st.slider(
                "Fehlervarianz je Testform",
                1.0, 30.0, 10.0, 1.0,
                key="ktt4b_vare"
            )

        source = st.selectbox(
            "Gemeinsame Störquelle",
            [
                "Schwierige Instruktionen beeinflussen beide Testformen",
                "Müdigkeit oder Konzentrationsschwankungen wirken in beiden Testformen ähnlich",
                "Die gleiche Testsituation oder Testleitung beeinflusst beide Formen",
                "Manuelle Einstellung"
            ],
            key="ktt4b_source"
        )

        source_map = {
            "Schwierige Instruktionen beeinflussen beide Testformen": 0.45,
            "Müdigkeit oder Konzentrationsschwankungen wirken in beiden Testformen ähnlich": 0.30,
            "Die gleiche Testsituation oder Testleitung beeinflusst beide Formen": 0.20,
        }

        if source == "Manuelle Einstellung":
            rho_e = st.slider(
                "Korrelation der Messfehler",
                -0.80, 0.80, 0.00, 0.05,
                key="ktt4b_rhoe"
            )
        else:
            rho_e = source_map[source]
            st.caption(
                f"Für dieses Beispiel wird eine Fehlerkorrelation von **{rho_e:.2f}** simuliert. "
                "Die Auswahl verändert die Daten und die Grafiken darunter."
            )

        # Tatsächliche Reliabilität und theoretische Paralleltest-Schätzung
        rel_true = var_t / (var_t + var_e)
        cov_e = rho_e * var_e
        rel_est_theoretical = (var_t + cov_e) / (var_t + var_e)

        # Simulierte parallele Testformen
        rng = np.random.default_rng(4202)
        n = 800
        T = rng.normal(0, np.sqrt(var_t), n)
        z1 = rng.normal(0, 1, n)
        z2 = rng.normal(0, 1, n)

        E1 = np.sqrt(var_e) * z1
        E2 = np.sqrt(var_e) * (
            rho_e * z1 + np.sqrt(max(1e-9, 1-rho_e**2)) * z2
        )

        X1 = T + E1
        X2 = T + E2
        r12 = float(np.corrcoef(X1, X2)[0, 1])

        metric_row([
            ("Tatsächliche Reliabilität", f"{rel_true:.3f}"),
            ("Paralleltest-Schätzung r", f"{r12:.3f}"),
            ("Fehlerkorrelation", f"{rho_e:.2f}"),
            ("Schätzfehler", f"{(r12-rel_true):+.3f}")
        ])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X1,
            y=X2,
            mode="markers",
            marker=dict(size=6, opacity=0.50),
            name="Personen"
        ))
        fig.update_layout(
            height=440,
            xaxis=dict(title="Testform A", fixedrange=True),
            yaxis=dict(title="Testform B", fixedrange=True),
            dragmode=False,
            showlegend=False
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False, "scrollZoom": False}
        )

        comp = pd.DataFrame({
            "Kennwert": [
                "tatsächliche Reliabilität",
                "Paralleltest-Schätzung"
            ],
            "Wert": [rel_true, r12]
        })
        fig2 = px.bar(
            comp,
            x="Kennwert",
            y="Wert",
            text="Wert"
        )
        fig2.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )
        ymax = min(1.05, max(rel_true, r12, 0.1) + 0.15)
        fig2.update_layout(
            height=330,
            yaxis=dict(
                title="Reliabilität",
                range=[0, ymax],
                fixedrange=True
            ),
            xaxis=dict(title="", fixedrange=True),
            dragmode=False
        )
        st.plotly_chart(
            fig2,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        if abs(rho_e) < 1e-9:
            observation(
                "Bei unkorrelierten Messfehlern entspricht die Korrelation der beiden parallelen Testformen "
                "der tatsächlichen Reliabilität."
            )
        elif rho_e > 0:
            observation(
                "Die beiden Testformen ähneln sich nicht nur wegen des gemeinsamen wahren Werts, sondern zusätzlich "
                "wegen gemeinsam schwankender Messfehler. Die Paralleltest-Korrelation fällt deshalb zu hoch aus."
            )
        else:
            observation(
                "Die Messfehler wirken in den beiden Testformen gegenläufig. "
                "Die Paralleltest-Korrelation fällt deshalb zu niedrig aus."
            )

        explanation(
            "Die tatsächliche Reliabilität eines einzelnen Tests hängt vom Verhältnis wahrer Varianz zur gesamten "
            "beobachteten Varianz ab. Die Korrelation zweier paralleler Testformen ist nur dann eine unverzerrte "
            "Reliabilitätsschätzung, wenn ihre Messfehler unkorreliert sind."
        )

        st.markdown("#### Referenz: Was passiert ohne gemeinsame Fehler?")
        st.write(
            f"Wenn die Fehlerkorrelation **0** wäre, läge die erwartete Paralleltest-Korrelation bei "
            f"**{rel_true:.3f}**. Bei der aktuell gewählten Fehlerkorrelation von **{rho_e:.2f}** liegt der "
            f"theoretisch erwartete Wert bei **{rel_est_theoretical:.3f}**."
        )

        with formula_expander():
            st.latex(r"X_1=T+E_1,\qquad X_2=T+E_2")
            st.latex(r"Rel(X)=\frac{Var(T)}{Var(T)+Var(E)}")
            st.latex(r"Cov(X_1,X_2)=Var(T)+Cov(E_1,E_2)")
            st.latex(
                r"r_{X_1X_2}="
                r"\frac{Var(T)+Cov(E_1,E_2)}{Var(T)+Var(E)}"
            )
            st.write(
                "Nur wenn Cov(E₁,E₂)=0 gilt, entspricht die Paralleltest-Korrelation der tatsächlichen Reliabilität. "
                "Positive Fehlerkovarianz führt zur Überschätzung, negative zur Unterschätzung."
            )

        quick_check(
            "Was passiert bei positiv korrelierten Messfehlern mit der Paralleltest-Schätzung der Reliabilität?",
            [
                "Sie überschätzt die tatsächliche Reliabilität.",
                "Sie unterschätzt die tatsächliche Reliabilität.",
                "Sie bleibt unverändert."
            ],
            0,
            "Die gemeinsame Fehlervarianz erhöht die Korrelation zwischen den Testformen zusätzlich zum gemeinsamen wahren Wert.",
            "ktt4b_check"
        )

    elif section.startswith("4C"):
        st.markdown("### 4C · Reliabilität und Testverlängerung")
        st.write(
            "Beispiel Sprachtest: Testteil A enthält 20 Wortschatzitems, Testteil B weitere 20 möglichst vergleichbare Items. "
            "Wie verändert sich die Reliabilität, wenn wir beide Testteile zu einem längeren Sprachtest kombinieren? "
            "Hier kommt die Spearman–Brown-Formel ins Spiel."
        )
        st.info(
            "**Reliabilität:** Anteil der beobachteten Varianz, der auf stabile wahre Unterschiede zurückgeht. "
            "**Spearman–Brown:** Schätzt, wie sich die Reliabilität verändert, wenn ein Test mit vergleichbaren Teilen verlängert oder verkürzt wird."
        )

        c1, c2 = st.columns(2)
        with c1:
            var_t_a = st.slider("Wahre Varianz Testteil A", 1.0, 100.0, 25.0, 1.0, key="m5c_varta")
            var_e_a = st.slider("Fehlervarianz Testteil A", 0.0, 100.0, 25.0, 1.0, key="m5c_varea")
        with c2:
            var_t_b = st.slider("Wahre Varianz Testteil B", 1.0, 100.0, 25.0, 1.0, key="m5c_vartb")
            var_e_b = st.slider("Fehlervarianz Testteil B", 0.0, 100.0, 25.0, 1.0, key="m5c_vareb")

        rho_t = st.slider(
            "Korrelation der wahren Werte zwischen A und B",
            0.0, 1.0, 1.0, 0.05,
            key="m5c_rhot"
        )

        rng = np.random.default_rng(77)
        n = 1000
        cov_t = rho_t * np.sqrt(var_t_a * var_t_b)
        latent = rng.multivariate_normal([50, 50], [[var_t_a, cov_t], [cov_t, var_t_b]], size=n)
        TA, TB = latent[:,0], latent[:,1]
        EA = rng.normal(0, np.sqrt(var_e_a), size=n)
        EB = rng.normal(0, np.sqrt(var_e_b), size=n)
        A = TA + EA
        B = TB + EB
        S = A + B

        rel_a = var_t_a/(var_t_a+var_e_a) if var_t_a+var_e_a>0 else np.nan
        rel_b = var_t_b/(var_t_b+var_e_b) if var_t_b+var_e_b>0 else np.nan
        r_ab = corr(A, B)
        rel_sb = (2*r_ab)/(1+r_ab) if (1+r_ab)!=0 else np.nan

        true_sum = TA + TB
        err_sum = EA + EB
        rel_emp_sum = sample_var(true_sum) / sample_var(S) if sample_var(S)>0 else np.nan

        metric_row([
            ("Rel(A)", f"{rel_a:.3f}"),
            ("Rel(B)", f"{rel_b:.3f}"),
            ("r(A,B)", f"{r_ab:.3f}"),
            ("Spearman–Brown", f"{rel_sb:.3f}"),
            ("Reliabilität A+B", f"{rel_emp_sum:.3f}")
        ])

        st.markdown("#### Zwei Perspektiven")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Wenn A und B wirklich parallel sind:**")
            st.write(
                "Dann kann die Korrelation der beiden Testhälften mit Spearman–Brown auf die Reliabilität "
                "des Gesamttests hochgerechnet werden."
            )
        with c2:
            st.write("**Wenn A und B unterschiedliche wahre Werte abbilden:**")
            st.write(
                "Dann ist die einfache Spearman–Brown-Interpretation nicht mehr sauber. "
                "Genau deshalb können Sie hier die wahre Beziehung zwischen A und B verändern."
            )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=A, y=B, mode="markers", marker=dict(size=5, opacity=.25)))
        fig.update_layout(
            height=430,
            xaxis=dict(title="Testteil A", range=teaching_range(A), fixedrange=True),
            yaxis=dict(title="Testteil B", range=teaching_range(B), fixedrange=True),
            dragmode=False
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("#### Test verlängern")
        k = st.slider("Wie oft soll der Test verlängert werden?", 1, 5, 2, 1, key="m5c_k")
        sb_k = (k*r_ab)/(1+(k-1)*r_ab) if (1+(k-1)*r_ab)!=0 else np.nan
        st.metric(f"Spearman–Brown bei k = {k}", f"{sb_k:.3f}")

        if st.session_state.show_formula:
            st.latex(r"r_{SB}=\frac{2r_{AB}}{1+r_{AB}}")
            st.latex(r"r_{neu}=\frac{k\,r}{1+(k-1)r}")

        if abs(var_t_a-var_t_b) < 1e-9 and abs(var_e_a-var_e_b) < 1e-9 and abs(rho_t-1.0) < 1e-9:
            st.success("✓ A und B sind hier parallel – Spearman–Brown ist in seiner klassischen Form passend.")
        else:
            st.warning(
                "A und B sind nicht vollständig parallel. Beobachten Sie, wie sich Spearman–Brown und die "
                "tatsächlich simulierte Reliabilität des Summenscores auseinanderentwickeln können."
            )

        exam_questions("m5c", [
            (
                "Was bedeutet eine Reliabilität von .80?",
                "80 % der beobachteten Varianz werden als wahre Varianz interpretiert, 20 % als Fehlervarianz."
            ),
            (
                "Warum kann ein längerer Test reliabler sein?",
                "Wenn zusätzliche Testteile vergleichbare wahre Information beitragen und ihre Fehler sich nicht vollständig gemeinsam bewegen, steigt der Anteil stabiler wahrer Varianz relativ zum Fehler."
            ),
            (
                "Unter welchen Bedingungen ist die klassische Spearman–Brown-Formel besonders passend?",
                "Wenn die Testteile parallel bzw. hinreichend vergleichbar sind."
            ),
        ])

    # =================================================
    # 5D
    # =================================================
    else:
        st.markdown("### 4D · Wann sind zwei Tests parallel – und wann nicht?")

        st.markdown("#### Zuerst die Begriffe")
        st.markdown(
            "- **Parallel:** gleiche wahren Werte bzw. gleiche wahre Struktur **und** gleiche Fehlervarianz.\n"
            "- **Tau-äquivalent:** gleiche wahre Struktur, aber die Fehlervarianzen dürfen verschieden sein.\n"
            "- **Essential tau-equivalent:** die wahren Werte dürfen sich zusätzlich um eine feste Konstante unterscheiden.\n"
            "- **Kongenerisch:** beide Tests messen dasselbe Merkmal, aber die wahren Personenunterschiede können unterschiedlich stark abgebildet werden."
        )

        st.markdown("#### Wie kann das bei Sprachtests in der Praxis passieren?")
        st.write(
            "**Parallel:** Zwei Wortschatzformen sind gleich schwer, gleich präzise und bilden dieselben sprachlichen Unterschiede ab. "
            "**Tau-äquivalent:** Beide erfassen dieselben wahren Unterschiede, aber eine Form enthält mehr missverständliche Items und damit mehr Messfehler. "
            "**Essential tau-equivalent:** Eine Form ist insgesamt etwas leichter, sodass alle Kinder im wahren Wert ungefähr gleich viel höher liegen. "
            "**Kongenerisch:** Eine Form trennt besonders stark zwischen sprachschwachen und sprachstarken Kindern, etwa weil sie deutlich schwierigere Wörter enthält."
        )

        example = st.radio(
            "Beispiel auswählen",
            [
                "Gleiche Schwierigkeit und gleiches Messrauschen",
                "Eine Form ist unzuverlässiger",
                "Eine Form ist insgesamt leichter",
                "Eine Form differenziert stärker zwischen Personen",
            ],
            key="m5d_example"
        )
        if example.startswith("Gleiche"):
            st.info("Typischer Fall für parallele Testformen.")
        elif example.startswith("Eine Form ist unzuverlässiger"):
            st.info(
                "Beispiel: Test B enthält missverständlichere Items. Die wahren Werte bleiben gleich, "
                "aber Var(E_B) ist grösser → nicht mehr parallel, sondern tau-äquivalent."
            )
        elif example.startswith("Eine Form ist insgesamt leichter"):
            st.info(
                "Beispiel: Alle Personen erzielen in Test B ungefähr fünf wahre Punkte mehr. "
                "Die Personenunterschiede bleiben gleich → essential tau-equivalent."
            )
        else:
            st.info(
                "Beispiel: Test B enthält besonders schwierige Items und trennt starke von schwachen Personen "
                "deutlicher als Test A. Die wahren Unterschiede werden unterschiedlich stark abgebildet → kongenerisch."
            )

        st.markdown("#### Jetzt selbst einstellen")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Test A**")
            shift_a = st.slider("Konstante Verschiebung wahrer Werte A", -10.0, 10.0, 0.0, 1.0, key="m5d_shift_a")
            scale_a = st.slider("Stärke wahrer Personenunterschiede A", 0.5, 1.5, 1.0, 0.1, key="m5d_scale_a")
            err_a = st.slider("Fehlervarianz A", 1.0, 50.0, 10.0, 1.0, key="m5d_err_a")
        with c2:
            st.markdown("**Test B**")
            shift_b = st.slider("Konstante Verschiebung wahrer Werte B", -10.0, 10.0, 0.0, 1.0, key="m5d_shift_b")
            scale_b = st.slider("Stärke wahrer Personenunterschiede B", 0.5, 1.5, 1.0, 0.1, key="m5d_scale_b")
            err_b = st.slider("Fehlervarianz B", 1.0, 50.0, 10.0, 1.0, key="m5d_err_b")

        same_shift = abs(shift_a-shift_b) < 1e-9
        same_scale = abs(scale_a-scale_b) < 1e-9
        same_error = abs(err_a-err_b) < 1e-9

        if same_shift and same_scale and same_error:
            model = "Parallel"
            explanation = "Gleiche wahre Struktur und gleiche Fehlervarianz."
        elif same_shift and same_scale:
            model = "Tau-äquivalent"
            explanation = "Gleiche wahre Struktur, aber unterschiedliche Fehlervarianzen."
        elif same_scale:
            model = "Essential tau-equivalent"
            explanation = "Wahre Werte unterscheiden sich nur um eine Konstante."
        else:
            model = "Kongenerisch"
            explanation = "Dasselbe Merkmal, aber wahre Unterschiede werden unterschiedlich stark abgebildet."

        st.success(f"**Aktuelles Modell: {model}**")
        st.write(explanation)

        st.markdown("#### Aufgabe")
        st.write(
            "Erzeugen Sie nacheinander ein **tau-äquivalentes**, ein **essential tau-äquivalentes** "
            "und ein **kongenerisches** Modell. Stellen Sie danach wieder **Parallelität** her."
        )

        with st.expander("Warum ist das später für Alpha und Omega wichtig?"):
            st.write(
                "Diese Modellannahmen bestimmen, wie austauschbar einzelne Testteile oder Items sind. "
                "Bei strengeren Annahmen funktionieren klassische Reliabilitätsinterpretationen einfacher; "
                "bei kongenerischen Messungen werden modellbasierte Reliabilitätsmasse wichtiger."
            )

        exam_questions("m5d", [
            (
                "Wodurch unterscheiden sich parallele und tau-äquivalente Tests?",
                "Bei parallelen Tests sind auch die Fehlervarianzen gleich. Tau-äquivalente Tests dürfen unterschiedliche Fehlervarianzen besitzen."
            ),
            (
                "Zwei Wortschatzformen messen dieselben wahren Unterschiede, aber eine Form ist unzuverlässiger. Welches Modell passt?",
                "Tau-äquivalent."
            ),
            (
                "Zwei Tests messen dasselbe Merkmal, bilden Personenunterschiede aber unterschiedlich stark ab. Welches Modell passt?",
                "Kongenerisch."
            ),
        ])




def difficulty_extensions():
    with st.expander("Ausprobieren: Mehrstufige Items, Ratekorrektur und Speedtests"):
        likert_tab, guessing_tab, speed_tab = st.tabs(
            ["Mehrstufige Items", "Ratekorrektur", "Speedtests"]
        )
        with likert_tab:
            st.markdown("#### Von den Antworten zur Itemschwierigkeit")
            st.write("Verändern Sie die Antworten in der Tabelle. Berechnen Sie anschließend die Schwierigkeit in drei Schritten: erst umkodieren, dann mitteln, dann durch den maximal erreichbaren Wert teilen.")
            answers = st.data_editor(
                pd.DataFrame({"Person": NAMES, "Antwort (1–4)": [1, 2, 3, 4, 3, 2, 4, 1, 3, 2]}),
                disabled=["Person"], hide_index=True, key="m6_likert_answers_v2",
                column_config={"Antwort (1–4)": st.column_config.NumberColumn(min_value=1, max_value=4, step=1, required=True)},
            )
            values = answers["Antwort (1–4)"].to_numpy(dtype=float)
            item_mean = float(np.mean(values))
            p = (item_mean - 1) / 3
            stage = st.radio("Rechenschritt", ["1 · Minimum abziehen", "2 · Mittelwert bilden", "3 · Durch Maximum teilen"], horizontal=True, key="m6_likert_stage")
            st.dataframe(pd.DataFrame({"Person": NAMES, "Originalantwort": values, "Umkodiert: Antwort − 1": values - 1}), hide_index=True)
            if stage.startswith("1"):
                st.info("Aus 1, 2, 3, 4 wird 0, 1, 2, 3. Die niedrigste Antwort liefert jetzt 0 Punkte, die höchste 3 Punkte.")
            elif stage.startswith("2"):
                st.metric("Mittelwert nach Umkodierung", f"{item_mean-1:.2f}")
                st.write(f"Summe der umkodierten Antworten / Personen: {np.sum(values-1):.0f} / 10 = {item_mean-1:.2f}.")
            else:
                metric_row([("Mittelwert nach Umkodierung", f"{item_mean-1:.2f}"), ("Maximal erreichbar", "3"), ("Schwierigkeit p", f"{p:.3f}")])
                st.write(f"{item_mean-1:.2f} / 3 = **{p:.3f}**. Die Schwierigkeit liegt zwischen 0 und 1.")
                st.warning(f"Direkt den ursprünglichen Mittelwert durch 4 teilen ergäbe {item_mean/4:.3f}. Selbst wenn alle Personen 1 wählen, käme dabei 0,25 statt 0 heraus.")
            st.latex(r"p_i=\frac{\bar X_i-X_{\min}}{X_{\max}-X_{\min}}")
            st.caption("Bei Likert-Items bedeutet ein hoher Wert hohe Zustimmung bzw. Merkmalsausprägung. Die Richtung muss vorher einheitlich kodiert sein; negativ gepolte Items werden gegebenenfalls umgepolt. Die Berechnung behandelt die Stufen als gleich weit auseinander.")
        with guessing_tab:
            st.write("Vorhersage: Wie viele richtige Antworten bleiben nach der Korrektur für zufälliges Raten übrig?")
            k = st.slider("Antwortoptionen k (genau eine richtig)", 2, 6, 4, key="m6_guess_k")
            right = st.slider("Richtige Antworten R", 0, 100, 25, key="m6_guess_right")
            remaining = 100 - right
            wrong_key = "m6_guess_wrong"
            if wrong_key in st.session_state:
                st.session_state[wrong_key] = min(int(st.session_state[wrong_key]), remaining)
            else:
                st.session_state[wrong_key] = min(75, remaining)
            wrong = st.number_input("Falsche Antworten F", min_value=0, max_value=remaining,
                                    step=1, key=wrong_key)
            omitted = 100 - right - wrong
            corrected_right = right - wrong / (k - 1)
            p = corrected_right / 100
            st.caption(f"N = 100 Personen; nicht beantwortet: {omitted}. Nicht beantwortete Items zählen nicht als F, bleiben aber im Nenner N.")
            st.latex(r"P_{\mathrm{korr}}=\frac{R-\frac{F}{k-1}}{N}")
            metric_row([("Unkorrigiert R/N", f"{right/100:.3f}"), ("Geschätzte Ratetreffer F/(k−1)", f"{wrong/(k-1):.2f}"), ("Ratekorrigiert", f"{p:.3f}")])
            st.write(f"({right} − {wrong}/({k} − 1)) / 100 = **{p:.3f}**")
            st.info("Die Korrektur setzt blindes Raten mit gleich wahrscheinlichen Antwortoptionen voraus. Teilwissen und das Ausschließen von Optionen verändern die Ratewahrscheinlichkeit. Ein negativer Wert bleibt sichtbar und wird nicht auf 0 gesetzt.")
        with speed_tab:
            st.write("Vorhersage: Werden späte Items scheinbar schwieriger, wenn immer weniger Personen sie innerhalb der Zeit erreichen?")
            time = st.slider("Zeitlimit (Sekunden)", 4, 16, 8, key="m6_speed_time")
            errors = st.slider("Personen mit falscher Antwort pro Item (ohne Zeitlimit)", 0, 4, 0, key="m6_speed_errors")
            chosen = st.selectbox("Welches Item untersuchen?", list(range(1, 9)), index=7, key="m6_speed_item")
            handling = st.radio("Nicht erreichte Items bei der Schwierigkeit", ["Als falsch zählen", "Aus dem Nenner ausschließen"], horizontal=True, key="m6_speed_handling")
            # Potential correctness is fixed; only reaching the items changes with time.
            potential = np.array([0] * errors + [1] * (12-errors))
            speed = np.linspace(0.5, 1.25, 12)
            positions = np.arange(1, 9)
            reached = positions[None, :] <= np.floor(time * speed[:, None])
            correct = np.column_stack([np.roll(potential, j) for j in range(8)]).astype(bool)
            shown = np.where(reached, np.where(correct, "richtig", "falsch"), "nicht erreicht")
            n_reached = reached.sum(axis=0)
            n_right = (reached & correct).sum(axis=0)
            conditional = np.divide(n_right, n_reached, out=np.full(8, np.nan), where=n_reached > 0)
            j = chosen - 1
            displayed = n_right[j]/12 if handling == "Als falsch zählen" else conditional[j]
            metric_row([("Richtig", str(n_right[j])), ("Falsch beantwortet", str(n_reached[j]-n_right[j])),
                        ("Nicht erreicht", str(12-n_reached[j])), (f"Schwierigkeit Item {chosen}", f"{displayed:.3f}" if np.isfinite(displayed) else "nicht definiert")])
            denominator = 12 if handling == "Als falsch zählen" else int(n_reached[j])
            st.write(f"Item {chosen}: **{n_right[j]} richtige Antworten / {denominator} Personen**. Verändern Sie das Zeitlimit und vergleichen Sie die beiden Auswertungen.")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=positions, y=n_right/12, mode="lines+markers", name="Nicht erreicht = falsch"))
            fig.add_trace(go.Scatter(x=positions, y=conditional, mode="lines+markers", name="Nur Personen, die das Item erreicht haben"))
            fig.update_layout(height=330, xaxis=dict(title="Itemposition", tickvals=positions, fixedrange=True), yaxis=dict(title="Itemschwierigkeit p", range=[0, 1.05], fixedrange=True), legend=dict(orientation="h", y=1.2), dragmode=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.dataframe(pd.DataFrame(shown, index=[f"Person {j+1}" for j in range(12)], columns=[f"Item {j}" for j in positions]))
            st.dataframe(pd.DataFrame({"Item": positions, "Erreicht": n_reached, "Richtig": n_right,
                "p: alle Personen": n_right / 12, "p: nur erreichte Items": conditional}), hide_index=True)
            st.latex(r"p_{\mathrm{alle}}=\frac{R}{N},\qquad p_{\mathrm{erreicht}}=\frac{R}{N_{\mathrm{erreicht}}}")
            st.write("Nicht erreichte Items werden für den Testscore häufig als 0 gewertet. Zur Analyse der Itemschwierigkeit muss man sie von tatsächlich falsch beantworteten Items unterscheiden. Bei reinen Speedtests soll gerade die Bearbeitungsgeschwindigkeit unterscheiden; die Items sind entsprechend meist sehr leicht.")
            st.caption(f"Ohne Zeitlimit sind pro Item {12-errors} von 12 Antworten richtig. Mit 0 falschen Antworten entsteht ein reiner Speedtest: Die Unterschiede im Score kommen ausschließlich aus dem Tempo. Unter Zeitdruck antwortet eine ausgewählte Gruppe; der Lösungsanteil unter den Bearbeitenden ist deshalb kein automatisch unverzerrter Schätzer. Ohne Bearbeitende ist er nicht definiert.")


def part_whole_experiment():
    with st.expander("Ausprobieren: Trennschärfe mit und ohne Part-Whole-Korrektur"):
        st.write("Vorhersage: Kann ein Zufallsitem mit dem Gesamtscore korrelieren, obwohl es mit den anderen Items nichts gemeinsam hat?")
        count = st.slider("Anzahl Items insgesamt", 2, 20, 4, key="m6_pw_count")
        kind = st.radio("Untersuchtes Item", ["Zufallsitem", "Item mit gemeinsamem Fähigkeitsbezug"], horizontal=True, key="m6_pw_kind")
        rng = np.random.default_rng(1)
        ability = rng.normal(size=400)
        rest_items = (0.8 * ability[:, None] + rng.normal(size=(400, 19)) > 0).astype(float)
        noise = rng.normal(size=400)
        item = (noise > 0).astype(float) if kind == "Zufallsitem" else (0.8 * ability + noise > 0).astype(float)
        rest = rest_items[:, :count-1].sum(axis=1)
        total = rest + item
        raw = corr(item, total)
        corrected = corr(item, rest)
        metric_row([("Itemschwierigkeit", f"{item.mean():.3f}"), ("Unkorrigiert: Item ↔ Gesamtscore", f"{raw:.3f}"), ("Korrigiert: Item ↔ Restscore", f"{corrected:.3f}")])
        st.latex(r"S=X_i+S_{-i},\qquad r_{i,S}=Cor(X_i,S),\qquad r_{i,S-i}=Cor(X_i,S-X_i)")
        columns = st.columns(2)
        for col, score, title, r in zip(columns, [total, rest], ["Gesamtscore: Item enthalten", "Restscore: Item entfernt"], [raw, corrected]):
            pairs = pd.DataFrame({"Itemantwort": item, "Score": score}).value_counts().reset_index(name="Personen")
            fig = go.Figure(go.Scatter(x=pairs["Score"], y=pairs["Itemantwort"], mode="markers",
                marker=dict(size=8 + 3*np.sqrt(pairs["Personen"])),
                customdata=pairs["Personen"], hovertemplate="Score: %{x}<br>Item: %{y}<br>Personen: %{customdata}<extra></extra>"))
            fig.update_layout(title=f"{title} · r = {r:.3f}", height=330,
                xaxis=dict(title="Testwert", range=[-0.5, count+0.5], fixedrange=True),
                yaxis=dict(title="Itemantwort", tickvals=[0, 1], range=[-0.25, 1.25], fixedrange=True), dragmode=False)
            col.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("Größere Punkte stehen für mehr Personen mit derselben Antwort und demselben Score. Beide Diagramme verwenden dieselbe x-Skala.")
        st.dataframe(pd.DataFrame({"Person": np.arange(1, 401), "Item": item.astype(int), "Gesamtscore S": total.astype(int), "Restscore S − Item": rest.astype(int)}), hide_index=True, height=220)
        st.latex(r"Cov(X_i,S)=Var(X_i)+Cov(X_i,S_{-i})")
        st.info("Der Gesamtscore enthält die Antwort auf das untersuchte Item. Dessen eigene Varianz erhöht bereits die Kovarianz mit dem Gesamtscore. Die Part-Whole-Korrektur entfernt diesen Eigenanteil, indem das Item aus dem Score herausgerechnet wird.")
        st.write("Verändern Sie die Testlänge und wechseln Sie zwischen dem Zufallsitem und einem Item mit Fähigkeitsbezug. Die 400 Personen bleiben dabei dieselben. Beim Zufallsitem liegt die korrigierte Trennschärfe typischerweise nahe 0. Bei kurzen Tests fällt der Eigenanteil besonders auf.")


def module6():
    version_badge()
    st.title("5 · Itemanalyse")
    st.subheader("Itemschwierigkeit und Trennschärfe")
    st.caption(
        "Durchgehendes Beispiel: Ein Sprachtest besteht aus Wortschatz- und Sprachverständnisitems. "
        "Wir untersuchen zwei zentrale Kennwerte der klassischen Itemanalyse: **Itemschwierigkeit** und **Trennschärfe**. Anschließend kombinieren wir verschiedene Itemtypen zu einem Test."
    )

    section = st.radio(
        "Lernschritt",
        [
            "5A · Itemschwierigkeit",
            "5B · Trennschärfe",
            "5C · Einen Test zusammenstellen",
        ],
        key="m6_section"
    )

    with st.expander("Wie wird simuliert?"):
        st.write(
            "Für die Simulation wird im Hintergrund festgelegt, wie stark eine Itemantwort mit der zugrunde liegenden "
            "Fähigkeit zusammenhängt. Dieser technische Simulationsparameter ist **nicht identisch mit der beobachteten "
            "Trennschärfe**. Die Trennschärfe wird erst aus den erzeugten Itemantworten und dem restlichen Test berechnet."
        )

    rng = np.random.default_rng(2026)

    # ==========================================================
    # 6A
    # ==========================================================
    if section.startswith("5A"):
        st.markdown("### 5A · Itemschwierigkeit")
        difficulty_extensions()
        st.info(
            "**Itemschwierigkeit p:** Anteil der Personen, die ein dichotomes Item richtig lösen. "
            "Ein Item mit p = .80 ist leicht, ein Item mit p = .50 mittelschwer und ein Item mit p = .20 schwer."
        )
        st.caption(
            "In diesem Abschnitt halten wir die Trennschärfe aller Items ungefähr konstant im mittleren Bereich. "
            "So lässt sich isoliert beobachten, wie die Itemschwierigkeit die Testwertverteilung beeinflusst."
        )

        c1, c2, c3 = st.columns(3)
        n_easy = c1.slider("Leichte Items (p≈.80)", 0, 20, 6, 1, key="m6a_easy")
        n_mid = c2.slider("Mittlere Items (p≈.50)", 0, 20, 8, 1, key="m6a_mid")
        n_hard = c3.slider("Schwere Items (p≈.20)", 0, 20, 6, 1, key="m6a_hard")

        n_items = n_easy + n_mid + n_hard
        n_people = 600

        if n_items == 0:
            st.warning("Wählen Sie mindestens ein Item.")
        else:
            ability = rng.normal(0, 1, n_people)
            target_ps = [0.80]*n_easy + [0.50]*n_mid + [0.20]*n_hard

            # fixed moderate differentiation; target p stays approximately constant
            differentiation = 0.45
            items = []
            observed_ps = []
            for p in target_ps:
                resp = simulate_ctt_item(ability, p, differentiation, rng)
                items.append(resp)
                observed_ps.append(resp.mean())

            mat = np.column_stack(items)
            scores = mat.sum(axis=1)

            observed_rit = []
            for j in range(n_items):
                rest = scores - mat[:, j]
                if np.std(mat[:, j]) > 0 and np.std(rest) > 0:
                    observed_rit.append(np.corrcoef(mat[:, j], rest)[0, 1])
                else:
                    observed_rit.append(np.nan)

            # Item-level overview: same latent differentiation, different difficulty
            difficulty_labels = (["leicht"] * n_easy) + (["mittel"] * n_mid) + (["schwer"] * n_hard)
            a_summary = pd.DataFrame({
                "Schwierigkeit": difficulty_labels,
                "p beobachtet": observed_ps,
                "Trennschärfe beobachtet": observed_rit,
                "Erwarteter Zusammenhang": [
                    theoretical_item_ability_corr(p, differentiation) for p in target_ps
                ],
                "Maximaler Zusammenhang bei dieser Schwierigkeit": [
                    theoretical_item_ability_corr(p, 0.999) for p in target_ps
                ],
            })

            metric_row([
                ("Items gesamt", str(n_items)),
                ("Mittlere Schwierigkeit p", f"{np.mean(observed_ps):.2f}"),
                ("Mittlere beobachtete Trennschärfe", f"{np.nanmean(observed_rit):.2f}"),
                ("Mittelwert Test", f"{scores.mean():.2f}"),
                ("SD Test", f"{scores.std(ddof=1):.2f}")
            ])

            st.markdown("#### Wie die Itemschwierigkeit die beobachtete Trennschärfe beeinflusst")
            grouped_a = (
                a_summary.groupby("Schwierigkeit", observed=True)
                .agg(
                    Items=("p beobachtet", "size"),
                    p=("p beobachtet", "mean"),
                    Trennschärfe=("Trennschärfe beobachtet", "mean"),
                    Erwarteter_Zusammenhang=("Erwarteter Zusammenhang", "mean"),
                    Theoretisches_Maximum=("Maximaler Zusammenhang bei dieser Schwierigkeit", "mean"),
                )
                .reset_index()
            )
            for col in ["p", "Trennschärfe", "Erwarteter_Zusammenhang", "Theoretisches_Maximum"]:
                grouped_a[col] = grouped_a[col].round(2)
            st.dataframe(
                grouped_a[["Schwierigkeit", "Items", "p", "Trennschärfe"]],
                use_container_width=True, hide_index=True
            )
            with st.expander("Vertiefung: Warum kann extreme Schwierigkeit die Trennschärfe begrenzen?"):
                tech_a = grouped_a[["Schwierigkeit", "Erwarteter_Zusammenhang", "Theoretisches_Maximum"]].rename(
                    columns={
                        "Erwarteter_Zusammenhang": "Erwarteter Zusammenhang mit Fähigkeit",
                        "Theoretisches_Maximum": "Theoretisches Maximum"
                    }
                )
                st.dataframe(tech_a, use_container_width=True, hide_index=True)
                st.write(
                    "Bei sehr leichten oder sehr schweren Items antworten fast alle Personen gleich. "
                    "Dadurch ist weniger Variation vorhanden und eine hohe beobachtete Item-Test-Korrelation wird schwieriger."
                )

            fig = px.histogram(x=scores, nbins=max(10, n_items+1))
            score_range = [0, max(n_items, int(np.max(scores)))] if st.session_state.get("zero_based_axes", True) else padded_range(scores, min_pad=1, frac=.08)
            fig.update_layout(
                height=430,
                xaxis=dict(title="Gesamttestwert", range=score_range, fixedrange=True),
                yaxis=dict(title="Häufigkeit", fixedrange=True)
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            mean_p = np.mean(observed_ps)
            if mean_p > .65:
                st.warning(
                    "Der Test enthält überwiegend leichte Items. Die Testwerte verschieben sich nach oben; "
                    "wenn die übrigen Itemeigenschaften vergleichbar bleiben, werden Deckeneffekte wahrscheinlicher."
                )
            elif mean_p < .35:
                st.warning(
                    "Der Test enthält überwiegend schwere Items. Die Testwerte verschieben sich nach unten; "
                    "wenn die übrigen Itemeigenschaften vergleichbar bleiben, werden Bodeneffekte wahrscheinlicher."
                )
            else:
                st.success(
                    "Die mittlere Schwierigkeit liegt im mittleren Bereich. "
                    "Die Testwerte verteilen sich stärker um die Mitte."
                )

            st.caption(
                "Wichtig: Die beobachtete klassische Trennschärfe hängt auch von der Schwierigkeit ab. "
                "Sehr leichte oder sehr schwere Items haben weniger Varianz und dadurch oft weniger Spielraum für sehr hohe Trennschärfen."
            )

        exam_questions("m6a", [
            (
                "Was bedeutet p = .80 bei einem dichotomen Item?",
                "Etwa 80 % der Personen lösen das Item richtig; das Item ist relativ leicht."
            ),
            (
                "Was bedeutet es für die Testwerte, wenn die Items für die untersuchte Gruppe überwiegend leicht sind?",
                "Viele Personen lösen viele Items richtig; die Testwerte liegen daher eher hoch. Wenn viele Personen die Höchstpunktzahl erreichen, kann der Test Unterschiede zwischen diesen Personen nicht mehr abbilden (Deckeneffekt)."
            ),
            (
                "Warum reicht die Itemschwierigkeit allein nicht aus, um die Qualität eines Items zu beurteilen?",
                "Zusätzlich sind die Trennschärfe, der Inhalt und der Testzweck wichtig. Ein leichtes oder schweres Item ist nicht allein wegen seiner Schwierigkeit ungeeignet."
            ),
        ])

    # ==========================================================
    # 6B
    # ==========================================================
    elif section.startswith("5B"):
        st.markdown("### 5B · Trennschärfe")
        part_whole_experiment()
        st.info(
            "**Trennschärfe:** Zusammenhang zwischen einem Item und dem restlichen Test. "
            "Eine hohe positive Trennschärfe bedeutet: Personen mit insgesamt hoher Testleistung lösen das Item häufiger."
        )
        st.caption(
            "Hier halten wir die Itemschwierigkeit ungefähr im mittleren Bereich (p≈.50) und verändern, wie gut das Item zwischen insgesamt stärkeren und schwächeren Personen unterscheidet. Die daraus beobachtete **Trennschärfe** wird direkt angezeigt."
        )

        discrimination = st.slider("Trennschärfe beeinflussen", -1.0, 1.0, 0.5, 0.05, key="m6b_disc")
        n = 700
        ability = rng.normal(0, 1, n)

        sign = -1 if discrimination < 0 else 1
        rho = min(0.90, abs(discrimination) * 0.85)
        eps = rng.normal(0, 1, n)
        latent_item = sign * rho * ability + np.sqrt(max(1e-9, 1-rho**2)) * eps
        threshold = norm_ppf(0.50)
        item = (latent_item > threshold).astype(int)

        rest_score = ability + rng.normal(0, .7, n)
        rit = np.corrcoef(item, rest_score)[0,1] if np.std(item)>0 else np.nan

        df = pd.DataFrame({"Gesamtleistung": rest_score, "Item": item})
        groups = pd.qcut(df["Gesamtleistung"], 3, labels=["niedrig", "mittel", "hoch"])
        rates = df.groupby(groups, observed=True)["Item"].mean().reset_index()

        theo_corr = theoretical_item_ability_corr(0.50, sign * rho)
        metric_row([
            ("Beobachtete Schwierigkeit p", f"{item.mean():.2f}"),
            ("Trennschärfe", f"{sign*rho:.2f}"),
            ("Erwarteter Zusammenhang", f"{theo_corr:.2f}"),
            ("Beobachtete Trennschärfe", f"{rit:.2f}")
        ])

        fig = px.bar(rates, x="Gesamtleistung", y="Item")
        fig.update_layout(
            height=400,
            xaxis_title="Gesamttestleistung",
            yaxis_title="Anteil richtig",
            yaxis_range=[0,1]
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if rit > .3:
            st.success(
                "Hohe positive Trennschärfe: Personen mit hoher Gesamttestleistung lösen das Item deutlich häufiger."
            )
        elif rit < 0:
            st.warning(
                "Negative Trennschärfe: Personen mit niedriger Gesamttestleistung lösen das Item häufiger. "
                "Das Item sollte überprüft werden."
            )
        else:
            st.info(
                "Niedrige Trennschärfe: Das Item unterscheidet nur wenig zwischen insgesamt stärkeren und schwächeren Personen."
            )

        exam_questions("m6b", [
            (
                "Was bedeutet eine hohe positive Trennschärfe?",
                "Personen mit hoher Gesamttestleistung lösen das Item deutlich häufiger als Personen mit niedriger Gesamttestleistung."
            ),
            (
                "Was bedeutet eine Trennschärfe nahe 0?",
                "Das Item unterscheidet kaum zwischen insgesamt stärkeren und schwächeren Personen."
            ),
            (
                "Warum ist eine negative Trennschärfe problematisch?",
                "Weil schwächere Personen das Item häufiger richtig lösen als stärkere. Das kann auf Missverständnisse, falsche Kodierung oder ein unpassendes Item hindeuten."
            ),
        ])

    # ==========================================================
    # 6C
    # ==========================================================
    else:
        st.markdown("### 5C · Einen Test zusammenstellen")
        st.write(
            "Jetzt kombinieren Sie Itemschwierigkeit und Trennschärfe. "
            "Für jede der neun Kombinationen legen Sie fest, wie viele Items davon im Sprachtest vorkommen sollen."
        )

        st.markdown("#### 3 × 3-Matrix der Itemtypen")

        col_label, c_low, c_mid, c_high = st.columns([1.35, 1, 1, 1])
        col_label.markdown("**Schwierigkeit**")
        c_low.markdown("**niedrige Trennschärfe**")
        c_mid.markdown("**mittlere Trennschärfe**")
        c_high.markdown("**hohe Trennschärfe**")

        matrix = {}
        rows = [
            ("leicht", 0.80),
            ("mittel", 0.50),
            ("schwer", 0.20),
        ]
        discs = [
            ("niedrig", 0.20),
            ("mittel", 0.50),
            ("hoch", 0.80),
        ]

        defaults = {
            ("leicht","niedrig"): 0, ("leicht","mittel"): 3, ("leicht","hoch"): 2,
            ("mittel","niedrig"): 0, ("mittel","mittel"): 5, ("mittel","hoch"): 5,
            ("schwer","niedrig"): 0, ("schwer","mittel"): 3, ("schwer","hoch"): 2,
        }

        for row_name, p in rows:
            label, a, b, c = st.columns([1.35,1,1,1])
            label.markdown(f"**{row_name.capitalize()}**")
            for container, (disc_name, disc_value) in zip([a,b,c], discs):
                with container:
                    matrix[(row_name, disc_name)] = st.number_input(
                        "Items",
                        min_value=0,
                        max_value=20,
                        value=defaults[(row_name, disc_name)],
                        step=1,
                        key=f"m6c_{row_name}_{disc_name}",
                        label_visibility="collapsed"
                    )

        item_specs = []
        for row_name, p in rows:
            for disc_name, disc_value in discs:
                count = int(matrix[(row_name, disc_name)])
                item_specs.extend([(p, disc_value, row_name, disc_name)] * count)

        n_items = len(item_specs)
        n_people = st.slider("Anzahl simulierter Personen", 200, 1200, 600, 50, key="m6c_people")

        if n_items == 0:
            st.warning("Wählen Sie mindestens ein Item.")
        else:
            ability = rng.normal(0, 1, n_people)
            responses = []
            observed_rit = []
            observed_p = []

            for p, disc_value, row_name, disc_name in item_specs:
                resp = simulate_ctt_item(ability, p, disc_value, rng)
                responses.append(resp)
                observed_p.append(resp.mean())

            mat = np.column_stack(responses)
            total = mat.sum(axis=1)

            # corrected item-total correlations
            for j in range(n_items):
                rest = total - mat[:,j]
                if np.std(mat[:,j]) > 0 and np.std(rest) > 0:
                    observed_rit.append(np.corrcoef(mat[:,j], rest)[0,1])
                else:
                    observed_rit.append(np.nan)

            item_vars = mat.var(axis=0, ddof=1)
            total_var = total.var(ddof=1)
            alpha = (
                (n_items/(n_items-1))*(1-item_vars.sum()/total_var)
                if n_items > 1 and total_var > 0 else np.nan
            )


            item_summary = pd.DataFrame({
                "Schwierigkeit": [spec[2] for spec in item_specs],
                "Trennschärfe-Kategorie": [spec[3] for spec in item_specs],
                "p beobachtet": observed_p,
                "Trennschärfe beobachtet": observed_rit,
                "Erwarteter Zusammenhang": [
                    theoretical_item_ability_corr(spec[0], spec[1]) for spec in item_specs
                ],
                "Theoretisches Maximum": [
                    theoretical_item_ability_corr(spec[0], 0.999) for spec in item_specs
                ],
            })

            metric_row([
                ("Items", f"{n_items}"),
                ("Mittlere Schwierigkeit p", f"{np.nanmean(observed_p):.2f}"),
                ("Mittlere Trennschärfe", f"{np.nanmean(observed_rit):.2f}"),
                ("SD des Gesamttests", f"{np.std(total, ddof=1):.2f}"),
                ("Cronbach α", f"{alpha:.2f}" if np.isfinite(alpha) else "–")
            ])

            centered = total - np.mean(total)
            sd_total = np.std(total, ddof=1)
            if sd_total > 0:
                skew = np.mean((centered/sd_total)**3)
                excess = np.mean((centered/sd_total)**4) - 3
            else:
                skew, excess = np.nan, np.nan

            st.caption(
                f"Form der Testwertverteilung: Schiefe = {skew:.2f}, Exzess = {excess:.2f}. "
                "Werte nahe 0 sprechen für eine ungefähr symmetrische, glockenförmige Verteilung."
            )

            left, right = st.columns([1.25, 1], gap="large")
            with left:
                fig = px.histogram(x=total, nbins=max(10, n_items+1))
                score_range = [0, max(n_items, int(np.max(total)))] if st.session_state.get("zero_based_axes", True) else padded_range(total, min_pad=1, frac=.08)
                fig.update_layout(
                    height=430,
                    xaxis=dict(title="Gesamttestwert", range=score_range, fixedrange=True),
                    yaxis=dict(title="Häufigkeit", fixedrange=True)
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with right:
                st.markdown("#### Zusammensetzung und beobachtete Werte")
                grouped = (
                    item_summary
                    .groupby(["Schwierigkeit", "Trennschärfe-Kategorie"], observed=True)
                    .agg(
                        Anzahl=("p beobachtet", "size"),
                        p=("p beobachtet", "mean"),
                        Trennschärfe=("Trennschärfe beobachtet", "mean"),
                        Erwarteter_Zusammenhang=("Erwarteter Zusammenhang", "mean"),
                        Theoretisches_Maximum=("Theoretisches Maximum", "mean"),
                    )
                    .reset_index()
                )
                for col in ["p", "Trennschärfe", "Erwarteter_Zusammenhang", "Theoretisches_Maximum"]:
                    grouped[col] = grouped[col].round(2)
                st.dataframe(
                    grouped[["Schwierigkeit", "Trennschärfe-Kategorie", "Anzahl", "p", "Trennschärfe"]],
                    use_container_width=True, hide_index=True
                )
                st.caption(
                    "Vergleichen Sie Schwierigkeit und Trennschärfe der Items. Die Schwierigkeit beschreibt, "
                    "wie häufig ein Item gelöst wird; die Trennschärfe beschreibt seinen Zusammenhang mit dem Testwert."
                )
                with st.expander("Vertiefung: technische Simulationswerte"):
                    tech_c = grouped[["Schwierigkeit", "Trennschärfe-Kategorie", "Erwarteter_Zusammenhang", "Theoretisches_Maximum"]].rename(
                        columns={
                            "Erwarteter_Zusammenhang": "Erwarteter Zusammenhang mit Fähigkeit",
                            "Theoretisches_Maximum": "Theoretisches Maximum"
                        }
                    )
                    st.dataframe(tech_c, use_container_width=True, hide_index=True)

            low_disc_count = sum(
                int(matrix[(row_name, "niedrig")]) for row_name, _ in rows
            )
            easy_count = sum(
                int(matrix[("leicht", disc_name)]) for disc_name, _ in discs
            )
            hard_count = sum(
                int(matrix[("schwer", disc_name)]) for disc_name, _ in discs
            )

            if low_disc_count > n_items/3:
                st.warning(
                    "Viele Items haben niedrige Trennschärfe. Der Test kann dadurch weniger klar zwischen "
                    "insgesamt stärkeren und schwächeren Personen unterscheiden."
                )
            if easy_count > n_items*0.6:
                st.warning(
                    "Der Test enthält überwiegend leichte Items. Hohe Testwerte und Deckeneffekte werden wahrscheinlicher."
                )
            elif hard_count > n_items*0.6:
                st.warning(
                    "Der Test enthält überwiegend schwere Items. Niedrige Testwerte und Bodeneffekte werden wahrscheinlicher."
                )
            elif low_disc_count <= n_items/3:
                st.success(
                    "Die Zusammensetzung enthält eine breitere Mischung von Schwierigkeiten und überwiegend mindestens mittlere Trennschärfen."
                )

            st.markdown("#### Challenge")
            st.write(
                "1. Bauen Sie einen Test mit starkem Deckeneffekt. "
                "2. Bauen Sie einen Test mit starkem Bodeneffekt. "
                "3. Bauen Sie einen Test aus leichten und schweren Items und vergleichen Sie ihn mit einem Test aus mittleren Items. "
                "Vergleichen Sie jeweils Testwertverteilung, Trennschärfen und Reliabilität. "
                "Diskutieren Sie: In welchen Leistungsbereichen soll der Test Unterschiede erfassen? "
                "Die Gesamtreliabilität allein zeigt nicht, wie präzise er in einzelnen Bereichen misst."
            )

        st.info(
            "**Merksatz:** Die Schwierigkeit allein bestimmt nicht die Qualität eines Items. "
            "Entscheidend sind auch seine Trennschärfe, sein Inhalt und der Testzweck. "
            "Ein Test soll besonders in den Leistungsbereichen gut unterscheiden, die für seine Verwendung wichtig sind."
        )

        st.caption(
            "Die Simulation hält die gewünschte Lösungswahrscheinlichkeit p jetzt auch bei unterschiedlicher "
            "Trennschärfe stabil. Dadurch werden leichte und schwere Items bei hoher Differenzierung "
            "nicht mehr künstlich in Richtung p=.50 gezogen."
        )

        if st.session_state.show_formula:
            st.markdown("**Schwierigkeit, Itemvarianz und Trennschärfe**")
            st.latex(r"Var(X_i)=p_i(1-p_i)")
            st.latex(r"r_{i,Y}=\frac{Cov(X_i,Y)}{SD(X_i)\,SD(Y)}")
            st.caption(
                "Bei dichotomen Items ist die Itemvarianz durch p bestimmt (hier als Anteil zwischen 0 und 1). "
                "Die Trennschärfe lässt sich daraus allein nicht berechnen: Sie hängt auch vom Zusammenhang mit dem Testwert Y ab. "
                "Bei der Part-Whole-Korrektur wird Y durch den Testwert ohne das betreffende Item ersetzt."
            )

        exam_questions("m6c", [
            (
                "Warum sollten die Itemschwierigkeiten zum Ziel eines Tests passen?",
                "Ein Test soll besonders dort gut zwischen Personen unterscheiden, wo Unterschiede für seine Verwendung wichtig sind. Interessieren vor allem der untere und obere Leistungsbereich, kann eine Kombination leichter und schwerer Items sinnvoll sein. Ob der Test dort präzise misst, muss zusätzlich geprüft werden."
            ),
            (
                "Kann man allein aus der Schwierigkeit eines Items auf seine Trennschärfe schließen?",
                "Nein. Die Schwierigkeit gibt an, wie häufig ein Item gelöst wird. Die Trennschärfe beschreibt, wie die Antworten auf das Item mit dem Testwert zusammenhängen. Aus der Lösungshäufigkeit allein geht nicht hervor, welche Personen das Item lösen."
            ),
            (
                "Reicht eine hohe Reliabilität allein aus, um einen guten Test zu kennzeichnen?",
                "Nein. Der Test muss auch das beabsichtigte Merkmal erfassen und für seinen Zweck geeignet sein. Eine hohe Gesamtreliabilität garantiert zudem nicht, dass er in jedem Leistungsbereich gleich präzise misst."
            ),
        ])

# -----------------------------
def alpha_from_covariance(matrix):
    matrix = np.asarray(matrix, dtype=float)
    n = len(matrix)
    total = float(matrix.sum())
    return n / (n-1) * (1-float(np.trace(matrix))/total) if total > 1e-12 else np.nan


def reliability_heatmap(matrix, title):
    names = [f"Item {i+1}" for i in range(len(matrix))]
    fig = go.Figure(go.Heatmap(z=matrix, x=names, y=names, zmin=-1, zmax=1,
        colorscale="RdBu", reversescale=True, text=np.round(matrix, 2), texttemplate="%{text:.2f}"))
    fig.update_layout(title=title, height=390, margin=dict(t=45, b=25))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def reliability_length():
    learning_question("Wie viel Testlänge brauchen wir, um die Reliabilität zu erhöhen – und wann lohnt sich eine weitere Verlängerung kaum noch?")
    initial = st.slider("Reliabilität des Ausgangstests Yₐ", 0.05, 0.95, 0.60, 0.05, key="rel_sb_initial")
    length = st.slider("Items im Ausgangstest Yₐ", 4, 40, 10, key="rel_sb_length")
    new_length = st.slider("Items im veränderten Test Y", 2, 200, 20, key="rel_sb_newlength")
    target = st.slider("Gewünschte Reliabilität (Planungsziel)", 0.50, 0.99, 0.80, 0.01, key="rel_sb_target")
    k = new_length / length
    predicted = k * initial / (1+(k-1)*initial)
    needed_k = target*(1-initial)/(initial*(1-target))
    needed_items = max(1, int(np.ceil(length*needed_k-1e-10)))
    metric_row([("Längenfaktor k", f"{k:.2f}"), ("Rel(Y)", f"{predicted:.3f}"),
                ("Items für das Planungsziel", str(needed_items))])
    grid = np.arange(1, max(200, min(needed_items+10, 500))+1)
    factors = grid / length
    curve = factors*initial/(1+(factors-1)*initial)
    fig = go.Figure(go.Scatter(x=grid, y=curve, mode="lines", name="Spearman–Brown"))
    fig.add_trace(go.Scatter(x=[length, new_length], y=[initial, predicted], mode="markers+text",
        text=["Ausgangstest Yₐ", "Veränderter Test Y"], textposition="bottom right", marker=dict(size=13), name="Ihre Tests"))
    fig.add_hline(y=target, line_dash="dot", annotation_text="Planungsziel")
    fig.update_layout(height=420, xaxis=dict(title="Anzahl vergleichbarer Items", range=[0, max(45, min(needed_items+5, len(grid)), new_length+5)], fixedrange=True),
        yaxis=dict(title="Reliabilität", range=[0,1.02], fixedrange=True), dragmode=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.latex(r"Rel(Y)=\frac{k\,Rel(Y_a)}{1+(k-1)Rel(Y_a)},\qquad k=\frac{\text{neue Testlänge}}{\text{ursprüngliche Testlänge}}")
    st.latex(r"k=2:\quad Rel(Y)=\frac{2\,Rel(Y_a)}{1+Rel(Y_a)}=\frac{2\,Corr(Y_a,Y_{a^{\prime}})}{1+Corr(Y_a,Y_{a^{\prime}})}")
    st.caption("Y: veränderter Gesamttest · Yₐ: Ausgangstest bzw. erste Hälfte · Yₐ′: parallele zweite Hälfte · k: neue Itemzahl geteilt durch ursprüngliche Itemzahl.")
    st.write("**Ausprobieren:** Verdoppeln und halbieren Sie die Itemzahl. Setzen Sie dann das Planungsziel auf .90 und .95. Beobachten Sie den zusätzlichen Aufwand für immer kleinere Verbesserungen.")
    st.info("Die Prognose setzt eine vergleichbare Qualität der hinzugefügten Items und passende Annahmen über die Fehler voraus; parallel aufgebaute Testteile sind der klassische Fall. Zusätzliche schlechte oder redundante Items erfüllen diese Bedingung nicht automatisch. Das Planungsziel ist frei wählbar und keine allgemeine Gütegrenze.")
    with st.expander("Split-Half: Von der Korrelation zweier Hälften zum ganzen Test"):
        half_r = st.slider("Korrelation zweier gleich langer Testhälften", 0.0, 0.95, 0.50, 0.05, key="rel_half_r")
        st.metric("Split-Half-Schätzung für den ganzen Test", f"{2*half_r/(1+half_r):.3f}")
        st.latex(r"\widehat{Rel}(Y)=\frac{2\,Corr(Y_a,Y_{a^{\prime}})}{1+Corr(Y_a,Y_{a^{\prime}})}")
        st.caption("Y = Yₐ + Yₐ′. Yₐ ist die erste Hälfte, Yₐ′ die parallele zweite Hälfte. Die Interpretation setzt passende, insbesondere parallele Hälften voraus. Verschiedene Aufteilungen können verschiedene Schätzwerte ergeben. Die Halbtestkorrelation ist nicht bereits die Reliabilität des ganzen Tests.")


def reliability_covariances():
    learning_question("Wie verändern die Kovarianzen die Testwertvarianz und Cronbachs Alpha?")
    st.write("Der Test besteht aus **5 Items mit jeweils Varianz 4**. Die Summe der Einzelvarianzen bleibt deshalb immer **20**. Verändern Sie nur, wie stark die Items gemeinsam variieren.")
    covariance_value = st.slider("Kovarianz zwischen je zwei Items", 0.0, 4.0, 1.2, 0.2, key="rel_alpha_cov")
    matrix = np.full((5, 5), covariance_value)
    np.fill_diagonal(matrix, 4.0)
    item_variances = float(np.trace(matrix))
    covariance_contribution = float(matrix.sum()-item_variances)
    total_variance = float(matrix.sum())
    alpha = alpha_from_covariance(matrix)
    metric_row([("Summe der Einzelvarianzen", f"{item_variances:.1f}"),
                ("Testwertvarianz", f"{total_variance:.1f}"), ("Cronbachs α", f"{alpha:.3f}")])
    fig = go.Figure()
    categories = ["Summe der Einzelvarianzen", "Varianz des Gesamttestwerts"]
    fig.add_trace(go.Bar(x=categories, y=[item_variances,item_variances], name="Einzelvarianzen", marker_color="#2878b5",
        text=[f"{item_variances:.0f}",f"{item_variances:.0f}"],textposition="inside"))
    fig.add_trace(go.Bar(x=categories, y=[0,covariance_contribution], name="Beitrag der Kovarianzen", marker_color="#d97706",
        text=["",f"{covariance_contribution:.1f}" if covariance_contribution else ""],textposition="inside"))
    fig.update_layout(barmode="stack", height=380, yaxis=dict(title="Varianz",range=[0,105],fixedrange=True),
        xaxis=dict(fixedrange=True),legend=dict(orientation="h",y=1.15),dragmode=False)
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    st.latex(r"Var(Y)=\sum_{i=1}^{5}Var(Y_i)+2\sum_{i<j}Cov(Y_i,Y_j)")
    st.write(f"**Testwertvarianz = {item_variances:.1f} + {covariance_contribution:.1f} = {total_variance:.1f}.** Der orange Anteil berücksichtigt jede Kovarianz zweimal.")
    st.latex(r"\alpha=\frac{5}{4}\left(1-\frac{\sum_i Var(Y_i)}{Var(Y)}\right)")
    st.latex(r"\alpha=\frac{5}{4}\left(1-\frac{20}{"+f"{total_variance:.1f}"+r"}\right)="+f"{alpha:.3f}")
    if covariance_value == 0:
        st.success("Keine Kovarianzen: Die Testwertvarianz ist genau die Summe der Einzelvarianzen. Der Quotient ist 1 und Alpha ist 0.")
    else:
        st.info("Je höher die Kovarianzen bei gleichen Itemvarianzen, desto größer die Testwertvarianz im Verhältnis zur Summe der Einzelvarianzen. Der Quotient wird kleiner – Alpha wird größer.")
    st.write("**Ausprobieren:** Schieben Sie die Kovarianz zuerst auf 0 und erhöhen Sie sie dann schrittweise. Beobachten Sie den orangefarbenen Anteil, den Quotienten in der Formel und Alpha.")


def score_intervals(score, reliability, mean_score, sd_score, z):
    sem = sd_score*np.sqrt(1-reliability)
    see = sd_score*np.sqrt(reliability*(1-reliability))
    estimated_true = mean_score+reliability*(score-mean_score)
    return sem, see, estimated_true, (score-z*sem, score+z*sem), (estimated_true-z*see, estimated_true+z*see)


def reliability_intervals():
    from statistics import NormalDist
    learning_question("Was bedeutet die Reliabilität für die Unsicherheit eines einzelnen Testwerts – und warum unterscheiden sich Messfehler- und Schätzfehlerintervalle?")
    a, b, c = st.columns(3)
    score = a.number_input("Beobachteter Testwert Y", min_value=-200.0, max_value=300.0, value=115.0, step=1.0, key="rel_ci_score")
    mean_score = b.number_input("Mittelwert der Bezugsgruppe M", min_value=-100.0, max_value=200.0, value=100.0, step=1.0, key="rel_ci_mean")
    sd = c.number_input("Standardabweichung der Testwerte SD(Y)", min_value=1.0, max_value=100.0, value=15.0, step=1.0, key="rel_ci_sd")
    reliability = st.slider("Reliabilität für den ausgewählten Test", 0.0, 1.0, 0.80, 0.01, key="rel_ci_r")
    coverage = st.selectbox("Intervallniveau", [90, 95, 99], index=1, key="rel_ci_level")
    z = NormalDist().inv_cdf(0.5+coverage/200)
    sem, see, estimated_true, classic, regression = score_intervals(score,reliability,mean_score,sd,z)
    metric_row([("Standardmessfehler", f"{sem:.2f}"), ("Standardschätzfehler", f"{see:.2f}"), ("Regressionsbasierte Schätzung des wahren Werts", f"{estimated_true:.2f}")])
    st.dataframe(pd.DataFrame({"Verfahren": ["Standardmessfehler", "Standardschätzfehler (Regression)"],
        "Intervallzentrum": [score,estimated_true], "Untere Grenze": [classic[0],regression[0]], "Obere Grenze": [classic[1],regression[1]]}), hide_index=True)
    st.latex(r"SEM=SD(Y)\sqrt{1-Rel(Y)},\qquad I_{\mathrm{Mess}}=Y\pm z\cdot SEM")
    st.latex(r"\hat T=M+Rel(Y)(Y-M),\quad SEE=SD(Y)\sqrt{Rel(Y)(1-Rel(Y))},\quad I_{\mathrm{Schätz}}=\hat T\pm z\cdot SEE")
    st.caption(f"Zweiseitiges {coverage}%-Niveau: z = {z:.3f}. M und SD(Y) stammen aus derselben relevanten Bezugsgruppe wie die Reliabilität. Die Unsicherheit dieser drei Kennwerte selbst wird hier nicht zusätzlich berücksichtigt.")
    comparison = st.multiselect("Weitere Reliabilitäten zum Vergleich", [0.0,0.20,0.40,0.60,0.80,0.90,0.95,1.0], default=[0.60,0.90], key="rel_ci_compare")
    rs = sorted(set(comparison+[reliability]))
    show_cut = st.checkbox("Entscheidungsgrenze einblenden", key="rel_ci_showcut")
    cutoff = st.number_input("Entscheidungsgrenze", value=110.0, step=1.0, key="rel_ci_cutoff") if show_cut else None
    first, second = st.tabs(["Verschiedene Reliabilitäten", "Verschiedene Testwerte"])
    with first:
        fig = go.Figure()
        rows = [score_intervals(score,r,mean_score,sd,z) for r in rs]
        for label, color, centers, errors, offset in [
            ("Messfehlerintervall um X", "#2878b5", [score]*len(rs), [z*v[0] for v in rows],0.13),
            ("Schätzfehlerintervall um regressionsbasierte Schätzung", "#d97706", [v[2] for v in rows], [z*v[1] for v in rows],-0.13)]:
            fig.add_trace(go.Scatter(x=centers, y=np.arange(len(rs))+offset, mode="markers", name=label,
                marker=dict(size=11,color=color),error_x=dict(type="data",array=errors,color=color,thickness=3),
                hovertemplate="Intervallzentrum: %{x:.2f}<extra>"+label+"</extra>"))
        fig.add_vline(x=mean_score,line_dash="dot",annotation_text="Gruppenmittel M")
        fig.add_vline(x=score,line_dash="dash",annotation_text="Testwert Y",annotation_position="bottom right")
        if cutoff is not None:
            fig.add_vline(x=cutoff,line_color="#8b1e3f",line_dash="dashdot",annotation_text="Entscheidungsgrenze")
        bounds=[v for row in rows for pair in row[3:] for v in pair]+[score,mean_score]
        if cutoff is not None:
            bounds.append(cutoff)
        fig.update_layout(height=max(350,len(rs)*85), xaxis=dict(title="Testwertskala",range=[min(bounds)-sd*.3,max(bounds)+sd*.3],fixedrange=True),
            yaxis=dict(title="Reliabilität",tickvals=list(range(len(rs))),ticktext=[f"{r:.2f}"+(" ← ausgewählt" if r==reliability else "") for r in rs],range=[-.5,len(rs)-.5],fixedrange=True),
            legend=dict(orientation="h",y=1.2),dragmode=False)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        st.write("**Ausprobieren:** Verschieben Sie Y weit über oder unter den Mittelwert. Bei kleiner Reliabilität wandert die regressionsbasierte Schätzung stärker zum Gruppenmittel; das Messfehlerintervall bleibt um Y zentriert.")
    with second:
        grid = np.linspace(mean_score-3*sd,mean_score+3*sd,101)
        centers = mean_score+reliability*(grid-mean_score)
        fig=go.Figure()
        for center,width,color,fill,label in [(grid,z*sem,"#2878b5","rgba(40,120,181,.14)","Messfehlerintervall"),
                                             (centers,z*see,"#d97706","rgba(217,119,6,.18)","regressionsbasiertes Intervall")]:
            fig.add_trace(go.Scatter(x=grid,y=center-width,mode="lines",line=dict(width=0),showlegend=False,hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=grid,y=center+width,mode="lines",line=dict(width=0),fill="tonexty",fillcolor=fill,name=label+f" ({coverage}%)",hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=grid,y=center,mode="lines",line=dict(color=color),name=label+": Zentrum"))
        fig.add_trace(go.Scatter(x=[score,score],y=[score,estimated_true],mode="markers",marker=dict(size=11,color=["#2878b5","#d97706"]),name="Aktueller Testwert"))
        if cutoff is not None:
            fig.add_hline(y=cutoff,line_dash="dashdot",line_color="#8b1e3f",annotation_text="Entscheidungsgrenze")
        fig.update_layout(height=450,xaxis=dict(title="Beobachteter Testwert Y",range=[min(grid[0],score-sd*.1),max(grid[-1],score+sd*.1)],fixedrange=True),
            yaxis=dict(title="Schätzung und Intervall für den wahren Wert",fixedrange=True),legend=dict(orientation="h",y=1.28),margin=dict(t=100),dragmode=False)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        st.caption("Bei gleicher Reliabilität und Standardabweichung bleiben die Intervallbreiten über Y konstant: Das Beispiel verwendet konstante Fehlervarianzen. Testwertabhängige Messfehler, etwa aus IRT-Modellen, werden hier nicht modelliert.")
    with st.expander("Warum wird der Standardschätzfehler empfohlen? Regression zur Mitte", expanded=True):
        st.write("Ein extremer beobachteter Testwert kann aus einer extremen Merkmalsausprägung und einem zufälligen Messfehler entstehen. Unter dem hier angenommenen Modell haben Personen mit sehr hohen beobachteten Werten im Mittel auch positive Messfehler; bei sehr niedrigen Werten ist es umgekehrt. Deshalb ist der erwartete wahre Wert weniger extrem als der beobachtete Wert.")
        st.write(f"**Ihr Beispiel:** Y = {score:.1f}, M = {mean_score:.1f}, Rel(Y) = {reliability:.2f}. "
                 f"Von der Abweichung Y − M = {score-mean_score:+.1f} bleiben in der Schätzung "
                 f"{reliability:.2f} × ({score-mean_score:+.1f}) = {estimated_true-mean_score:+.1f}. "
                 f"Der geschätzte wahre Wert ist **{estimated_true:.1f}**.")
        st.write("Bei einer unabhängigen parallelen Wiederholungsmessung wäre unter diesem Modell derselbe weniger extreme Wert im Mittel zu erwarten. Das ist Regression zur Mitte: Es muss keine echte Veränderung der Fähigkeit stattgefunden haben. Der wahre Wert selbst bewegt sich dabei nicht zum Mittelwert; unsere Schätzung berücksichtigt die zusätzliche Information aus der Bezugsgruppe.")
        mse_classic = sd**2*(1-reliability)
        mse_regression = sd**2*reliability*(1-reliability)
        metric_row([("Mittlerer quadratischer Fehler: Y als Schätzung", f"{mse_classic:.2f}"),
                    ("Mittlerer quadratischer Fehler: Regression", f"{mse_regression:.2f}")])
        st.caption("Diese Fehler beziehen sich auf die gesamte angenommene Population, nicht auf die einzelne ausgewählte Person. Einheit: Testwertpunkte². Die Bezugsgruppenparameter werden als bekannt behandelt.")
        st.write("**Der Grund für die Empfehlung:** Wenn das Modell und die Bezugsgruppe passen, vermindert die regressionsbasierte Schätzung den durchschnittlichen quadratischen Schätzfehler. Der Standardschätzfehler beschreibt die verbleibende Streuung der wahren Werte um diese Schätzung. Er gehört deshalb zum regressionsbasierten Intervall; er darf nicht einfach als kleinerer Fehler um den unveränderten Testwert gelegt werden.")
        st.write("**Die Abwägung:** Die Schätzung nutzt Gruppeninformation und ist für eine Person mit festem, dauerhaft extremem wahrem Wert im Allgemeinen verzerrt zur Mitte. Eine ungeeignete Bezugsgruppe kann ebenfalls irreführen. Für Entscheidungen zählt daher, welche Fehler im konkreten Fall besonders folgenreich sind; das schmalere Intervall allein entscheidet nicht über das bessere Verfahren.")
        st.write("**Ausprobieren:** Wählen Sie einen Testwert zwei Standardabweichungen über dem Mittelwert. Vergleichen Sie Reliabilität .50 und .95 und betrachten Sie die Ansicht ‚Verschiedene Testwerte‘. Setzen Sie Y anschließend genau auf M: Die Intervallzentren stimmen dann überein, die Breiten im Allgemeinen nicht.")
    st.info("Zwei unterschiedliche Aussagen: Das klassische Intervall Y ± z·SEM ist bei normalverteilten, konstanten Messfehlern ein Konfidenzintervall für einen festen wahren Wert. Das regressionsbasierte Intervall beschreibt unter einem passenden normalverteilten Populationsmodell die bedingte Verteilung wahrer Werte bei gegebenem Y. Es hat nicht automatisch dieselbe Überdeckung für jede einzelne Person. Deshalb bedeutet ein schmaleres regressionsbasiertes Intervall nicht allgemein eine überlegene Diagnose.")
    if reliability < .10:
        st.warning("Bei Reliabilität nahe 0 wird die regressionsbasierte Schätzung fast vollständig vom Gruppenmittel bestimmt. Bei exakt 0 enthält das angenommene Modell keine Unterschiede in den wahren Werten; das regressionsbasierte Intervall schrumpft deshalb auf M. Das ist eine Konsequenz dieses Modells, kein Beleg für präzise individuelle Messung.")
    if show_cut:
        st.write("Die Grenze dient dem Vergleich: Reicht ein Intervall über die Grenze, ist eine harte Zuordnung besonders unsicher. Auch ein vollständig auf einer Seite liegendes Intervall garantiert keine fehlerfreie Entscheidung. Welche zusätzlichen Informationen nötig sind, hängt von den Folgen falsch positiver und falsch negativer Entscheidungen ab.")
    st.markdown("Methodischer Hintergrund: [Bias and precision in true-score estimation](https://doi.org/10.1111/bmsp.70058).")


def module_reliability():
    version_badge()
    st.title("6 · Reliabilitätsschätzung")
    st.caption("Vom Verlängern eines Tests über Itemzusammenhänge bis zur Unsicherheit einzelner Testwerte. Wir betrachten jeweils die Reliabilität eines ungewichteten Summenscores.")
    section = st.radio("Lernschritt", ["6A · Spearman–Brown", "6B · Kovarianzen und Alpha", "6C · Testwerte und Intervalle"], horizontal=True, key="rel_section")
    if section.startswith("6A"):
        reliability_length()
    elif section.startswith("6B"):
        reliability_covariances()
    else:
        reliability_intervals()
    with st.expander("Einordnen und weiterlesen"):
        st.write("Interne Konsistenz ist keine Prüfung der zeitlichen Stabilität. Für Retest- und Paralleltestschätzungen braucht man entsprechende zusätzliche Messungen. Reliabilität hängt auch von der untersuchten Personengruppe ab. Ein hoher Wert allein beantwortet weder die Validitätsfrage noch die Frage nach den Folgen von Fehlentscheidungen.")


def validity_population(paths, residual_link):
    """Standardized Gaussian star model, with two within-time residual pairs."""
    b = np.asarray(paths, dtype=float)
    residual = np.diag(1 - b*b)
    for i, j in [(0, 1), (2, 3)]:
        residual[i, j] = residual[j, i] = residual_link * np.sqrt(residual[i, i]*residual[j, j])
    sigma = np.eye(5)
    sigma[0, 1:] = sigma[1:, 0] = b
    sigma[1:, 1:] = np.outer(b, b) + residual
    return sigma


def validity_fit(data, paired=False):
    """Exact normal-theory ML for common-predictor regressions with block residuals.

    All four slopes free; residual covariance diagonal or two 2x2 blocks.
    ML covariance uses divisor N; likelihood-ratio statistic is N * F_ML.
    """
    from scipy.stats import chi2
    s = np.cov(data, rowvar=False, bias=True)
    n = len(data)
    b = s[1:, 0] / s[0, 0]
    residual = s[1:, 1:] - np.outer(b, b)*s[0, 0]
    psi = np.diag(np.diag(residual))
    if paired:
        for i, j in [(0, 1), (2, 3)]:
            psi[i, j] = psi[j, i] = residual[i, j]
    sigma = s.copy()
    sigma[1:, 1:] = np.outer(b, b)*s[0, 0] + psi
    def discrepancy(model):
        return max(0.0, np.linalg.slogdet(model)[1] + np.trace(np.linalg.solve(model, s))
                   - np.linalg.slogdet(s)[1] - 5)
    stat = n * discrepancy(sigma)
    baseline = n * discrepancy(np.diag(np.diag(s)))
    df = 4 if paired else 6  # 15 moments minus 11 or 9 free parameters
    excess = max(stat - df, 0.0)
    base_excess = baseline - 10
    cfi = 1 - excess / max(excess, base_excess) if base_excess > 1e-10 else np.nan
    tli = (baseline/10 - stat/df)/(baseline/10 - 1) if base_excess > 1e-10 else np.nan
    return dict(sigma=sigma, beta=b*np.sqrt(s[0, 0]/np.diag(s)[1:]),
                chi2=stat, df=df, p=chi2.sf(stat, df),
                RMSEA=np.sqrt(excess/(df*(n-1))), CFI=cfi, TLI=tli)


def module_validity():
    version_badge()
    st.title("7 · Validität")
    st.write("Passt das Muster der Zusammenhänge zu dem, was unser Test erfassen soll? Wir vergleichen einen Ausgangstest mit vier Kriterien: zwei gleichzeitig und zwei später erhobenen Variablen.")
    st.info("Hier bedeutet positiv bzw. negativ die erwartete Richtung des Zusammenhangs – nicht gute bzw. schlechte Validität. Auch ein negativer Zusammenhang kann die Interpretation eines Tests stützen. Für ein theoretisch unabhängiges Merkmal kann ein Zusammenhang nahe null erwartet werden.")
    names = ["Ausgangstest", "Simultan +", "Simultan −", "Prospektiv +", "Prospektiv −"]
    st.markdown("#### 1 · Zusammenhänge simulieren")
    st.caption("Beispiel: Ein Kompetenztest hängt positiv mit Leistung und negativ mit Schwierigkeiten zusammen, jeweils gleichzeitig und später erhoben. Die Regler bestimmen die Zusammenhänge in der simulierten Population.")
    cols = st.columns(2)
    paths = []
    for i, default in enumerate([0.6, -0.5, 0.4, -0.3]):
        with cols[i//2]:
            paths.append(st.slider(f"Ausgangstest → {names[i+1]}", -0.9, 0.9, default, 0.05, key=f"val_path_{i}"))
    n = st.select_slider("Anzahl Personen", options=[100, 200, 500, 1000, 2000], value=500, key="val_n")
    with st.expander("Zusätzliche Gemeinsamkeiten der Kriterien"):
        residual_link = st.slider("Zusammenhang der Restanteile innerhalb desselben Erhebungszeitpunkts", -0.8, 0.8, 0.0, 0.1, key="val_residual")
        st.write("Diese Gemeinsamkeiten bestehen zusätzlich zum Zusammenhang mit dem Ausgangstest, etwa durch eine gemeinsame Erhebungsmethode. Derselbe Wert gilt für das simultane und das prospektive Kriterienpaar.")
    if "val_seed" not in st.session_state:
        st.session_state.val_seed = 42
    if st.button("Neue Stichprobe ziehen", key="val_resample"):
        st.session_state.val_seed += 1
    population = validity_population(paths, residual_link)
    noise = np.random.default_rng(st.session_state.val_seed).normal(size=(n, 5))
    data = noise @ np.linalg.cholesky(population).T
    observed = np.corrcoef(data, rowvar=False)
    st.markdown("#### 2 · Korrelationsmatrix")
    fig = go.Figure(go.Heatmap(z=observed, x=names, y=names, zmin=-1, zmax=1,
                              colorscale="RdBu", reversescale=True, text=np.round(observed, 2),
                              texttemplate="%{text:.2f}", colorbar=dict(title="r")))
    fig.update_layout(height=400, margin=dict(t=20, b=20), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Korrelationen der gezogenen Stichprobe. Bei unveränderten Einstellungen bleibt die Stichprobe gleich; die Regler lösen keine neue Zufallsziehung aus.")
    st.markdown("#### 3 · Pfadmodell schätzen")
    paired = st.checkbox("Zusätzliche Zusammenhänge innerhalb beider Kriterienpaare im Modell zulassen", key="val_paired")
    result = validity_fit(data, paired)
    # Native Plotly diagram: four directed paths, freely estimated standardized coefficients.
    diagram = go.Figure()
    positions = [(1, 3), (1, 2), (1, 1), (1, 0)]
    diagram.add_trace(go.Scatter(x=[0]+[1]*4, y=[1.5, 3, 2, 1, 0], mode="markers+text",
                                marker=dict(size=24, color=["#2574a9"]+["#e79b45"]*4),
                                text=names, textposition="top center", hoverinfo="skip"))
    for i, (x, y) in enumerate(positions):
        diagram.add_annotation(x=x-0.04, y=y, ax=0.04, ay=1.5, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#78909c")
        diagram.add_annotation(x=0.6, y=1.5+0.6*(y-1.5), text=f"β = {result['beta'][i]:.2f}", showarrow=False, bgcolor="white")
    if paired:
        for top, bottom in [(3, 2), (1, 0)]:
            diagram.add_annotation(x=1.15, y=top, ax=1.15, ay=bottom, xref="x", yref="y", axref="x", ayref="y",
                                   showarrow=True, arrowside="end+start", arrowhead=2, startarrowhead=2, arrowcolor="#bd7d31")
    diagram.update_layout(height=380, showlegend=False, margin=dict(t=30,b=10),
                          xaxis=dict(visible=False, range=[-0.35, 1.5]), yaxis=dict(visible=False, range=[-0.3, 3.5]))
    st.plotly_chart(diagram, use_container_width=True)
    st.caption("Pfeile: geschätzte standardisierte Regressionskoeffizienten. Doppelpfeile, falls aktiviert: erlaubte Zusammenhänge der Restanteile. Alle vier Regressionspfade werden frei geschätzt. Die Pfeile sind keine Nachweise kausaler Wirkungen; simultane Messungen liefern auch keine zeitliche Reihenfolge.")
    metrics = st.columns(4)
    metrics[0].metric(f"χ² (df = {result['df']})", f"{result['chi2']:.2f}")
    for col, key in zip(metrics[1:], ["RMSEA", "CFI", "TLI"]):
        col.metric(key, f"{result[key]:.3f}" if np.isfinite(result[key]) else "nicht bestimmbar")
    st.caption(f"χ²-Test: p = {result['p']:.4f}. Freiheitsgrade: 15 Varianzen/Kovarianzen minus {15-result['df']} freie Parameter.")
    st.write("χ² prüft die exakte Modellpassung und reagiert auch auf die Stichprobengröße. Ein kleinerer RMSEA spricht für weniger Fehlanpassung pro Freiheitsgrad. CFI und TLI vergleichen das Modell mit einem Modell ohne Zusammenhänge; Werte nahe 1 sprechen für eine bessere relative Passung. Bei wenigen Freiheitsgraden können die Indizes stark schwanken. Es gibt hier keine automatische Bewertung anhand starrer Grenzwerte.")
    if not np.isfinite(result['CFI']):
        st.warning("Das Vergleichsmodell ohne Zusammenhänge passt bereits sehr gut. CFI und TLI lassen sich dann nicht sinnvoll interpretieren.")
    with st.expander("Was prüft der Modellfit hier genau?"):
        st.write("Das Grundmodell schätzt vier Pfade, die Varianz des Ausgangstests und vier Restvarianzen (9 Parameter, df = 6). Es setzt alle sechs Zusammenhänge zwischen den Restanteilen auf null. Die Erweiterung schätzt zusätzlich zwei Restkovarianzen (11 Parameter, df = 4). Ein Modell mit allen sechs freien Restkovarianzen wäre gesättigt (df = 0): Sein perfekter Fit wäre keine Prüfung.")
        st.write("Da jeder Zielwert denselben einzigen Prädiktor hat, sind die standardisierten Pfade hier gleich den jeweiligen Korrelationen mit dem Ausgangstest. Der Fit prüft vor allem die Einschränkungen der Restkovarianzen. Er prüft nicht, ob die vier Pfade theoretisch die richtige Richtung oder ausreichende Stärke haben.")
        fitted = result['sigma'] / np.sqrt(np.outer(np.diag(result['sigma']), np.diag(result['sigma'])))
        st.write("Vom Modell vorhergesagte Korrelationsmatrix")
        st.dataframe(pd.DataFrame(fitted, index=names, columns=names).round(3), use_container_width=True)
        st.write("Abweichungen: Stichprobe minus Modell")
        st.dataframe(pd.DataFrame(observed-fitted, index=names, columns=names).round(3), use_container_width=True)
        st.caption("Normalverteilte kontinuierliche Variablen, vollständige Daten, normaltheoretische Maximum-Likelihood-Schätzung. Dieses einfache Modell hat eine direkte Lösung ohne iterative Optimierung. χ² = N · FML bei einer Kovarianzmatrix mit Divisor N; RMSEA verwendet N−1. CFI wird auf 0–1 begrenzt, TLI nicht. Grundlage der Fitindizes: https://semopy.com/docs/stats.html")
    st.info("Validität betrifft die begründete Interpretation und Verwendung der Testwerte. Ein passendes Zusammenhangsmuster liefert dafür Evidenz. Ein guter Modellfit allein beweist weder Validität noch Kausalität. Für Vorhersagen bei neuen Personen ist zusätzlich eine Prüfung an unabhängigen Daten wichtig.")
    st.markdown("#### Ausprobieren")
    st.write("1. Schwächen Sie die beiden prospektiven Zusammenhänge ab: Was ändert sich an der Vorhersagekraft, was am Modellfit?\n\n2. Erhöhen Sie die zusätzlichen Gemeinsamkeiten auf 0,6. Vergleichen Sie das Grundmodell mit der Erweiterung.\n\n3. Verändern Sie die Stichprobengröße und ziehen Sie neue Stichproben: Wie stabil sind Pfade und Fitwerte?")
    exam_questions("validity", [
        ("Bedeutet eine negative Korrelation, dass der Test nicht valide ist?", "Nein. Wenn die Theorie einen negativen Zusammenhang vorhersagt, kann dieser die Interpretation der Testwerte stützen."),
        ("Belegt ein guter Modellfit die Validität des Ausgangstests?", "Nein. Er zeigt, dass die Einschränkungen des Modells mit den Daten vereinbar sind. Zusätzlich müssen Richtung und Stärke der Zusammenhänge zur Theorie und zum Verwendungszweck passen."),
        ("Was unterscheidet simultane und prospektive Validierung?", "Bei simultaner Validierung werden Test und Kriterium gleichzeitig erhoben. Bei prospektiver Validierung wird das Kriterium später erhoben; damit lässt sich eine Vorhersage über die Zeit untersuchen.")
    ])


# Router
# -----------------------------
module = st.session_state.module
if module.startswith("1"):
    module1()
elif module.startswith("2"):
    module2()
elif module.startswith("3"):
    module4()
elif module.startswith("4"):
    module5()
elif module.startswith("5"):
    module6()
elif module.startswith("6"):
    module_reliability()
elif module.startswith("7"):
    module_validity()

st.divider()
st.caption(
    f"FLAPs · Version {APP_VERSION} · Forschungs- und Lern-App für angewandte Psychometrie. "
    "Alle Personen und Testwerte sind fiktiv."
)
