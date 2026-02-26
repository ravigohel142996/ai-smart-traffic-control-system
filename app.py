"""
AI Smart Traffic Control and Accident Prediction System
Production-level Streamlit Dashboard
"""

import time
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Smart Traffic Control System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS – dark professional theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        background-color: #0A0A0F;
        color: #E0E0E0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0D0D16;
        border-right: 1px solid #1E1E2E;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #12121A;
        border: 1px solid #1E1E2E;
        border-radius: 8px;
        padding: 16px 20px;
    }

    /* Plotly chart backgrounds */
    .js-plotly-plot .plotly {
        background-color: #12121A !important;
    }

    /* Section headings */
    .section-header {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #00D4FF;
        margin-bottom: 4px;
        border-bottom: 1px solid #1E1E2E;
        padding-bottom: 6px;
    }

    /* Status badges */
    .badge-safe {
        display: inline-block;
        background-color: #0A2A1A;
        color: #00C853;
        border: 1px solid #00C853;
        border-radius: 4px;
        padding: 4px 12px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1.5px;
    }
    .badge-warning {
        display: inline-block;
        background-color: #2A1F00;
        color: #FFB300;
        border: 1px solid #FFB300;
        border-radius: 4px;
        padding: 4px 12px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1.5px;
    }
    .badge-critical {
        display: inline-block;
        background-color: #2A0A0A;
        color: #FF1744;
        border: 1px solid #FF1744;
        border-radius: 4px;
        padding: 4px 12px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1.5px;
    }

    /* Dividers */
    hr {
        border: none;
        border-top: 1px solid #1E1E2E;
        margin: 12px 0;
    }

    /* Streamlit default overrides */
    .stMetric label { color: #8888AA; font-size: 11px; letter-spacing: 1px; text-transform: uppercase; }
    .stMetric [data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; color: #E0E0E0; }
    .stMetric [data-testid="stMetricDelta"] { font-size: 12px; }
    .stSelectbox label, .stSlider label { color: #8888AA; font-size: 12px; }
    div[data-baseweb="select"] { background-color: #12121A; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
WEATHER_CONDITIONS = ["clear", "rain", "fog", "snow"]
SIGNAL_STATUSES = ["green", "yellow", "red"]
INTERSECTIONS = [
    "Junction Alpha", "Junction Beta", "Junction Gamma",
    "Junction Delta", "Junction Epsilon",
]
HISTORY_WINDOW = 60   # data-points kept for time-series charts
RISK_SAFE_THRESHOLD = 0.35
RISK_WARNING_THRESHOLD = 0.65

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "history_vehicle_count": [],
        "history_density": [],
        "history_risk": [],
        "history_timestamps": [],
        "last_live_data": None,
        "model": None,
        "model_r2": None,
        "model_mse": None,
        "feature_importances": None,
        "le_weather": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dataset & model training
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training accident prediction model …")
def train_model():
    rng = np.random.default_rng(42)
    n = 5000

    weather_raw = rng.choice(WEATHER_CONDITIONS, size=n)
    le = LabelEncoder()
    weather_enc = le.fit_transform(weather_raw)

    vehicle_count  = rng.integers(5, 200, size=n).astype(float)
    density        = rng.uniform(0.1, 1.0, size=n)
    avg_speed      = rng.uniform(10, 120, size=n)
    time_of_day    = rng.uniform(0, 24, size=n)
    visibility     = rng.uniform(10, 1000, size=n)

    # Risk formula (deterministic signal + noise)
    risk = (
        0.30 * (vehicle_count / 200)
        + 0.25 * density
        + 0.20 * (1 - avg_speed / 120)
        + 0.15 * (weather_enc / (len(le.classes_) - 1))
        + 0.05 * np.abs(time_of_day - 12) / 12
        + 0.05 * (1 - visibility / 1000)
        + rng.normal(0, 0.03, size=n)
    )
    risk = np.clip(risk, 0.0, 1.0)

    X = pd.DataFrame({
        "vehicle_count":  vehicle_count,
        "traffic_density": density,
        "average_speed":  avg_speed,
        "weather_condition": weather_enc,
        "time_of_day":    time_of_day,
        "visibility":     visibility,
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X, risk, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    return model, r2, mse, importances, le


# ─────────────────────────────────────────────────────────────────────────────
# Data simulation helpers
# ─────────────────────────────────────────────────────────────────────────────
def simulate_live_data() -> dict:
    hour = time.localtime().tm_hour
    peak = (7 <= hour <= 9) or (17 <= hour <= 19)

    vehicle_count   = random.randint(80, 180) if peak else random.randint(10, 90)
    traffic_density = round(random.uniform(0.5, 1.0) if peak else random.uniform(0.1, 0.6), 3)
    average_speed   = round(random.uniform(10, 45) if peak else random.uniform(40, 110), 1)
    weather         = random.choice(WEATHER_CONDITIONS)
    signal          = random.choice(SIGNAL_STATUSES)

    return {
        "vehicle_count":    vehicle_count,
        "traffic_density":  traffic_density,
        "average_speed":    average_speed,
        "weather_condition": weather,
        "signal_status":    signal,
        "timestamp":        pd.Timestamp.now(),
    }


def predict_risk(model, le, live: dict) -> float:
    hour = live["timestamp"].hour + live["timestamp"].minute / 60
    visibility = random.uniform(100, 900)

    weather_enc = le.transform([live["weather_condition"]])[0]

    X = pd.DataFrame([{
        "vehicle_count":    live["vehicle_count"],
        "traffic_density":  live["traffic_density"],
        "average_speed":    live["average_speed"],
        "weather_condition": weather_enc,
        "time_of_day":      hour,
        "visibility":       visibility,
    }])
    return float(np.clip(model.predict(X)[0], 0.0, 1.0))


def risk_label(score: float) -> str:
    if score < RISK_SAFE_THRESHOLD:
        return "SAFE"
    if score < RISK_WARNING_THRESHOLD:
        return "WARNING"
    return "CRITICAL"


def badge_html(label: str) -> str:
    css = {"SAFE": "badge-safe", "WARNING": "badge-warning", "CRITICAL": "badge-critical"}
    return f'<span class="{css[label]}">{label}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers (Plotly dark style)
# ─────────────────────────────────────────────────────────────────────────────
_LAYOUT_BASE = dict(
    paper_bgcolor="#12121A",
    plot_bgcolor="#12121A",
    font=dict(color="#E0E0E0", size=11),
    margin=dict(l=40, r=20, t=36, b=40),
    xaxis=dict(gridcolor="#1E1E2E", showline=False, zeroline=False),
    yaxis=dict(gridcolor="#1E1E2E", showline=False, zeroline=False),
)


def chart_line(y_values, timestamps, title: str, color: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=y_values,
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=color.replace(")", ", 0.08)").replace("rgb", "rgba"),
    ))
    fig.update_layout(title=dict(text=title, font=dict(size=13)), **_LAYOUT_BASE)
    return fig


def chart_gauge(score: float) -> go.Figure:
    label = risk_label(score)
    color_map = {"SAFE": "#00C853", "WARNING": "#FFB300", "CRITICAL": "#FF1744"}
    gauge_color = color_map[label]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": gauge_color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#444", "tickwidth": 1},
            "bar": {"color": gauge_color, "thickness": 0.25},
            "bgcolor": "#1A1A2E",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35],  "color": "#0A2A1A"},
                {"range": [35, 65], "color": "#2A1F00"},
                {"range": [65, 100], "color": "#2A0A0A"},
            ],
            "threshold": {
                "line": {"color": gauge_color, "width": 3},
                "thickness": 0.85,
                "value": round(score * 100, 1),
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#12121A",
        font=dict(color="#E0E0E0"),
        margin=dict(l=20, r=20, t=20, b=20),
        height=220,
    )
    return fig


def chart_feature_importance(importances: pd.Series) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=importances.values[::-1],
        y=importances.index[::-1],
        orientation="h",
        marker=dict(color="#00D4FF", opacity=0.85),
    ))
    fig.update_layout(
        title=dict(text="Feature Importance", font=dict(size=13)),
        **_LAYOUT_BASE,
        height=280,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### AI TRAFFIC CONTROL")
    st.markdown("<hr>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Dashboard Overview",
            "Live Traffic Monitoring",
            "Accident Prediction",
            "Analytics",
            "Model Insights",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<small style='color:#555'>Smart City Infrastructure v1.0</small>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Load / train model
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state["model"] is None:
    model, r2, mse, importances, le = train_model()
    st.session_state["model"]             = model
    st.session_state["model_r2"]          = r2
    st.session_state["model_mse"]         = mse
    st.session_state["feature_importances"] = importances
    st.session_state["le_weather"]        = le
else:
    model      = st.session_state["model"]
    r2         = st.session_state["model_r2"]
    mse        = st.session_state["model_mse"]
    importances = st.session_state["feature_importances"]
    le         = st.session_state["le_weather"]

# ─────────────────────────────────────────────────────────────────────────────
# Fetch / update live data once per page render
# ─────────────────────────────────────────────────────────────────────────────
live = simulate_live_data()
risk_score = predict_risk(model, le, live)
st.session_state["last_live_data"] = live

# Append to rolling history
for key, val in [
    ("history_vehicle_count", live["vehicle_count"]),
    ("history_density",       live["traffic_density"]),
    ("history_risk",          risk_score),
    ("history_timestamps",    live["timestamp"]),
]:
    st.session_state[key].append(val)
    if len(st.session_state[key]) > HISTORY_WINDOW:
        st.session_state[key].pop(0)

# ─────────────────────────────────────────────────────────────────────────────
# Helper to get rolling history as lists (safe even when short)
# ─────────────────────────────────────────────────────────────────────────────
def _hist(key):
    return list(st.session_state[key])


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Dashboard Overview
# ─────────────────────────────────────────────────────────────────────────────
if page == "Dashboard Overview":
    st.markdown('<p class="section-header">Dashboard Overview</p>', unsafe_allow_html=True)
    st.title("AI Smart Traffic Control and Accident Prediction System")
    st.markdown("<hr>", unsafe_allow_html=True)

    label = risk_label(risk_score)
    col_status, col_risk, col_intersections, col_speed = st.columns(4)

    with col_status:
        st.metric("Traffic Status", label)
        st.markdown(badge_html(label), unsafe_allow_html=True)

    with col_risk:
        st.metric("Accident Risk Score", f"{risk_score:.3f}")

    with col_intersections:
        active = random.randint(3, len(INTERSECTIONS))
        st.metric("Active Intersections", f"{active} / {len(INTERSECTIONS)}")

    with col_speed:
        st.metric("Average Speed", f"{live['average_speed']} km/h")

    st.markdown("<hr>", unsafe_allow_html=True)

    col_gauge, col_summary = st.columns([1, 1])

    with col_gauge:
        st.markdown('<p class="section-header">Risk Gauge</p>', unsafe_allow_html=True)
        st.plotly_chart(chart_gauge(risk_score), use_container_width=True)

    with col_summary:
        st.markdown('<p class="section-header">Current Conditions</p>', unsafe_allow_html=True)
        summary_data = {
            "Parameter": [
                "Vehicle Count",
                "Traffic Density",
                "Average Speed",
                "Weather",
                "Signal Status",
            ],
            "Value": [
                live["vehicle_count"],
                f"{live['traffic_density']:.3f}",
                f"{live['average_speed']} km/h",
                live["weather_condition"].capitalize(),
                live["signal_status"].upper(),
            ],
        }
        st.dataframe(
            pd.DataFrame(summary_data),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Risk Trend (Last 60 Readings)</p>', unsafe_allow_html=True)
    if len(_hist("history_risk")) > 1:
        st.plotly_chart(
            chart_line(_hist("history_risk"), _hist("history_timestamps"), "Accident Risk Score", "rgb(255,23,68)"),
            use_container_width=True,
        )

    # Auto-refresh every 2 s
    time.sleep(2)
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Live Traffic Monitoring
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Live Traffic Monitoring":
    st.markdown('<p class="section-header">Live Traffic Monitoring</p>', unsafe_allow_html=True)
    st.title("Real-Time Traffic Feed")
    st.caption("Data refreshes every 2 seconds.")
    st.markdown("<hr>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Vehicle Count",    live["vehicle_count"])
    c2.metric("Traffic Density",  f"{live['traffic_density']:.3f}")
    c3.metric("Average Speed",    f"{live['average_speed']} km/h")
    c4.metric("Signal Status",    live["signal_status"].upper())
    c5.metric("Weather",          live["weather_condition"].capitalize())

    st.markdown("<hr>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<p class="section-header">Traffic Density</p>', unsafe_allow_html=True)
        if len(_hist("history_density")) > 1:
            st.plotly_chart(
                chart_line(_hist("history_density"), _hist("history_timestamps"), "Traffic Density", "rgb(0,212,255)"),
                use_container_width=True,
            )

    with col_b:
        st.markdown('<p class="section-header">Vehicle Count</p>', unsafe_allow_html=True)
        if len(_hist("history_vehicle_count")) > 1:
            st.plotly_chart(
                chart_line(_hist("history_vehicle_count"), _hist("history_timestamps"), "Vehicle Count", "rgb(0,200,83)"),
                use_container_width=True,
            )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Intersection Status</p>', unsafe_allow_html=True)

    cols = st.columns(len(INTERSECTIONS))
    for i, junction in enumerate(INTERSECTIONS):
        signal = random.choice(SIGNAL_STATUSES)
        color_map = {"green": "#00C853", "yellow": "#FFB300", "red": "#FF1744"}
        with cols[i]:
            st.markdown(
                f"""
                <div style='background:#12121A;border:1px solid #1E1E2E;border-radius:8px;
                            padding:12px;text-align:center;'>
                  <div style='font-size:11px;color:#8888AA;letter-spacing:1px;
                              text-transform:uppercase;margin-bottom:6px;'>{junction}</div>
                  <div style='width:14px;height:14px;border-radius:50%;
                              background:{color_map[signal]};margin:auto;'></div>
                  <div style='font-size:12px;color:{color_map[signal]};
                              margin-top:4px;font-weight:600;'>{signal.upper()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    time.sleep(2)
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Accident Prediction
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Accident Prediction":
    st.markdown('<p class="section-header">Accident Prediction Module</p>', unsafe_allow_html=True)
    st.title("Accident Risk Prediction")
    st.markdown("<hr>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown('<p class="section-header">Input Parameters</p>', unsafe_allow_html=True)

        inp_vehicle   = st.slider("Vehicle Count",    5,   200,  live["vehicle_count"])
        inp_density   = st.slider("Traffic Density",  0.1, 1.0,  live["traffic_density"])
        inp_speed     = st.slider("Average Speed (km/h)", 10, 120, int(live["average_speed"]))
        inp_weather   = st.selectbox("Weather Condition", WEATHER_CONDITIONS,
                                     index=WEATHER_CONDITIONS.index(live["weather_condition"]))
        inp_hour      = st.slider("Time of Day (hour)", 0, 23, time.localtime().tm_hour)
        inp_visibility = st.slider("Visibility (m)", 10, 1000, 500)

        st.button("Run Prediction", use_container_width=True)

    # Always compute prediction live – sliders trigger a Streamlit rerun on change
    weather_enc = le.transform([inp_weather])[0]
    X_input = pd.DataFrame([{
        "vehicle_count":    inp_vehicle,
        "traffic_density":  inp_density,
        "average_speed":    inp_speed,
        "weather_condition": weather_enc,
        "time_of_day":      inp_hour,
        "visibility":       inp_visibility,
    }])
    pred_score = float(np.clip(model.predict(X_input)[0], 0.0, 1.0))
    pred_label = risk_label(pred_score)

    with col_result:
        st.markdown('<p class="section-header">Prediction Result</p>', unsafe_allow_html=True)
        st.plotly_chart(chart_gauge(pred_score), use_container_width=True)

        st.markdown(
            f"<div style='text-align:center;margin-top:8px;'>"
            f"{badge_html(pred_label)}"
            f"</div>",
            unsafe_allow_html=True,
        )

        color_map = {"SAFE": "#00C853", "WARNING": "#FFB300", "CRITICAL": "#FF1744"}
        st.markdown(
            f"""
            <div style='background:#12121A;border:1px solid #1E1E2E;border-radius:8px;
                        padding:16px;margin-top:12px;'>
              <div style='font-size:11px;color:#8888AA;letter-spacing:1px;
                          text-transform:uppercase;margin-bottom:8px;'>Risk Score</div>
              <div style='font-size:40px;font-weight:700;
                          color:{color_map[pred_label]};'>{pred_score:.4f}</div>
              <div style='font-size:12px;color:#555;margin-top:4px;'>
                Scale: 0.0 (no risk) — 1.0 (maximum risk)
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Classification Thresholds</p>', unsafe_allow_html=True)
    thresh_df = pd.DataFrame({
        "Status":    ["SAFE",       "WARNING",    "CRITICAL"],
        "Range":     ["0.00 – 0.35", "0.35 – 0.65", "0.65 – 1.00"],
        "Meaning":   [
            "Normal traffic conditions",
            "Elevated risk, increased monitoring recommended",
            "High accident probability, immediate action required",
        ],
    })
    st.dataframe(thresh_df, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Analytics
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Analytics":
    st.markdown('<p class="section-header">Analytics Dashboard</p>', unsafe_allow_html=True)
    st.title("Traffic Analytics")
    st.markdown("<hr>", unsafe_allow_html=True)

    ts   = _hist("history_timestamps")
    vc   = _hist("history_vehicle_count")
    den  = _hist("history_density")
    risk = _hist("history_risk")

    if len(ts) < 2:
        st.info("Collecting data… Please wait a moment and refresh.")
    else:
        st.markdown('<p class="section-header">Traffic Density Over Time</p>', unsafe_allow_html=True)
        st.plotly_chart(
            chart_line(den, ts, "Traffic Density", "rgb(0,212,255)"),
            use_container_width=True,
        )

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown('<p class="section-header">Vehicle Count Trend</p>', unsafe_allow_html=True)
            st.plotly_chart(
                chart_line(vc, ts, "Vehicle Count", "rgb(0,200,83)"),
                use_container_width=True,
            )

        with col_r:
            st.markdown('<p class="section-header">Accident Risk Score Trend</p>', unsafe_allow_html=True)
            st.plotly_chart(
                chart_line(risk, ts, "Risk Score", "rgb(255,23,68)"),
                use_container_width=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Summary Statistics</p>', unsafe_allow_html=True)
        stats = pd.DataFrame({
            "Metric":  ["Vehicle Count", "Traffic Density", "Risk Score"],
            "Min":     [f"{min(vc)}", f"{min(den):.3f}", f"{min(risk):.3f}"],
            "Max":     [f"{max(vc)}", f"{max(den):.3f}", f"{max(risk):.3f}"],
            "Average": [f"{sum(vc)/len(vc):.1f}", f"{sum(den)/len(den):.3f}", f"{sum(risk)/len(risk):.3f}"],
        })
        st.dataframe(stats, hide_index=True, use_container_width=True)

    time.sleep(2)
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Model Insights
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Model Insights":
    st.markdown('<p class="section-header">Model Insights</p>', unsafe_allow_html=True)
    st.title("Accident Prediction Model")
    st.markdown("<hr>", unsafe_allow_html=True)

    col_acc, col_mse, col_est = st.columns(3)
    col_acc.metric("R² Score",         f"{r2:.4f}")
    col_mse.metric("Mean Squared Error", f"{mse:.6f}")
    col_est.metric("Estimators",        "120 trees")

    st.markdown("<hr>", unsafe_allow_html=True)

    col_imp, col_cfg = st.columns([1, 1])

    with col_imp:
        st.markdown('<p class="section-header">Feature Importance</p>', unsafe_allow_html=True)
        st.plotly_chart(chart_feature_importance(importances), use_container_width=True)

        imp_df = importances.reset_index()
        imp_df.columns = ["Feature", "Importance"]
        imp_df["Importance"] = imp_df["Importance"].map(lambda x: f"{x:.4f}")
        st.dataframe(imp_df, hide_index=True, use_container_width=True)

    with col_cfg:
        st.markdown('<p class="section-header">Model Configuration</p>', unsafe_allow_html=True)
        config = {
            "Algorithm":          "Random Forest Regressor",
            "Estimators":         120,
            "Max Depth":          10,
            "Min Samples Split":  5,
            "Min Samples Leaf":   2,
            "Training Samples":   4000,
            "Test Samples":       1000,
            "Target":             "Accident Risk Score [0, 1]",
            "Framework":          "scikit-learn",
        }
        cfg_df = pd.DataFrame(list(config.items()), columns=["Parameter", "Value"])
        st.dataframe(cfg_df, hide_index=True, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Input Features</p>', unsafe_allow_html=True)
        feat_df = pd.DataFrame({
            "Feature":     [
                "vehicle_count", "traffic_density", "average_speed",
                "weather_condition", "time_of_day", "visibility",
            ],
            "Type":        ["Integer", "Float", "Float", "Categorical (encoded)",
                            "Float (0–24)", "Float (metres)"],
            "Contribution": importances.values,
        })
        feat_df["Contribution"] = feat_df["Contribution"].map(lambda x: f"{x:.4f}")
        st.dataframe(feat_df, hide_index=True, use_container_width=True)
