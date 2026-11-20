
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

from pathlib import Path
from sqlalchemy import text

from backend.database import SessionLocal
from backend.models import PredictionLog


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Predictive Maintenance | Industrial AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "train_FD001.txt"

FEATURE_COLUMNS = [
    "cycle",
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21"
]


# =====================================================
# INDUSTRIAL GREEN THEME
# =====================================================

st.markdown("""
<style>

/* Application */

.stApp {
    background: #0C1110;
    color: #E5EDE8;
}

[data-testid="stAppViewContainer"] {
    background: #0C1110;
}

[data-testid="stHeader"] {
    background: rgba(12, 17, 16, 0.95);
}

.block-container {
    max-width: 1550px;
    padding-top: 1.6rem;
    padding-bottom: 2rem;
}

/* Typography */

h1, h2, h3, h4 {
    color: #F0F5F1 !important;
    font-weight: 650 !important;
}

p, label {
    color: #B5C3B9;
}

[data-testid="stCaptionContainer"] {
    color: #829389;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: #121B17;
    border-right: 1px solid #26372E;
}

[data-testid="stSidebar"] > div {
    background: #121B17;
}

[data-testid="stSidebar"] * {
    color: #DCE8DF;
}

/* Main header */

.dashboard-title {
    color: #F0F5F1;
    font-size: 30px;
    font-weight: 750;
    letter-spacing: -0.8px;
    margin-top: 12px;
    margin-bottom: 5px;
    line-height: 1.4;
    padding-top: 5px;
    overflow: visible;
}

.dashboard-subtitle {
    color: #829389;
    font-size: 12px;
    letter-spacing: 2px;
    font-weight: 600;
    line-height: 1.5;
    padding-bottom: 5px;
}


/* Section heading */

.section-heading {
    color: #E7F0E9;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 5px;
}

.section-description {
    color: #829389;
    font-size: 12px;
    margin-bottom: 16px;
}

/* KPI cards */

.kpi-card {
    background: linear-gradient(145deg, #19251F, #141D18);
    border: 1px solid #2A3D32;
    border-radius: 12px;
    padding: 19px;
    min-height: 125px;
}

.kpi-label {
    color: #91A398;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

.kpi-value {
    color: #4ADE80;
    font-size: 29px;
    font-weight: 750;
    margin-top: 12px;
    line-height: 1.2;
}

.kpi-description {
    color: #819187;
    font-size: 11px;
    margin-top: 8px;
}

/* Analysis cards */

.analysis-card {
    background: #16211B;
    border: 1px solid #2A3D32;
    border-radius: 12px;
    padding: 21px;
    min-height: 150px;
}

.analysis-label {
    color: #9AADA0;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}

.analysis-value {
    color: #4ADE80;
    font-size: 32px;
    font-weight: 750;
    margin-top: 14px;
}

.analysis-description {
    color: #8FA195;
    font-size: 12px;
    margin-top: 7px;
}

/* Status indicators */

.status-online {
    display: inline-block;
    color: #22D3EE;
    font-size: 13px;
    font-weight: 700;
    line-height: 1.5;
    padding: 4px 0;
    overflow: visible;
}

.status-offline {
    color: #F87171;
    font-weight: 700;
}

.status-warning {
    color: #FBBF24;
    font-weight: 700;
}

/* Demo banner */

.demo-banner {
    background: #17231C;
    border-left: 4px solid #4ADE80;
    border-radius: 7px;
    padding: 13px 17px;
    color: #C6D5CA;
    font-size: 13px;
    margin: 17px 0;
}

.demo-banner strong {
    color: #4ADE80;
    letter-spacing: 1px;
}

/* Buttons */

div.stButton > button {
    background: #238348;
    color: #FFFFFF !important;
    border: 1px solid #329A59;
    border-radius: 8px;
    min-height: 44px;
    font-weight: 700;
    transition: 0.2s;
}

div.stButton > button:hover {
    background: #2DA35A;
    border-color: #4ADE80;
    color: #FFFFFF !important;
}

/* Inputs */

div[data-baseweb="select"] > div {
    background: #17231C;
    border-color: #344A3B;
    color: #FFFFFF;
}

div[data-baseweb="select"] span {
    color: #E5EDE8;
}

/* Expanders */

[data-testid="stExpander"] {
    background: #141E18;
    border: 1px solid #2A3D32;
    border-radius: 9px;
}

/* Dataframe */

[data-testid="stDataFrame"] {
    border: 1px solid #2A3D32;
    border-radius: 8px;
}

/* Dividers */

hr {
    border-color: #26372E;
}

/* Footer */

.footer {
    text-align: center;
    color: #718278;
    font-size: 11px;
    line-height: 2;
    padding: 18px 0;
    letter-spacing: 0.5px;
}

/* Metric defaults */

[data-testid="stMetricValue"] {
    color: #4ADE80;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def check_api_status():

    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=3
        )

        if response.status_code == 200:
            data = response.json()

            return {
                "online": data.get("status") == "healthy",
                "rul_model": data.get("rul_model") == "loaded",
                "anomaly_model": data.get("anomaly_model") == "loaded"
            }

    except requests.exceptions.RequestException:
        pass

    return {
        "online": False,
        "rul_model": False,
        "anomaly_model": False
    }


def check_database_status():

    db = None

    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))

        return True

    except Exception:
        return False

    finally:
        if db:
            db.close()


def render_kpi(label, value, description):

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_analysis_card(label, value, description, value_color="#4ADE80"):

    st.markdown(
        f"""
        <div class="analysis-card">
            <div class="analysis-label">{label}</div>
            <div class="analysis-value" style="color:{value_color};">
                {value}
            </div>
            <div class="analysis-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# DATA LOADING
# =====================================================

@st.cache_data
def load_dataset():

    columns = (
        ["engine_id", "cycle"]
        + [
            f"operational_setting_{i}"
            for i in range(1, 4)
        ]
        + [
            f"sensor_{i}"
            for i in range(1, 22)
        ]
    )

    data = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None
    )

    data = data.iloc[:, :26]
    data.columns = columns

    return data


try:

    df = load_dataset()

except Exception as error:

    st.error(f"Dataset loading failed: {error}")
    st.stop()


# =====================================================
# SYSTEM STATUS
# =====================================================

api_status = check_api_status()
database_online = check_database_status()


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding:8px 0 16px 0;">
            <div style="
                color:#4ADE80;
                font-size:21px;
                font-weight:800;
                letter-spacing:1px;">
                ⚙ INDUSTRIAL AI
            </div>
            <div style="
                color:#829389;
                font-size:10px;
                letter-spacing:2px;
                margin-top:5px;">
                PREDICTIVE MAINTENANCE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("#### MACHINE SELECTION")

    engine_ids = sorted(
        df["engine_id"].unique().tolist()
    )

    selected_engine = st.selectbox(
        "Select machine",
        engine_ids,
        format_func=lambda x: f"ENGINE {x:03d}"
    )

    engine_data = (
        df[df["engine_id"] == selected_engine]
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    latest_readings = engine_data.tail(50).copy()

    st.divider()

    st.markdown("#### MACHINE INFORMATION")

    st.metric(
        "Machine ID",
        f"ENG-{selected_engine:03d}"
    )

    st.metric(
        "Operating Cycles",
        int(engine_data["cycle"].max())
    )

    st.metric(
        "Available Records",
        len(engine_data)
    )

    st.divider()

    st.markdown("#### SYSTEM COMPONENTS")

    if api_status["online"]:
        st.markdown(
            '<span class="status-online">● API ONLINE</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-offline">● API OFFLINE</span>',
            unsafe_allow_html=True
        )

    if api_status["rul_model"]:
        st.markdown(
            '<span class="status-online">● LSTM MODEL LOADED</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-offline">● LSTM MODEL UNAVAILABLE</span>',
            unsafe_allow_html=True
        )

    if api_status["anomaly_model"]:
        st.markdown(
            '<span class="status-online">● ANOMALY MODEL LOADED</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-offline">● ANOMALY MODEL UNAVAILABLE</span>',
            unsafe_allow_html=True
        )

    if database_online:
        st.markdown(
            '<span class="status-online">● DATABASE CONNECTED</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-offline">● DATABASE DISCONNECTED</span>',
            unsafe_allow_html=True
        )

    st.divider()

    st.caption("DATASET")
    st.caption("NASA C-MAPSS FD001")

    st.caption("MODE")
    st.caption("HISTORICAL DATA DEMONSTRATION")


if len(latest_readings) < 50:

    st.error("At least 50 readings are required.")
    st.stop()


# =====================================================
# TOP HEADER
# =====================================================

header_left, header_right = st.columns([3, 1])

with header_left:

    st.markdown(
        """
        <div class="dashboard-title">
            Machine Intelligence Center
        </div>
        <div class="dashboard-subtitle">
            INDUSTRIAL HEALTH & PREDICTIVE ANALYTICS
        </div>
        """,
        unsafe_allow_html=True
    )

with header_right:

    if api_status["online"] and database_online:

        st.markdown(
            """
            <div style="text-align:right; padding-top:9px;">
                <span class="status-online">
                    ● SYSTEM OPERATIONAL
                </span>
                <div style="color:#829389; font-size:11px; margin-top:5px;">
                    API AND DATABASE CONNECTED
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif api_status["online"]:

        st.markdown(
            """
            <div style="text-align:right; padding-top:9px;">
                <span class="status-warning">
                    ● PARTIAL CONNECTION
                </span>
                <div style="color:#829389; font-size:11px; margin-top:5px;">
                    DATABASE UNAVAILABLE
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div style="text-align:right; padding-top:9px;">
                <span class="status-offline">
                    ● SYSTEM OFFLINE
                </span>
                <div style="color:#829389; font-size:11px; margin-top:5px;">
                    BACKEND NOT CONNECTED
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


st.markdown(
    """
    <div class="demo-banner">
        <strong>DEMO ENVIRONMENT</strong>
        &nbsp; Historical NASA C-MAPSS data is being used.
        This dashboard is not connected to live industrial machinery.
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# MACHINE OVERVIEW
# =====================================================

st.markdown(
    '<div class="section-heading">Machine Overview</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">Current operating profile of the selected machine</div>',
    unsafe_allow_html=True
)

k1, k2, k3, k4 = st.columns(4)

with k1:
    render_kpi(
        "Machine Identifier",
        f"ENG-{selected_engine:03d}",
        "Selected engine"
    )

with k2:
    render_kpi(
        "Current Cycle",
        f"{int(engine_data['cycle'].max())}",
        "Latest operating cycle"
    )

with k3:
    render_kpi(
        "Total Observations",
        f"{len(engine_data):,}",
        "Historical sensor records"
    )

with k4:
    render_kpi(
        "LSTM Sequence",
        "50",
        "Required input observations"
    )


# =====================================================
# SENSOR ANALYTICS
# =====================================================

st.divider()

st.markdown(
    '<div class="section-heading">Sensor Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">Explore historical sensor behavior and operating trends</div>',
    unsafe_allow_html=True
)

sensor_options = [
    f"sensor_{i}"
    for i in range(1, 22)
]

sensor_left, sensor_right = st.columns([1, 3])

with sensor_left:

    selected_sensor = st.selectbox(
        "Choose sensor",
        sensor_options
    )

    latest_sensor_value = engine_data[selected_sensor].iloc[-1]

    render_analysis_card(
        "LATEST SENSOR READING",
        f"{latest_sensor_value:.3f}",
        selected_sensor.upper()
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.metric(
        "Current Operating Cycle",
        int(engine_data["cycle"].max())
    )

    st.metric(
        "Sensor Observations",
        len(engine_data)
    )

with sensor_right:

    fig = px.line(
        engine_data,
        x="cycle",
        y=selected_sensor,
        title=f"{selected_sensor.upper()} | Operating Trend",
        markers=True
    )

    fig.update_traces(
        line=dict(
            color="#4ADE80",
            width=2.5
        ),
        marker=dict(
            size=5,
            color="#86EFAC"
        ),
        hovertemplate=(
            "Cycle: %{x}<br>"
            "Sensor value: %{y:.3f}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#141E18",
        plot_bgcolor="#141E18",
        font=dict(
            color="#B5C3B9",
            family="Arial"
        ),
        title=dict(
            font=dict(
                size=15,
                color="#E5EDE8"
            )
        ),
        margin=dict(
            l=15,
            r=15,
            t=55,
            b=20
        ),
        xaxis_title="Operating Cycle",
        yaxis_title="Sensor Measurement",
        hovermode="x unified",
        height=350
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#26372E",
        zeroline=False
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#26372E",
        zeroline=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False}
    )


# =====================================================
# MACHINE HEALTH ANALYSIS
# =====================================================

st.divider()

analysis_left, analysis_right = st.columns([3, 1])

with analysis_left:

    st.markdown(
        '<div class="section-heading">AI Machine Health Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
            Run the trained LSTM and Isolation Forest models
            against the latest 50 machine observations.
        </div>
        """,
        unsafe_allow_html=True
    )

with analysis_right:

    st.markdown(
        f"""
        <div style="text-align:right; padding-top:8px;">
            <span style="color:#91A398; font-size:11px;">
                SELECTED MACHINE
            </span>
            <div style="color:#4ADE80; font-size:19px; font-weight:700;">
                ENG-{selected_engine:03d}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


if st.button(
    "RUN MACHINE HEALTH ANALYSIS",
    use_container_width=True
):

    readings = []

    for _, row in latest_readings.iterrows():

        readings.append({
            column: float(row[column])
            for column in FEATURE_COLUMNS
        })

    payload = {
        "engine_id": int(selected_engine),
        "readings": [
            {"values": reading}
            for reading in readings
        ]
    }

    try:

        with st.spinner("Analyzing machine condition..."):

            response = requests.post(
                f"{API_URL}/predict/rul",
                json=payload,
                timeout=60
            )

        if response.status_code == 200:

            result = response.json()

            st.session_state["latest_result"] = result

            st.success(
                "Machine health analysis completed successfully."
            )

        else:

            st.error(
                f"API Error {response.status_code}: {response.text}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Backend API is not running. Start FastAPI in another terminal."
        )

    except requests.exceptions.RequestException as error:

        st.error(f"Request failed: {error}")


# =====================================================
# PREDICTION RESULTS
# =====================================================

if "latest_result" in st.session_state:

    result = st.session_state["latest_result"]

    if result["engine_id"] == selected_engine:

        st.divider()

        st.markdown(
            '<div class="section-heading">Machine Health Assessment</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-description">Results from the selected machine analysis</div>',
            unsafe_allow_html=True
        )

        rul_col, anomaly_col, score_col = st.columns(3)

        with rul_col:

            render_analysis_card(
                "ESTIMATED REMAINING USEFUL LIFE",
                f"{result['predicted_rul_cycles']:.2f}",
                "Predicted remaining operating cycles"
            )

        with anomaly_col:

            if result["anomaly_status"] == "anomaly":

                status_color = "#F87171"
                status_text = "ANOMALY"

            else:

                status_color = "#4ADE80"
                status_text = "NORMAL"

            render_analysis_card(
                "ANOMALY DETECTION",
                status_text,
                "Latest sensor observation",
                status_color
            )

        with score_col:

            render_analysis_card(
                "ISOLATION FOREST SCORE",
                f"{result['anomaly_score']:.4f}",
                "Anomaly detection score"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.warning(
            "An anomaly means an unusual sensor pattern, not confirmed machine failure. "
            "RUL is a model estimate, not a guaranteed failure countdown."
        )

        if result.get("database_saved"):

            st.caption(
                "✓ Prediction and sensor readings successfully saved to PostgreSQL."
            )

        else:

            st.caption(
                "Database save was not confirmed."
            )


# =====================================================
# SENSOR DATA TABLE
# =====================================================

st.divider()

st.markdown(
    '<div class="section-heading">Recent Sensor Records</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">Latest 50 observations used for model inference</div>',
    unsafe_allow_html=True
)

with st.expander("VIEW SENSOR DATA"):

    st.dataframe(
        latest_readings,
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# PREDICTION HISTORY
# =====================================================

st.divider()

st.markdown(
    '<div class="section-heading">Prediction History</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">Previously recorded machine health predictions</div>',
    unsafe_allow_html=True
)

try:

    db = SessionLocal()

    try:

        history = (
            db.query(PredictionLog)
            .order_by(PredictionLog.created_at.desc())
            .limit(50)
            .all()
        )

        if history:

            history_data = [
                {
                    "Engine": f"ENG-{item.engine_id:03d}",
                    "Predicted RUL": (
                        round(item.predicted_rul, 2)
                        if item.predicted_rul is not None
                        else "N/A"
                    ),
                    "Anomaly Status": item.anomaly_status.upper(),
                    "Anomaly Score": round(item.anomaly_score, 4),
                    "Timestamp": item.created_at
                }
                for item in history
            ]

            history_df = pd.DataFrame(history_data)

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info("No prediction history available yet.")

    finally:

        db.close()

except Exception as error:

    st.error(f"Could not load prediction history: {error}")


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <strong style="color:#4ADE80;">
            PREDICTIVE MAINTENANCE SYSTEM
        </strong>
        <br>
        NASA C-MAPSS FD001
        &nbsp; | &nbsp; LSTM
        &nbsp; | &nbsp; ISOLATION FOREST
        &nbsp; | &nbsp; FASTAPI
        &nbsp; | &nbsp; POSTGRESQL
        <br>
        AI / ML ENGINEERING PROJECT
    </div>
    """,
    unsafe_allow_html=True
)
