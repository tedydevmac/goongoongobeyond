"""Streamlit dashboard for rail vehicle condition monitoring."""
import io
import os
import tempfile

import pandas as pd
import streamlit as st

from subsystems import get_subsystem, list_subsystems


st.set_page_config(
    page_title="Rail CdM Model Hub",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background: #f7f9fc; }
    [data-testid="stHeader"] { background: rgba(247,249,252,0.85); }
    .hero {
        padding: 1.6rem 2rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(125deg, #102a43 0%, #1f5f8b 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0 0 .35rem; font-size: 2.15rem; }
    .hero p { margin: 0; color: #d9ecff; font-size: 1.02rem; }
    .section-title { margin-top: .7rem; margin-bottom: .15rem; }
    .muted { color: #52606d; font-size: .9rem; }
    .status-card {
        border: 1px solid #d9e2ec;
        border-radius: 12px;
        padding: .85rem 1rem;
        background: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


SUBSYSTEM_DETAILS = {
    "Rail Corrugation": {
        "icon": "🛤️",
        "task": "Three-class fault classification",
        "metric": "Normal · Side I · Side II",
        "format": "CSV vibration/shock files",
    },
    "Door": {
        "icon": "🚪",
        "task": "Cycle detection and resistance diagnosis",
        "metric": "Normal vs. Abnormal resistance",
        "format": "One continuous CSV stream",
    },
    "SHM": {
        "icon": "📈",
        "task": "Cumulative fatigue-damage regression",
        "metric": "Numeric damage estimate",
        "format": "CSV stress time-series files",
    },
}


def _read_preview(uploaded_file):
    """Read a small preview without consuming the uploader's buffer."""
    try:
        data = uploaded_file.getvalue()
        if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(data), nrows=5)
        return pd.read_csv(io.BytesIO(data), nrows=5)
    except Exception:
        return None


def _render_result_summary(result_df, subsystem_name):
    count = len(result_df)
    if subsystem_name == "Door":
        abnormal = int((result_df["prediction"] == "Abnormal resistance").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Detected cycles", count)
        c2.metric("Normal cycles", count - abnormal)
        c3.metric("Abnormal resistance", abnormal)
    elif subsystem_name == "SHM":
        values = pd.to_numeric(result_df["prediction"], errors="coerce")
        c1, c2 = st.columns(2)
        c1.metric("Files processed", count)
        c2.metric("Highest estimated damage", f"{values.max():.4g}" if values.notna().any() else "—")
    else:
        st.metric("Files processed", count)


with st.sidebar:
    st.markdown("## 🚆 Model Hub")
    st.caption("Rail vehicle condition monitoring")
    st.divider()
    st.markdown("### Workflow")
    st.markdown("1. Select a subsystem\n2. Upload its sensor data\n3. Review the result\n4. Download the submission CSV")

st.markdown(
    """
    <div class="hero">
      <h1>Rail Vehicle Condition Monitoring</h1>
      <p>Run condition-monitoring models for three independent train subsystems.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Choose a subsystem")
subsystem_name = st.selectbox(
    "Subsystem",
    list_subsystems(),
    format_func=lambda name: f"{SUBSYSTEM_DETAILS[name]['icon']}  {name}",
    label_visibility="collapsed",
)
subsystem = get_subsystem(subsystem_name)
details = SUBSYSTEM_DETAILS[subsystem_name]

info_left, info_mid, info_right = st.columns([1.1, 1.1, 1.4])
with info_left:
    st.markdown(f"**Task**  \n{details['task']}")
with info_mid:
    st.markdown(f"**Output**  \n{details['metric']}")
with info_right:
    st.markdown(f"**Input**  \n{details['format']}")

ready = subsystem.is_ready()
if ready:
    st.success("Model is ready — upload data to begin.", icon="✅")
else:
    st.warning(
        f"{subsystem_name} is not configured yet. Add the required artifact(s) under "
        f"`models/{os.path.basename(subsystem.MODEL_DIR)}/` before running it.",
        icon="⚠️",
    )

st.markdown('<div class="section-title">Upload data</div>', unsafe_allow_html=True)
st.caption(subsystem.FILE_HINT)
accepted_types = ["csv"]
if subsystem.INPUT_MODE == "single_stream":
    uploaded = st.file_uploader(
        "Continuous stream file",
        type=accepted_types,
        accept_multiple_files=False,
        help="Upload the complete unlabeled stream. The model detects individual cycles.",
    )
    uploaded_files = [uploaded] if uploaded is not None else []
else:
    uploaded_files = st.file_uploader(
        "Input file(s)",
        type=accepted_types,
        accept_multiple_files=True,
        help="You can select multiple files for batch inference.",
    ) or []

if uploaded_files:
    st.markdown(f"**{len(uploaded_files)} file(s) selected**")
    file_rows = []
    for uploaded_file in uploaded_files:
        preview = _read_preview(uploaded_file)
        file_rows.append(
            {
                "File": uploaded_file.name,
                "Size": f"{uploaded_file.size / 1024:.1f} KB",
                "Preview": "Available" if preview is not None else "Could not read",
            }
        )
    st.dataframe(pd.DataFrame(file_rows), hide_index=True, width="stretch")
    with st.expander("Preview first selected file", expanded=False):
        preview = _read_preview(uploaded_files[0])
        if preview is None:
            st.info("A preview is unavailable, but the file will still be passed to the model.")
        else:
            st.dataframe(preview, hide_index=True, width="stretch")

run_clicked = st.button(
    "▶  Run prediction",
    disabled=(not uploaded_files or not ready),
    type="primary",
    use_container_width=True,
)

if run_clicked:
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_paths = []
        for uploaded_file in uploaded_files:
            path = os.path.join(tmp_dir, uploaded_file.name)
            with open(path, "wb") as output_file:
                output_file.write(uploaded_file.getbuffer())
            file_paths.append(path)

        st.markdown("### Prediction progress")
        progress_bar = st.progress(0.0, text="Preparing input files…")
        result_df = None
        try:
            result_df = subsystem.run(
                file_paths,
                progress_callback=lambda fraction: progress_bar.progress(
                    max(0.0, min(1.0, float(fraction))),
                    text=f"Running {subsystem_name} model…",
                ),
            )
        except NotImplementedError as error:
            st.error(str(error), icon="❌")
        except (ValueError, KeyError) as error:
            st.error(f"Input validation failed: {error}", icon="❌")
        except Exception as error:
            st.error(f"Prediction failed: {error}", icon="❌")
        finally:
            progress_bar.empty()

    if result_df is not None:
        st.markdown("### Results")
        st.success(f"Prediction complete — {len(result_df)} result row(s) generated.", icon="✅")
        _render_result_summary(result_df, subsystem_name)
        st.dataframe(result_df, hide_index=True, width="stretch")
        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇  Download {subsystem.OUTPUT_FILENAME}",
            data=csv_bytes,
            file_name=subsystem.OUTPUT_FILENAME,
            mime="text/csv",
            type="primary",
            use_container_width=True,
            help="This CSV uses the exact filename and column schema required for submission.",
        )
