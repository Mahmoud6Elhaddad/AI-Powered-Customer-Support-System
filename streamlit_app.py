import streamlit as st
import requests
import os

# ==============================
# Configuration
# ==============================

FASTAPI_URL = os.getenv(
    "FASTAPI_URL",
    "http://localhost:8000/ticket"
)

HEALTH_URL = FASTAPI_URL.replace("/ticket", "/health")

st.set_page_config(
    page_title="Support Ticket System",
    page_icon="🎫",
    layout="wide"
)


# ==============================
# Session State
# ==============================

if "result" not in st.session_state:
    st.session_state.result = None


# ==============================
# UI Layout
# ==============================

st.title("🎫 Support Ticket Processing System")
st.markdown("### Submit a support ticket and get automated analysis")

col1, col2 = st.columns([1, 1])


# ==============================
# Left Column (Form)
# ==============================

with col1:
    st.subheader("📝 Ticket Details")

    ticket_title = st.text_input(
        "Ticket Title",
        placeholder="Enter ticket title...",
        help="Brief summary of the issue"
    )

    ticket_description = st.text_area(
        "Ticket Description",
        placeholder="Describe the issue in detail...",
        height=200,
        help="Detailed description of the problem"
    )

    submit_button = st.button(
        "Process Ticket",
        type="primary",
        use_container_width=True
    )


# ==============================
# Right Column (Results)
# ==============================

with col2:
    st.subheader("📊 Analysis Results")
    result_placeholder = st.empty()


# ==============================
# Process Ticket
# ==============================

if submit_button:

    if not ticket_title or not ticket_description:
        st.error("⚠️ Please fill in both title and description!")
        st.stop()

    with st.spinner("Processing your ticket..."):

        try:
            payload = {
                "title": ticket_title,
                "description": ticket_description
            }

            response = requests.post(
                FASTAPI_URL,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                try:
                    result = response.json()
                except Exception:
                    st.error("Invalid JSON response from backend.")
                    st.stop()

                st.session_state.result = result

            else:
                st.error(
                    f"Backend error {response.status_code}: {response.text}"
                )
                st.stop()

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Cannot connect to FastAPI backend.\n\n"
                "Make sure it is running."
            )
            st.stop()

        except requests.exceptions.Timeout:
            st.error("⏳ Backend request timed out.")
            st.stop()

        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            st.stop()


# ==============================
# Display Results
# ==============================

if st.session_state.result:

    result = st.session_state.result

    with result_placeholder.container():

        st.success("✅ Ticket processed successfully!")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric(
                label="Category",
                value=result.get("category", "N/A").upper()
            )

        with metric_col2:
            priority = result.get("priority", "N/A").upper()

            if priority == "HIGH":
                st.metric("Priority", f"🔴 {priority}")
            elif priority == "MEDIUM":
                st.metric("Priority", f"🟡 {priority}")
            else:
                st.metric("Priority", f"🟢 {priority}")

        with metric_col3:
            sentiment = result.get("sentiment", "N/A").upper()

            if sentiment == "POSITIVE":
                st.metric("Sentiment", f"😊 {sentiment}")
            elif sentiment == "NEGATIVE":
                st.metric("Sentiment", f"😞 {sentiment}")
            else:
                st.metric("Sentiment", f"😐 {sentiment}")

        st.divider()

        st.markdown("#### 💡 Suggested Solution")
        st.info(result.get("suggested_solution", "No solution available"))

        with st.expander("🔍 View Raw JSON Response"):
            st.json(result)


# ==============================
# Sidebar
# ==============================

with st.sidebar:

    st.header("About")

    st.markdown("""
This system automatically processes support tickets using:

- **ML Classification**
- **Priority Detection**
- **Sentiment Analysis**
- **RAG System**
""")

    st.markdown("---")
    st.subheader("Backend Status")

    try:
        health_response = requests.get(HEALTH_URL, timeout=2)

        if health_response.status_code == 200:
            st.success("✅ Backend is running")
        else:
            st.warning("⚠️ Backend responding with errors")

    except:
        st.error("❌ Backend is not running")

    st.markdown("---")
    st.subheader("How to Run")

    st.code(
        "uvicorn app:app --reload",
        language="bash"
    )

    st.code(
        "streamlit run streamlit_app.py",
        language="bash"
    )


# ==============================
# Footer
# ==============================

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;'>"
    "Built with Streamlit + FastAPI + RAG BY ENG/MAHMOUD MOHAMED"
    "</div>",
    unsafe_allow_html=True
)