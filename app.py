import os
import streamlit as st
from dotenv import load_dotenv

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Health Symptom Checker",
    page_icon="🩺",
    layout="centered"
)

# ── Load credentials ───────────────────────────────────────
load_dotenv()
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY") or st.secrets.get("WATSONX_API_KEY")
PROJECT_ID      = os.getenv("PROJECT_ID")      or st.secrets.get("PROJECT_ID")
WATSONX_URL     = os.getenv("WATSONX_URL", "https://eu-gb.ml.cloud.ibm.com") or st.secrets.get("WATSONX_URL", "https://eu-gb.ml.cloud.ibm.com")

MODEL_ID   = "mistralai/mistral-small-3-1-24b-instruct-2503"

SYSTEM_PROMPT = """You are an AI Health Symptom Checker. Only answer questions related to health symptoms and medications.
Do NOT answer questions about programming, academics, or any unrelated topics.

ALWAYS respond using this exact structure:

🩺 Possible Causes: [1-3 likely health issues, no medical diagnosis]

📊 Urgency Level: [Low / Medium / High]
- Low: Can be monitored at home.
- Medium: Monitor closely; consult a doctor if symptoms worsen.
- High: Immediate medical attention recommended.

🏠 Home Care Advice: [Safe, evidence-based tips only]

👨‍⚕️ When to See a Doctor: [Clear signs or duration thresholds]

📚 Note: This is not a diagnosis. Always consult a healthcare professional for medical concerns.

URGENCY RULES:
Set Urgency HIGH for: chest pain, difficulty breathing, severe headache,
unconsciousness, blood in vomit/stool, high fever in infants, seizures, sudden vision loss.
For HIGH urgency add: 🚨 Please visit a hospital or emergency center immediately.

SAFETY RULES:
- Never diagnose conditions
- Never prescribe medications
- Only use WHO/CDC verified sources
- Always end with the safety disclaimer"""


@st.cache_resource(show_spinner="🔌 Connecting to IBM Watsonx.ai...")
def load_model():
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference

    credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
    client      = APIClient(credentials=credentials, project_id=PROJECT_ID)

    model = ModelInference(
        model_id=MODEL_ID,
        api_client=client,
        params={
            "max_new_tokens": 800,
            "temperature":    0.12,
            "top_p":          1
        }
    )
    return model

def ask_model(question: str) -> str:
    if not question.strip():
        return "Please describe your symptoms."
    try:
        model = load_model()
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {question}\nAssistant:"
        response = model.generate_text(prompt=full_prompt)
        return response
    except Exception as e:
        return f"❌ Error: {e}\n\nPlease check your credentials and try again."
# ── UI ─────────────────────────────────────────────────────
st.title("🩺 Health Symptom Checker")
st.caption("Powered by IBM Watsonx.ai · For educational purposes only · Not a medical diagnosis")

st.info(
    "Describe your symptoms in plain language. "
    "Example: *'I have a sore throat and fever'*",
    icon="💡"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Describe your symptoms here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing your symptoms..."):
            result = ask_model(prompt)
            st.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This AI agent analyzes symptoms and provides:
    - 🩺 Possible causes
    - 📊 Urgency level
    - 🏠 Home care advice
    - 👨‍⚕️ When to see a doctor

    **Built with:**
    - IBM Watsonx.ai
    - Mistral Small
    - Streamlit
    """)
    st.divider()
    st.warning("⚠️ For educational purposes only. Not a substitute for professional medical advice.", icon="🚨")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
