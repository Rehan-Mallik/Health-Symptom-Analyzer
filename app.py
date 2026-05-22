import os
import streamlit as st
from dotenv import load_dotenv
from langchain_ibm import ChatWatsonx
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models.utils import Toolkit
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Health Symptom Checker",
    page_icon="🩺",
    layout="centered"
)

# ── Load credentials ───────────────────────────────────────
load_dotenv()
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
PROJECT_ID      = os.getenv("PROJECT_ID")
WATSONX_URL     = os.getenv("WATSONX_URL", "https://eu-gb.ml.cloud.ibm.com")

MODEL_ID   = "mistralai/mistral-small-3-1-24b-instruct-2503"
PARAMETERS = {
    "frequency_penalty": -0.22,
    "max_tokens":         2000,
    "presence_penalty":   0.37,
    "temperature":        0.12,
    "top_p":              1
}

AGENT_INSTRUCTIONS = """
You are an AI Health Symptom Checker. Only answer questions related to health symptoms and medications.
Do NOT answer questions about programming, academics, or any unrelated topics.

ALWAYS respond using this exact structure:

🩺 Possible Causes: [1–3 likely health issues, no medical diagnosis]

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
- Never prescribe medications or dosages
- Only recommend remedies from WHO/CDC/verified sources
- If input is vague, ask for more details
- Always end with the safety disclaimer
"""

# ── Initialize agent (cached so it only builds once) ───────
@st.cache_resource(show_spinner="🔌 Connecting to IBM Watsonx.ai...")
def load_agent():
    credentials = {"url": WATSONX_URL, "apikey": WATSONX_API_KEY}

    client = APIClient(credentials=credentials, project_id=PROJECT_ID)

    def create_utility_tool(tool_name, params):
        toolkit      = Toolkit(api_client=client)
        utility_tool = toolkit.get_tool(tool_name)
        description  = (
            utility_tool.get("agent_description")
            or utility_tool.get("description", tool_name)
        )
        tool_schema = utility_tool.get("input_schema") or {
            "type": "object",
            "additionalProperties": False,
            "$schema": "http://json-schema.org/draft-07/schema#",
            "properties": {"input": {"description": "input", "type": "string"}}
        }
        def run_tool(**tool_input):
            query = tool_input if utility_tool.get("input_schema") else tool_input.get("input")
            try:
                return utility_tool.run(input=query, config=params).get("output", "No results.")
            except Exception as e:
                return f"Tool error: {e}"
        return StructuredTool(name=tool_name, description=description,
                              func=run_tool, args_schema=tool_schema)

    tools = []
    for name, cfg in [("GoogleSearch", None), ("DuckDuckGo", {}), ("WebCrawler", {})]:
        try:
            tools.append(create_utility_tool(name, cfg))
        except Exception:
            pass

    chat_model = ChatWatsonx(
        model_id=MODEL_ID,
        url=WATSONX_URL,
        project_id=PROJECT_ID,
        params=PARAMETERS,
        watsonx_client=client,
    )

    memory = MemorySaver()
    agent  = create_react_agent(
        chat_model,
        tools=tools,
        checkpointer=memory,
        prompt=AGENT_INSTRUCTIONS
    )
    return agent


# ── UI ─────────────────────────────────────────────────────
st.title("🩺 Health Symptom Checker")
st.caption("Powered by IBM Watsonx.ai · For educational purposes only · Not a medical diagnosis")

st.info(
    "Describe your symptoms in plain language and get structured health guidance. "
    "Examples: *'I have a sore throat and fever'* or *'My child has a runny nose and cough'*",
    icon="💡"
)

# Chat history stored in session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
if prompt := st.chat_input("Describe your symptoms here..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing your symptoms..."):
            try:
                agent = load_agent()
                response = agent.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    {"configurable": {"thread_id": "streamlit-session"}}
                )
                result = response["messages"][-1].content
                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})

            except Exception as e:
                err = f"❌ Error: {e}\n\nPlease check your credentials and try again."
                st.error(err)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This AI agent analyzes symptoms you describe and provides:
    - 🩺 Possible causes
    - 📊 Urgency level
    - 🏠 Home care advice
    - 👨‍⚕️ When to see a doctor

    **Built with:**
    - IBM Watsonx.ai
    - Mistral Small (LLM)
    - LangGraph ReAct Agent
    - Streamlit
    """)
    st.divider()
    st.warning(
        "⚠️ This tool is for **educational purposes only** "
        "and does not replace professional medical advice.",
        icon="🚨"
    )
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
