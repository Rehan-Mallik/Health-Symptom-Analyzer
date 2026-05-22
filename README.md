# 🤖 Health Symptom Checker AI Agent

> An AI-powered health symptom analyzer built with IBM Watsonx.ai. Users describe symptoms in natural language and receive structured, trustworthy health guidance — not a diagnosis.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![IBM Watsonx](https://img.shields.io/badge/IBM-Watsonx.ai-052FAD)
![LangGraph](https://img.shields.io/badge/LangGraph-ReAct_Agent-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎥 Demo
> [Add your Loom/YouTube demo link here]

---

## 💡 What It Does

Users type symptoms like *"I have a sore throat and fever"* and receive:

| Field | Example Output |
|---|---|
| 🩺 Possible Causes | Viral pharyngitis, strep throat, tonsillitis |
| 📊 Urgency Level | Medium |
| 🏠 Home Care Advice | Rest, warm fluids, salt-water gargle |
| 👨‍⚕️ When to See a Doctor | If fever exceeds 103°F or persists >3 days |

---

## 🏗️ Architecture

```
User Input (natural language symptoms)
        ↓
LangGraph ReAct Agent
        ↓
IBM Watsonx.ai (Mistral Large)
        ↓
Web Search Tools (GoogleSearch / DuckDuckGo / WebCrawler)
        ↓
Structured Health Response
```

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Rehan-Mallik/Health-Symptom-Analyzer.git
cd Health-Symptom-Analyzer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Credentials
```bash
cp .env.example .env
```
Open `.env` and fill in your IBM credentials:
```
WATSONX_API_KEY=your_api_key_here
PROJECT_ID=your_project_id_here
WATSONX_URL=https://eu-gb.ml.cloud.ibm.com
```

**How to get credentials:**
- API Key: [IBM Cloud IAM](https://cloud.ibm.com/iam/apikeys) → Create API Key
- Project ID: Open your [Watsonx.ai project](https://dataplatform.cloud.ibm.com) → Manage → General → Project ID

### 4. Run the Notebook
```bash
jupyter notebook Health-Symptoms-Checker.ipynb
```
Run cells from top to bottom. Enter your symptoms when prompted in Step 11.

---

## 📁 Project Structure

```
Health-Symptom-Analyzer/
├── Health-Symptoms-Checker.ipynb   # Main agent notebook
├── requirements.txt                 # Python dependencies
├── .env.example                     # Credentials template (copy to .env)
├── .gitignore                       # Excludes .env from git
└── README.md
```

---

## 🧪 Example Usage

**Input:**
```
I have a headache and nausea since this morning
```

**Output:**
```
🩺 Possible Causes: Migraine, dehydration, viral infection

📊 Urgency Level: Medium
   Monitor closely; consult a doctor if symptoms worsen.

🏠 Home Care Advice: Rest in a dark quiet room, stay hydrated,
   avoid screens, take OTC pain relief if needed.

👨‍⚕️ When to See a Doctor: If headache is severe/sudden,
   accompanied by fever, stiff neck, or persists beyond 48 hours.

📚 Note: This is not a diagnosis. Always consult a healthcare
   professional for medical concerns.
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Mistral Large via IBM Watsonx.ai |
| Agent Framework | LangGraph (ReAct pattern) |
| LLM Interface | LangChain IBM |
| Search Tools | GoogleSearch, DuckDuckGo, WebCrawler |
| Memory | LangGraph MemorySaver (multi-turn) |
| Language | Python 3.11 |

---

## ⚠️ Limitations

- Not a substitute for professional medical advice
- English-optimized (multilingual support planned)
- Requires active IBM Watsonx.ai account and API credits
- Cannot handle image inputs (text-only)

---

## 🤝 Contributing

Pull requests welcome. Ideas for contribution:
- Multilingual support via IBM Language Translator
- Streamlit web UI
- Integration with symptom APIs (e.g. Infermedica)
- Evaluation dataset expansion

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

> ❗ **Medical Disclaimer:** This tool is for educational purposes only and does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.
