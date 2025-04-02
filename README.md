# MAINA - Maintenance Assistant (with Voice Input)

MAINA is an intelligent maintenance assistant built with Streamlit + LangChain. It lets you:

- Upload maintenance manuals (PDFs)
- Ask questions in text **or voice (WAV)**
- Get step-by-step fixes and source references

## 🚀 How to Run

1. Clone this repo:
```
git clone https://github.com/yourname/MAINA_app_voice.git
cd MAINA_app_voice
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the app:
```
streamlit run maina_app.py
```

## 📁 Folder Structure

```
MAINA_app_voice/
├── maina_app.py
├── requirements.txt
├── README.md
└── sample_manuals/
    └── aztech_flexo_manual.pdf  # (Add this manually)
```

## 🌐 Deploy on Streamlit Cloud

1. Push this folder to GitHub
2. Go to https://streamlit.io/cloud
3. Choose this repo and set `maina_app.py` as the entry point

---
🎙️ Voice input requires `.wav` file uploads using Google Speech Recognition API.

🔧 Built with ❤️ for maintenance professionals.
