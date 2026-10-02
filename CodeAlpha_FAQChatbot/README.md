# 🤖 CodeAlpha FAQ Chatbot — Task 2

> **AI-powered FAQ chatbot** built with Python, CustomTkinter, and scikit-learn NLP.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🎨 **Dark Chat UI** | WhatsApp/Telegram-style dark theme built with CustomTkinter |
| 💬 **Message Bubbles** | User messages (right, blue) · Bot messages (left, dark green) |
| 🧠 **NLP Engine** | TF-IDF vectorization + cosine similarity via scikit-learn |
| 🎯 **Intent Detection** | Greetings, farewells, thanks, and FAQ matching |
| 📊 **Confidence Score** | Every answer shows a `% match` confidence |
| ⌨️ **Typing Indicator** | Animated "AI is thinking…" while processing |
| 📚 **Sidebar Topics** | Click any category to ask about it instantly |
| 🕒 **Timestamps** | Every message shows the time it was sent |
| 🔍 **Smart Fallback** | Suggests related questions when no good match is found |
| 🗑️ **Clear Chat** | One-click chat reset |
| 📈 **Status Bar** | Live category · confidence · message count display |

---

## 📁 Project Structure

```
CodeAlpha_FAQChatbot/
├── chatbot.py          # Main application (UI + event handlers)
├── nlp_engine.py       # NLP processing (TF-IDF, cosine similarity, intents)
├── faqs.py             # FAQ data — 40+ Q&A pairs across 6 categories
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the chatbot

```bash
python chatbot.py
```

NLTK stopwords and punkt tokenizer are downloaded automatically on first run.

---

## 📚 FAQ Categories

| Category | Count | Topics |
|---|---|---|
| 🐍 Python Basics | 8 | Data types, lists, dicts, tuples, lambdas, comprehensions |
| 🔷 OOP | 5 | Classes, inheritance, polymorphism, encapsulation |
| 🤖 Machine Learning | 16 | Supervised/unsupervised, neural nets, sklearn, TensorFlow, PyTorch |
| 📊 Data Science | 5 | pandas, numpy, matplotlib, preprocessing, feature engineering |
| 🐙 Git/GitHub | 3 | git commands, GitHub, pushing code |
| 💡 General | 3 | APIs, JSON, virtual environments |

---

## 🧠 How the NLP Works

```
User Input
    │
    ▼
Intent Classification (regex keyword matching)
    │
    ├── greeting  →  "Hello! 👋 ..."
    ├── farewell  →  "Goodbye! 👋 ..."
    ├── thanks    →  "You're welcome! 😊 ..."
    │
    └── faq  ──►  Preprocess text (lowercase, remove punctuation, remove stopwords)
                      │
                      ▼
                  TF-IDF vectorize input + question corpus
                      │
                      ▼
                  Cosine similarity → best match
                      │
                      ├── confidence > 0.10  →  Return matched answer + category
                      └── confidence ≤ 0.10  →  Fallback with top-3 suggestions
```

---

## 📦 Dependencies

```
customtkinter==5.2.2   # Modern Tkinter UI framework
scikit-learn==1.5.2    # TF-IDF + cosine similarity
nltk==3.9.1            # Stopwords, tokenization
numpy==1.26.4          # Array operations
pillow==10.4.0         # Image support for CustomTkinter
```

---

## 🎓 About

**CodeAlpha Internship — Artificial Intelligence Track**  
Task 2: FAQ Chatbot using NLP

Built with ❤️ using Python 3.10+
