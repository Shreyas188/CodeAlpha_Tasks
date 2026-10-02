# 🤖 CodeAlpha AI Internship Projects

> **All 4 AI/ML Tasks | October 2026**  
> Professional implementations with modern UIs and real-world techniques

---

## 📋 Projects Overview

| # | Task | Tech Stack | Status |
|---|------|-----------|--------|
| 1 | [Language Translation Tool](./CodeAlpha_LanguageTranslation/) | deep-translator, CustomTkinter, pyttsx3 | ✅ Complete |
| 2 | [FAQ Chatbot](./CodeAlpha_FAQChatbot/) | scikit-learn, NLTK, TF-IDF, CustomTkinter | ✅ Complete |
| 3 | [Music Generation with AI](./CodeAlpha_MusicGeneration/) | TensorFlow, LSTM, midiutil, Matplotlib | ✅ Complete |
| 4 | [Object Detection & Tracking](./CodeAlpha_ObjectDetection/) | YOLOv8, OpenCV, SORT Tracker | ✅ Complete |

---

## ⚡ Quick Start

```bash
# Install all dependencies at once
python setup_all.py

# Run each project
cd CodeAlpha_LanguageTranslation && python app.py
cd CodeAlpha_FAQChatbot          && python chatbot.py  
cd CodeAlpha_MusicGeneration     && python app.py
cd CodeAlpha_ObjectDetection     && python app.py
```

---

## 🛠️ Tech Stack Overview

```
Python 3.12
├── UI Framework:     CustomTkinter (modern dark UI)
├── Translation:      deep-translator (Google Translate API)
├── NLP:             scikit-learn + NLTK (TF-IDF cosine similarity)
├── Deep Learning:   TensorFlow/Keras (LSTM for music)
├── Computer Vision: YOLOv8 + OpenCV (object detection)
├── Tracking:        SORT algorithm (Kalman + Hungarian)
└── Music:           midiutil + music21 (MIDI generation)
```

---

## 📁 Repository Structure

```
CodeAlpha_AI/
├── setup_all.py                    # One-click dependency installer
├── README.md                       # This file
│
├── CodeAlpha_LanguageTranslation/  # Task 1
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
│
├── CodeAlpha_FAQChatbot/           # Task 2
│   ├── chatbot.py
│   ├── faqs.py
│   ├── nlp_engine.py
│   ├── requirements.txt
│   └── README.md
│
├── CodeAlpha_MusicGeneration/      # Task 3
│   ├── app.py
│   ├── music_model.py
│   ├── music_data.py
│   ├── midi_utils.py
│   ├── requirements.txt
│   └── README.md
│
└── CodeAlpha_ObjectDetection/      # Task 4
    ├── app.py
    ├── detector.py
    ├── sort_tracker.py
    ├── requirements.txt
    └── README.md
```

---

## 👨‍💻 Author

**Shrey** | CodeAlpha AI Internship | October 2026

---

*Submitted as part of CodeAlpha AI Internship Program*
