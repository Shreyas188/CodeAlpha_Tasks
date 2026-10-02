# 🌐 CodeAlpha – Language Translation Tool

> **CodeAlpha Internship | Task 1** — A professional desktop language translation application built with Python and Google Translate.

---

## 📖 Description

The **Language Translation Tool** is a sleek, fully-featured desktop application that lets you instantly translate text between 24+ languages. It features auto language detection, text-to-speech playback, clipboard copy, and a polished dark/light theme UI — all powered by Google Translate via the `deep-translator` library.

---

## ✨ Features

- 🔤 **Auto Language Detection** — Automatically identifies the source language using `langdetect`
- 🌍 **24+ Languages** — English, Spanish, French, German, Hindi, Japanese, Arabic, Chinese, and more
- ⇄ **Swap Languages** — One-click swap of source ↔ target language (and text content)
- 🔊 **Text-to-Speech** — Listen to both source and translated text via `pyttsx3`
- 📋 **Copy to Clipboard** — Instantly copy the translated text with one click
- 🗑️ **Clear All** — Reset both text panels in a single click
- 📊 **Character Count** — Live character counters for both input and output panels
- ☀️🌙 **Dark / Light Mode Toggle** — Switch themes on the fly
- ⚡ **Non-blocking Translations** — API calls run on background threads; the UI never freezes
- ✅ **Error Handling** — User-friendly error dialogs for network or API failures

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| `customtkinter` | Modern, themed Tkinter UI components |
| `deep-translator` | Google Translate API wrapper |
| `langdetect` | Automatic source language detection |
| `pyttsx3` | Offline text-to-speech engine |
| `pyperclip` | Cross-platform clipboard access |
| `Pillow` | Image handling (future icon/asset support) |

---

## 🚀 Installation

### Prerequisites

- Python **3.8+** installed and added to PATH

### Steps

1. **Clone / download** this project folder.

2. **Navigate** to the project directory:
   ```bash
   cd CodeAlpha_LanguageTranslation
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

---

## 📋 Usage

1. Launch the app with `python app.py`.
2. Select a **source language** (or leave it as *Auto Detect*).
3. Select a **target language** from the **To:** dropdown.
4. Type or paste your text into the **Source Text** panel.
5. Click **TRANSLATE ➡️** to get the translation.
6. Use the action buttons below each panel:
   - **🔊 Speak** — Hear the text read aloud.
   - **📋 Copy** — Copy the translation to your clipboard.
   - **🗑️ Clear** — Wipe both panels clean.
7. Use **⇄ Swap** to reverse the language pair and swap the text.
8. Toggle **☀️ Light Mode / 🌙 Dark Mode** from the top-right corner.

---

## 📸 Screenshots

> _Screenshots will be added after first run._

| Dark Mode | Light Mode |
|-----------|------------|
| _(coming soon)_ | _(coming soon)_ |

---

## 📁 Project Structure

```
CodeAlpha_LanguageTranslation/
│
├── app.py              # Main application (UI + logic)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 CodeAlpha Internship

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

*Built with ❤️ for the CodeAlpha AI Internship Program.*
