# 🎵 CodeAlpha - AI Music Generation

> **Task 3 | CodeAlpha AI Internship**  
> Generate original music compositions using Deep Learning (LSTM Neural Networks)

---

## 🌟 Overview

This project implements an **AI music composition system** using Long Short-Term Memory (LSTM) neural networks. The model learns musical patterns from note sequences and generates original compositions that can be exported as MIDI files.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **LSTM Model** | Dual-input LSTM for simultaneous note + duration prediction |
| 🎵 **Music Generation** | Generate 16–128 notes with temperature-controlled creativity |
| 🎹 **Piano Roll** | Real-time matplotlib piano roll visualization |
| 💾 **MIDI Export** | Save as standard MIDI files playable in any DAW |
| 📈 **Training Charts** | Live loss and accuracy plots during training |
| 🎚️ **Instrument Selection** | Piano, Violin, Flute, Guitar, Trumpet, Organ |
| ⚡ **Data Augmentation** | Auto-transposes training data ±2 semitones |
| 🎨 **Dark Modern UI** | Professional CustomTkinter interface |

## 🧠 Model Architecture

```
Note Input (seq_len=16)   Duration Input (seq_len=16)
         │                          │
   Embedding(64-dim)          Embedding(16-dim)
         │                          │
         └──────── Concat ──────────┘
                      │
               LSTM(256 units)
                      │
               LSTM(128 units)
                      │
               Dense(256, ReLU)
              /                \
    Dense(n_notes, softmax)   Dense(n_durations, softmax)
    [Note Output]             [Duration Output]
```

## 🎼 Training Data

The model trains on **32 musical sequences** covering:
- 🎼 **Classical** - Major/minor scale patterns, Ode to Joy motifs
- 🎷 **Jazz** - Chromatic runs, seventh chord arpeggios  
- 🎸 **Blues** - Blues scale patterns
- 🎵 **Pentatonic** - Folk-style melodies
- 🌊 **Modal** - Dorian mode patterns

Auto-augmented with ±1, ±2 semitone transpositions for richer training data.

## 🛠️ Technologies

- **TensorFlow / Keras** — LSTM neural network
- **midiutil** — MIDI file creation
- **music21** — Music theory support
- **CustomTkinter** — Modern dark UI
- **Matplotlib** — Training charts + piano roll

## 📦 Installation

```bash
# Navigate to project directory
cd CodeAlpha_MusicGeneration

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## 🚀 Usage

1. **Launch** the application: `python app.py`
2. **Set epochs** (100 recommended for good results, 200 for better)
3. **Click "Train Model"** — watch live training charts
4. **Click "Generate Music"** — adjust creativity temperature
5. **View** the piano roll visualization
6. **Save** as MIDI and play in Windows Media Player / GarageBand / MuseScore

## 🎛️ Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| Epochs | 50-200 | Training iterations (more = better quality) |
| Notes | 16-128 | Length of generated composition |
| Temperature | 0.3-2.0 | 0.5=safe, 1.0=balanced, 1.5+=experimental |
| Tempo | 60-180 BPM | Playback speed |
| Instrument | 6 options | MIDI instrument program |

## 📁 File Structure

```
CodeAlpha_MusicGeneration/
├── app.py              # Main GUI application
├── music_model.py      # LSTM model architecture + training
├── music_data.py       # Training note sequences + vocabulary
├── midi_utils.py       # MIDI file creation utilities
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── models/             # Saved model weights (created on training)
    ├── music_model_best.keras
    └── music_model_final.keras
└── generated/          # Auto-saved MIDI outputs
```

## 📸 Screenshots

> UI screenshots to be added after running the application.

## 📄 License

MIT License — Free to use and modify.

---

**Made with ❤️ for CodeAlpha AI Internship**  
*Shrey | October 2026*
