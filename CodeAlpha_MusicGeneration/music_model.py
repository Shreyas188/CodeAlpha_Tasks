"""
CodeAlpha - Music Generation with AI
LSTM Model for Music Generation
Author: CodeAlpha Intern
"""
import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from music_data import SAMPLE_SEQUENCES, NOTE_VOCAB, DURATION_VOCAB

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
SEQUENCE_LENGTH = 16      # Input context window
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001
EMBEDDING_DIM = 64
LSTM_UNITS = 256
DROPOUT_RATE = 0.3
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  Vocabulary & Encoding
# ─────────────────────────────────────────────
class MusicVocabulary:
    """Handles note/duration encoding and decoding."""
    
    def __init__(self):
        self.note2idx = {note: i for i, note in enumerate(NOTE_VOCAB)}
        self.idx2note = {i: note for note, i in self.note2idx.items()}
        self.dur2idx = {dur: i for i, dur in enumerate(DURATION_VOCAB)}
        self.idx2dur = {i: dur for dur, i in self.dur2idx.items()}
        
        self.n_notes = len(NOTE_VOCAB)
        self.n_durations = len(DURATION_VOCAB)
    
    def encode_note(self, note):
        """Map note name to index, with fallback."""
        # Handle enharmonic equivalents
        enharmonics = {"Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B"}
        if note in enharmonics:
            note = enharmonics[note]
        return self.note2idx.get(note, self.note2idx.get("C4", 13))
    
    def encode_duration(self, duration):
        """Map duration to nearest available index."""
        closest = min(DURATION_VOCAB, key=lambda d: abs(d - duration))
        return self.dur2idx[closest]
    
    def decode_note(self, idx):
        return self.idx2note.get(int(idx), "C4")
    
    def decode_duration(self, idx):
        return self.idx2dur.get(int(idx), 1.0)
    
    def save(self, path):
        data = {
            "note_vocab": NOTE_VOCAB,
            "duration_vocab": DURATION_VOCAB
        }
        with open(path, "w") as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path):
        vocab = cls()
        return vocab


# ─────────────────────────────────────────────
#  Data Preparation
# ─────────────────────────────────────────────
def prepare_training_data(vocab: MusicVocabulary, seq_len: int = SEQUENCE_LENGTH):
    """
    Convert SAMPLE_SEQUENCES into X/y pairs for training.
    Returns: X_notes, X_durations, y_notes, y_durations
    """
    all_notes = []
    all_durations = []
    
    # Flatten all sequences into one long sequence
    for sequence in SAMPLE_SEQUENCES:
        for item in sequence:
            note, duration = item
            all_notes.append(vocab.encode_note(note))
            all_durations.append(vocab.encode_duration(duration))
    
    # Data augmentation: transpose sequences
    augmented_notes = list(all_notes)
    augmented_durations = list(all_durations)
    
    for transpose in [-2, -1, 1, 2]:  # Transpose by semitones
        for sequence in SAMPLE_SEQUENCES:
            for item in sequence:
                note, duration = item
                if note != "rest":
                    note_idx = vocab.encode_note(note)
                    new_idx = max(0, min(vocab.n_notes - 1, note_idx + transpose))
                    augmented_notes.append(new_idx)
                else:
                    augmented_notes.append(vocab.encode_note("rest"))
                augmented_durations.append(vocab.encode_duration(duration))
    
    # Create sequences
    X_notes, X_durations = [], []
    y_notes, y_durations = [], []
    
    for i in range(len(augmented_notes) - seq_len):
        X_notes.append(augmented_notes[i:i + seq_len])
        X_durations.append(augmented_durations[i:i + seq_len])
        y_notes.append(augmented_notes[i + seq_len])
        y_durations.append(augmented_durations[i + seq_len])
    
    return (
        np.array(X_notes),
        np.array(X_durations),
        np.array(y_notes),
        np.array(y_durations)
    )


# ─────────────────────────────────────────────
#  Model Architecture
# ─────────────────────────────────────────────
def build_model(vocab: MusicVocabulary, seq_len: int = SEQUENCE_LENGTH):
    """
    Build LSTM model with dual inputs (notes + durations) and dual outputs.
    Uses functional API for multi-input/output.
    """
    # Note input branch
    note_input = tf.keras.Input(shape=(seq_len,), name="note_input")
    note_embed = Embedding(vocab.n_notes, EMBEDDING_DIM, name="note_embedding")(note_input)
    
    # Duration input branch
    dur_input = tf.keras.Input(shape=(seq_len,), name="duration_input")
    dur_embed = Embedding(vocab.n_durations, 16, name="duration_embedding")(dur_input)
    
    # Merge branches
    merged = tf.keras.layers.Concatenate(axis=-1)([note_embed, dur_embed])
    
    # LSTM layers
    x = LSTM(LSTM_UNITS, return_sequences=True, name="lstm_1")(merged)
    x = Dropout(DROPOUT_RATE)(x)
    x = BatchNormalization()(x)
    
    x = LSTM(LSTM_UNITS // 2, return_sequences=False, name="lstm_2")(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = BatchNormalization()(x)
    
    x = Dense(256, activation="relu", name="dense_shared")(x)
    x = Dropout(DROPOUT_RATE / 2)(x)
    
    # Output heads
    note_output = Dense(vocab.n_notes, activation="softmax", name="note_output")(x)
    dur_output = Dense(vocab.n_durations, activation="softmax", name="duration_output")(x)
    
    model = tf.keras.Model(
        inputs=[note_input, dur_input],
        outputs=[note_output, dur_output]
    )
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss={
            "note_output": "sparse_categorical_crossentropy",
            "duration_output": "sparse_categorical_crossentropy"
        },
        loss_weights={"note_output": 1.0, "duration_output": 0.5},
        metrics={"note_output": "accuracy", "duration_output": "accuracy"}
    )
    
    return model


# ─────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────
def train_model(progress_callback=None):
    """
    Train the LSTM music generation model.
    progress_callback: function(epoch, total, loss, accuracy) for UI updates
    """
    print("Preparing music vocabulary...")
    vocab = MusicVocabulary()
    vocab.save(os.path.join(MODEL_DIR, "vocab.json"))
    
    print("Preparing training data...")
    X_notes, X_durations, y_notes, y_durations = prepare_training_data(vocab)
    
    print(f"Training samples: {len(X_notes)}")
    print(f"Note vocabulary size: {vocab.n_notes}")
    print(f"Duration vocabulary size: {vocab.n_durations}")
    
    model = build_model(vocab)
    model.summary()
    
    # Callbacks
    checkpoint_path = os.path.join(MODEL_DIR, "music_model_best.keras")
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor="note_output_accuracy",
            save_best_only=True,
            mode="max",
            verbose=0
        ),
        EarlyStopping(
            monitor="note_output_accuracy",
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=15,
            min_lr=1e-5,
            verbose=1
        )
    ]
    
    # Custom callback for UI progress
    if progress_callback:
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                progress_callback(
                    epoch + 1,
                    EPOCHS,
                    logs.get("loss", 0),
                    logs.get("note_output_accuracy", 0)
                )
        callbacks.append(ProgressCallback())
    
    history = model.fit(
        {"note_input": X_notes, "duration_input": X_durations},
        {"note_output": y_notes, "duration_output": y_durations},
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_split=0.1,
        verbose=1
    )
    
    # Save final model
    final_path = os.path.join(MODEL_DIR, "music_model_final.keras")
    model.save(final_path)
    print(f"\nModel saved to: {final_path}")
    
    return model, vocab, history


# ─────────────────────────────────────────────
#  Music Generation
# ─────────────────────────────────────────────
def generate_music(
    model,
    vocab: MusicVocabulary,
    seed_sequence=None,
    num_notes: int = 64,
    temperature: float = 1.0
):
    """
    Generate a music sequence using the trained LSTM model.
    
    Args:
        model: Trained Keras model
        vocab: MusicVocabulary instance
        seed_sequence: List of (note, duration) tuples as seed (optional)
        num_notes: Number of notes to generate
        temperature: Creativity (0.5=conservative, 1.0=balanced, 1.5=creative)
    
    Returns:
        List of (note_name, duration) tuples
    """
    if seed_sequence is None:
        # Default seed: C major arpeggio
        seed_sequence = [
            ("C4", 1), ("E4", 1), ("G4", 1), ("C5", 1),
            ("G4", 1), ("E4", 1), ("C4", 1), ("D4", 1),
            ("F4", 1), ("A4", 1), ("C5", 1), ("A4", 1),
            ("F4", 1), ("D4", 1), ("G4", 1), ("B4", 1)
        ]
    
    # Encode seed
    seed_notes = [vocab.encode_note(n) for n, d in seed_sequence]
    seed_durs = [vocab.encode_duration(d) for n, d in seed_sequence]
    
    # Pad/trim to SEQUENCE_LENGTH
    seed_notes = seed_notes[-SEQUENCE_LENGTH:]
    seed_durs = seed_durs[-SEQUENCE_LENGTH:]
    
    while len(seed_notes) < SEQUENCE_LENGTH:
        seed_notes.insert(0, vocab.encode_note("C4"))
        seed_durs.insert(0, vocab.encode_duration(1.0))
    
    generated = []
    current_notes = list(seed_notes)
    current_durs = list(seed_durs)
    
    for _ in range(num_notes):
        x_notes = np.array([current_notes])
        x_durs = np.array([current_durs])
        
        note_probs, dur_probs = model.predict(
            {"note_input": x_notes, "duration_input": x_durs},
            verbose=0
        )
        
        # Temperature sampling
        note_probs = note_probs[0].astype("float64")
        dur_probs = dur_probs[0].astype("float64")
        
        note_probs = np.log(note_probs + 1e-10) / temperature
        note_probs = np.exp(note_probs) / np.sum(np.exp(note_probs))
        
        dur_probs = np.log(dur_probs + 1e-10) / temperature
        dur_probs = np.exp(dur_probs) / np.sum(np.exp(dur_probs))
        
        next_note_idx = np.random.choice(len(note_probs), p=note_probs)
        next_dur_idx = np.random.choice(len(dur_probs), p=dur_probs)
        
        note_name = vocab.decode_note(next_note_idx)
        duration = vocab.decode_duration(next_dur_idx)
        
        generated.append((note_name, duration))
        
        current_notes = current_notes[1:] + [next_note_idx]
        current_durs = current_durs[1:] + [next_dur_idx]
    
    return generated


def load_trained_model():
    """Load the best saved model if it exists."""
    model_path = os.path.join(MODEL_DIR, "music_model_best.keras")
    final_path = os.path.join(MODEL_DIR, "music_model_final.keras")
    
    for path in [model_path, final_path]:
        if os.path.exists(path):
            print(f"Loading model from: {path}")
            return load_model(path), MusicVocabulary()
    
    return None, None
