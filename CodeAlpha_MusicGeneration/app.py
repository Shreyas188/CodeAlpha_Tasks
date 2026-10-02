"""
CodeAlpha - Music Generation with AI
Main Application UI
Author: CodeAlpha Intern
"""
import customtkinter as ctk
import threading
import os
import time
from tkinter import messagebox, filedialog
from datetime import datetime
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class MusicGenerationApp:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("CodeAlpha - AI Music Generation")
        self.root.geometry("1100x750")
        self.root.resizable(True, True)

        self.model = None
        self.vocab = None
        self.generated_sequence = None
        self.training_thread = None
        self.is_training = False

        self.epoch_data = []
        self.loss_data = []
        self.acc_data = []

        self.setup_ui()

    # ─── UI Setup ─────────────────────────────────────────────────────────────
    def setup_ui(self):
        # Title bar
        title_frame = ctk.CTkFrame(self.root, fg_color="#1a1a2e", height=70)
        title_frame.pack(fill="x")
        title_frame.pack_propagate(False)

        ctk.CTkLabel(
            title_frame,
            text="🎵 AI Music Generation",
            font=ctk.CTkFont(family="Helvetica", size=24, weight="bold"),
            text_color="#00d4ff"
        ).pack(side="left", padx=20, pady=15)

        ctk.CTkLabel(
            title_frame,
            text="LSTM Neural Network | CodeAlpha Internship",
            font=ctk.CTkFont(size=11),
            text_color="#888888"
        ).pack(side="left", padx=5, pady=15)

        # Main layout
        main = ctk.CTkFrame(self.root, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=10, pady=5)

        # Left panel - Controls
        left = ctk.CTkFrame(main, fg_color="#16213e", width=320)
        left.pack(side="left", fill="y", padx=(0, 5))
        left.pack_propagate(False)
        self._build_control_panel(left)

        # Right panel - Visualization + Output
        right = ctk.CTkFrame(main, fg_color="#0f0f23")
        right.pack(side="left", fill="both", expand=True)
        self._build_right_panel(right)

        # Status bar
        self.status_bar = ctk.CTkLabel(
            self.root,
            text="✅ Ready. Train the model first, then generate music!",
            font=ctk.CTkFont(size=11),
            text_color="#888888",
            fg_color="#0a0a1a",
            height=30
        )
        self.status_bar.pack(fill="x")

    def _build_control_panel(self, parent):
        ctk.CTkLabel(
            parent, text="⚙️ Controls",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#00d4ff"
        ).pack(pady=(20, 5), padx=15, anchor="w")

        ctk.CTkLabel(
            parent, text="─" * 35,
            font=ctk.CTkFont(size=10), text_color="#333355"
        ).pack(padx=15, anchor="w")

        # Training section
        ctk.CTkLabel(
            parent, text="🧠 Model Training",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=(15, 5), padx=15, anchor="w")

        # Epochs
        ep_row = ctk.CTkFrame(parent, fg_color="transparent")
        ep_row.pack(fill="x", padx=15, pady=3)
        ctk.CTkLabel(ep_row, text="Epochs:", width=100, anchor="w").pack(side="left")
        self.epochs_var = ctk.StringVar(value="100")
        ctk.CTkEntry(ep_row, textvariable=self.epochs_var, width=80).pack(side="left")

        # LSTM units
        lstm_row = ctk.CTkFrame(parent, fg_color="transparent")
        lstm_row.pack(fill="x", padx=15, pady=3)
        ctk.CTkLabel(lstm_row, text="LSTM Units:", width=100, anchor="w").pack(side="left")
        self.lstm_var = ctk.StringVar(value="256")
        ctk.CTkComboBox(lstm_row, values=["128", "256", "512"], variable=self.lstm_var, width=80).pack(side="left")

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(parent, width=270)
        self.progress_bar.pack(pady=(10, 5), padx=15)
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(
            parent, text="Epoch: 0/0 | Loss: - | Acc: -",
            font=ctk.CTkFont(size=11), text_color="#888888"
        )
        self.progress_label.pack(padx=15, pady=(0, 5))

        # Train button
        self.train_btn = ctk.CTkButton(
            parent,
            text="🚀 Train Model",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=45,
            command=self.start_training,
            fg_color="#00d4ff",
            text_color="#000000",
            hover_color="#00a8cc"
        )
        self.train_btn.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(
            parent, text="─" * 35,
            font=ctk.CTkFont(size=10), text_color="#333355"
        ).pack(padx=15, anchor="w", pady=5)

        # Generation section
        ctk.CTkLabel(
            parent, text="🎵 Music Generation",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=(5, 5), padx=15, anchor="w")

        # Number of notes
        notes_row = ctk.CTkFrame(parent, fg_color="transparent")
        notes_row.pack(fill="x", padx=15, pady=3)
        ctk.CTkLabel(notes_row, text="Notes:", width=100, anchor="w").pack(side="left")
        self.num_notes_var = ctk.IntVar(value=64)
        ctk.CTkSlider(
            notes_row, from_=16, to=128,
            variable=self.num_notes_var,
            command=self.update_notes_label,
            width=120
        ).pack(side="left", padx=5)
        self.notes_label = ctk.CTkLabel(notes_row, text="64", width=30)
        self.notes_label.pack(side="left")

        # Temperature
        temp_row = ctk.CTkFrame(parent, fg_color="transparent")
        temp_row.pack(fill="x", padx=15, pady=3)
        ctk.CTkLabel(temp_row, text="Creativity:", width=100, anchor="w").pack(side="left")
        self.temp_var = ctk.DoubleVar(value=1.0)
        ctk.CTkSlider(
            temp_row, from_=0.3, to=2.0,
            variable=self.temp_var,
            command=self.update_temp_label,
            width=120
        ).pack(side="left", padx=5)
        self.temp_label = ctk.CTkLabel(temp_row, text="1.0", width=40)
        self.temp_label.pack(side="left")

        # Tempo
        tempo_row = ctk.CTkFrame(parent, fg_color="transparent")
        tempo_row.pack(fill="x", padx=15, pady=3)
        ctk.CTkLabel(tempo_row, text="Tempo (BPM):", width=100, anchor="w").pack(side="left")
        self.tempo_var = ctk.StringVar(value="120")
        ctk.CTkComboBox(
            tempo_row,
            values=["60", "80", "100", "120", "140", "160", "180"],
            variable=self.tempo_var, width=80
        ).pack(side="left")

        # Instrument
        inst_row = ctk.CTkFrame(parent, fg_color="transparent")
        inst_row.pack(fill="x", padx=15, pady=3)
        ctk.CTkLabel(inst_row, text="Instrument:", width=100, anchor="w").pack(side="left")
        self.instrument_var = ctk.StringVar(value="Piano")
        ctk.CTkComboBox(
            inst_row,
            values=["Piano", "Violin", "Flute", "Guitar", "Trumpet", "Organ"],
            variable=self.instrument_var, width=120
        ).pack(side="left")

        # Generate button
        self.gen_btn = ctk.CTkButton(
            parent,
            text="🎼 Generate Music",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=45,
            command=self.generate_music,
            fg_color="#7c3aed",
            hover_color="#6d28d9",
            state="disabled"
        )
        self.gen_btn.pack(fill="x", padx=15, pady=5)

        # Save & Play buttons
        btn_row = ctk.CTkFrame(parent, fg_color="transparent")
        btn_row.pack(fill="x", padx=15, pady=3)

        self.save_btn = ctk.CTkButton(
            btn_row, text="💾 Save MIDI",
            command=self.save_midi,
            fg_color="#1b5e20", hover_color="#2e7d32",
            state="disabled", width=120
        )
        self.save_btn.pack(side="left", padx=2)

        self.play_btn = ctk.CTkButton(
            btn_row, text="▶️ Play",
            command=self.play_midi,
            fg_color="#0f3460", hover_color="#16213e",
            state="disabled", width=100
        )
        self.play_btn.pack(side="left", padx=2)

        # Load model button
        ctk.CTkButton(
            parent,
            text="📂 Load Saved Model",
            command=self.load_model,
            fg_color="#333355",
            hover_color="#444466",
            height=35
        ).pack(fill="x", padx=15, pady=(10, 5))

    def _build_right_panel(self, parent):
        # Tab view
        self.tabview = ctk.CTkTabview(parent)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # Tab 1: Training Chart
        tab_train = self.tabview.add("📈 Training Progress")
        self._build_training_chart(tab_train)

        # Tab 2: Generated Notes
        tab_notes = self.tabview.add("🎵 Generated Notes")
        self._build_notes_display(tab_notes)

        # Tab 3: Piano Roll
        tab_piano = self.tabview.add("🎹 Piano Roll")
        self._build_piano_roll(tab_piano)

        # Tab 4: Model Info
        tab_info = self.tabview.add("ℹ️ Model Info")
        self._build_model_info(tab_info)

    def _build_training_chart(self, parent):
        self.fig_train, (self.ax_loss, self.ax_acc) = plt.subplots(1, 2, figsize=(8, 4))
        self.fig_train.patch.set_facecolor("#0f0f23")
        for ax in [self.ax_loss, self.ax_acc]:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white")
            ax.spines["bottom"].set_color("#444466")
            ax.spines["left"].set_color("#444466")
            ax.spines["top"].set_color("#444466")
            ax.spines["right"].set_color("#444466")

        self.ax_loss.set_title("Training Loss", color="white")
        self.ax_loss.set_xlabel("Epoch", color="white")
        self.ax_loss.set_ylabel("Loss", color="white")

        self.ax_acc.set_title("Note Accuracy", color="white")
        self.ax_acc.set_xlabel("Epoch", color="white")
        self.ax_acc.set_ylabel("Accuracy", color="white")

        self.canvas_train = FigureCanvasTkAgg(self.fig_train, master=parent)
        self.canvas_train.get_tk_widget().pack(fill="both", expand=True)

    def _build_notes_display(self, parent):
        self.notes_text = ctk.CTkTextbox(
            parent,
            font=ctk.CTkFont(family="Courier", size=12),
            fg_color="#0f0f23",
            wrap="word"
        )
        self.notes_text.pack(fill="both", expand=True)
        self.notes_text.insert("1.0", "Generated notes will appear here after generation...\n\nTrain the model first, then click 'Generate Music'!")
        self.notes_text.configure(state="disabled")

    def _build_piano_roll(self, parent):
        self.fig_piano, self.ax_piano = plt.subplots(figsize=(8, 4))
        self.fig_piano.patch.set_facecolor("#0f0f23")
        self.ax_piano.set_facecolor("#16213e")
        self.ax_piano.set_title("Piano Roll Visualization", color="white")
        self.ax_piano.set_xlabel("Time (beats)", color="white")
        self.ax_piano.set_ylabel("MIDI Note", color="white")
        self.ax_piano.tick_params(colors="white")

        self.canvas_piano = FigureCanvasTkAgg(self.fig_piano, master=parent)
        self.canvas_piano.get_tk_widget().pack(fill="both", expand=True)

    def _build_model_info(self, parent):
        info_text = """
╔══════════════════════════════════════════════════╗
║     CodeAlpha - AI Music Generation              ║
║     Task 3: Music Generation with AI             ║
╚══════════════════════════════════════════════════╝

📌 Architecture:
   • Model Type:    Bidirectional LSTM
   • Input:         Note sequences (length = 16)
   • Embeddings:    Note (64-dim) + Duration (16-dim)
   • LSTM Layer 1:  256 units (return sequences)
   • LSTM Layer 2:  128 units
   • Outputs:       Note prediction + Duration prediction
   • Optimization:  Adam + ReduceLROnPlateau

📌 Training Data:
   • 32 hand-crafted musical sequences
   • Augmented with ±1, ±2 semitone transpositions
   • Covers: Classical, Jazz, Blues, Pentatonic, Modal

📌 Generation:
   • Temperature sampling for creativity control
   • 0.5 = conservative (closer to training data)
   • 1.0 = balanced
   • 1.5+ = experimental/creative

📌 Output:
   • MIDI file (.mid) playable on any music app
   • Piano roll visualization
   • Note sequence display with durations

📌 Technologies:
   • TensorFlow / Keras (LSTM model)
   • music21 (music theory)
   • midiutil (MIDI file creation)
   • CustomTkinter (modern UI)
   • Matplotlib (visualization)
"""
        info_box = ctk.CTkTextbox(
            parent,
            font=ctk.CTkFont(family="Courier", size=12),
            fg_color="#0f0f23"
        )
        info_box.pack(fill="both", expand=True)
        info_box.insert("1.0", info_text)
        info_box.configure(state="disabled")

    # ─── Actions ──────────────────────────────────────────────────────────────
    def update_notes_label(self, val):
        self.notes_label.configure(text=str(int(float(val))))

    def update_temp_label(self, val):
        self.temp_label.configure(text=f"{float(val):.1f}")

    def start_training(self):
        if self.is_training:
            return
        self.is_training = True
        self.train_btn.configure(text="⏳ Training...", state="disabled")
        self.epoch_data.clear()
        self.loss_data.clear()
        self.acc_data.clear()
        self.progress_bar.set(0)

        epochs = int(self.epochs_var.get() or 100)

        threading.Thread(
            target=self._train_thread,
            args=(epochs,),
            daemon=True
        ).start()

    def _train_thread(self, epochs):
        try:
            import music_model as mm
            mm.EPOCHS = epochs

            self.root.after(0, lambda: self.status_bar.configure(
                text="🧠 Preparing training data..."
            ))

            model, vocab, history = mm.train_model(
                progress_callback=self._on_epoch_end
            )
            self.model = model
            self.vocab = vocab

            self.root.after(0, self._on_training_complete)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.root.after(0, lambda: messagebox.showerror("Training Error", f"{e}\n\n{tb}"))
            self.root.after(0, lambda: self.train_btn.configure(text="🚀 Train Model", state="normal"))
            self.is_training = False

    def _on_epoch_end(self, epoch, total, loss, accuracy):
        progress = epoch / total
        self.root.after(0, lambda: self.progress_bar.set(progress))
        self.root.after(0, lambda: self.progress_label.configure(
            text=f"Epoch: {epoch}/{total} | Loss: {loss:.4f} | Acc: {accuracy:.2%}"
        ))
        self.root.after(0, lambda: self.status_bar.configure(
            text=f"🧠 Training... Epoch {epoch}/{total} | Loss: {loss:.4f} | Accuracy: {accuracy:.2%}"
        ))

        self.epoch_data.append(epoch)
        self.loss_data.append(loss)
        self.acc_data.append(accuracy)

        if epoch % 5 == 0:
            self.root.after(0, self._update_training_chart)

    def _update_training_chart(self):
        self.ax_loss.clear()
        self.ax_acc.clear()

        for ax in [self.ax_loss, self.ax_acc]:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white")

        self.ax_loss.plot(self.epoch_data, self.loss_data, color="#00d4ff", linewidth=2)
        self.ax_loss.fill_between(self.epoch_data, self.loss_data, alpha=0.2, color="#00d4ff")
        self.ax_loss.set_title("Training Loss", color="white")
        self.ax_loss.set_xlabel("Epoch", color="white")
        self.ax_loss.set_ylabel("Loss", color="white")

        self.ax_acc.plot(self.epoch_data, self.acc_data, color="#7c3aed", linewidth=2)
        self.ax_acc.fill_between(self.epoch_data, self.acc_data, alpha=0.2, color="#7c3aed")
        self.ax_acc.set_title("Note Accuracy", color="white")
        self.ax_acc.set_xlabel("Epoch", color="white")
        self.ax_acc.set_ylabel("Accuracy", color="white")

        self.fig_train.tight_layout()
        self.canvas_train.draw()

    def _on_training_complete(self):
        self.is_training = False
        self.progress_bar.set(1)
        self.train_btn.configure(text="✅ Re-Train", state="normal")
        self.gen_btn.configure(state="normal")
        self.status_bar.configure(text="✅ Training complete! Click 'Generate Music' to create a composition.")
        messagebox.showinfo("Training Complete", "🎉 Model trained successfully!\nNow click 'Generate Music' to create a composition.")

    def generate_music(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Please train the model first!")
            return

        self.gen_btn.configure(state="disabled", text="⏳ Generating...")
        self.status_bar.configure(text="🎵 Generating music sequence...")

        threading.Thread(target=self._generate_thread, daemon=True).start()

    def _generate_thread(self):
        try:
            from music_model import generate_music
            num_notes = int(self.num_notes_var.get())
            temperature = self.temp_var.get()

            sequence = generate_music(
                self.model,
                self.vocab,
                num_notes=num_notes,
                temperature=temperature
            )
            self.generated_sequence = sequence

            self.root.after(0, self._on_generation_complete, sequence)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.root.after(0, lambda: messagebox.showerror("Generation Error", f"{e}\n\n{tb}"))
        finally:
            self.root.after(0, lambda: self.gen_btn.configure(state="normal", text="🎼 Generate Music"))

    def _on_generation_complete(self, sequence):
        # Update notes display
        self.notes_text.configure(state="normal")
        self.notes_text.delete("1.0", "end")

        lines = [f"{'Note':<10} {'Duration':<12} {'Beat':>8}"]
        lines.append("─" * 35)

        beat = 0
        for note, duration in sequence:
            lines.append(f"{note:<10} {duration:<12.2f} {beat:>8.2f}")
            beat += duration

        lines.append("─" * 35)
        lines.append(f"Total notes: {len(sequence)}")
        lines.append(f"Total beats: {beat:.1f}")

        self.notes_text.insert("1.0", "\n".join(lines))
        self.notes_text.configure(state="disabled")

        # Draw piano roll
        self._draw_piano_roll(sequence)

        # Enable save/play
        self.save_btn.configure(state="normal")
        self.play_btn.configure(state="normal")

        # Auto-save MIDI
        os.makedirs("generated", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_midi_path = os.path.join("generated", f"music_{ts}.mid")
        self._save_to_path(self.current_midi_path)

        self.status_bar.configure(
            text=f"✅ Generated {len(sequence)} notes! MIDI saved to: {self.current_midi_path}"
        )

    def _draw_piano_roll(self, sequence):
        from midi_utils import NOTE_TO_MIDI_NUM
        self.ax_piano.clear()
        self.ax_piano.set_facecolor("#16213e")

        time = 0
        cmap = plt.cm.plasma
        total_beats = sum(d for _, d in sequence)

        for note, duration in sequence:
            if note != "rest":
                midi_num = NOTE_TO_MIDI_NUM.get(note, 60)
                color = cmap(midi_num / 127)
                self.ax_piano.barh(
                    midi_num, duration, left=time,
                    height=0.7, color=color, alpha=0.85
                )
            time += duration

        self.ax_piano.set_title("Piano Roll", color="white", fontsize=12)
        self.ax_piano.set_xlabel("Time (beats)", color="white")
        self.ax_piano.set_ylabel("MIDI Note", color="white")
        self.ax_piano.tick_params(colors="white")
        self.ax_piano.set_xlim(0, total_beats)

        self.fig_piano.tight_layout()
        self.canvas_piano.draw()
        self.tabview.set("🎹 Piano Roll")

    def _save_to_path(self, path):
        from midi_utils import sequence_to_midi
        instrument_programs = {
            "Piano": 0, "Violin": 40, "Flute": 73,
            "Guitar": 25, "Trumpet": 56, "Organ": 19
        }
        program = instrument_programs.get(self.instrument_var.get(), 0)
        tempo = int(self.tempo_var.get() or 120)
        sequence_to_midi(self.generated_sequence, path, tempo=tempo, program=program)

    def save_midi(self):
        if not self.generated_sequence:
            messagebox.showwarning("No Music", "Generate music first!")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".mid",
            filetypes=[("MIDI files", "*.mid"), ("All files", "*.*")],
            initialfile=f"CodeAlpha_Music_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mid"
        )
        if path:
            self._save_to_path(path)
            self.current_midi_path = path
            messagebox.showinfo("Saved", f"🎵 MIDI file saved!\n{path}")

    def play_midi(self):
        if not hasattr(self, "current_midi_path") or not os.path.exists(self.current_midi_path):
            messagebox.showwarning("No MIDI", "Generate and save music first!")
            return
        from midi_utils import play_midi_windows
        if not play_midi_windows(self.current_midi_path):
            messagebox.showinfo("Play", f"Open this file in a MIDI player:\n{self.current_midi_path}")

    def load_model(self):
        self.status_bar.configure(text="📂 Looking for saved model...")
        threading.Thread(target=self._load_model_thread, daemon=True).start()

    def _load_model_thread(self):
        try:
            from music_model import load_trained_model
            model, vocab = load_trained_model()
            if model:
                self.model = model
                self.vocab = vocab
                self.root.after(0, lambda: self.gen_btn.configure(state="normal"))
                self.root.after(0, lambda: self.status_bar.configure(
                    text="✅ Model loaded! Click 'Generate Music' to create a composition."
                ))
                self.root.after(0, lambda: messagebox.showinfo("Model Loaded", "✅ Saved model loaded successfully!"))
            else:
                self.root.after(0, lambda: messagebox.showwarning(
                    "No Model Found", "No saved model found.\nPlease train the model first."
                ))
                self.root.after(0, lambda: self.status_bar.configure(text="No saved model found. Train first."))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Load Error", str(e)))

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = MusicGenerationApp()
    app.run()
