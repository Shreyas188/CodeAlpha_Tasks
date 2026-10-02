import customtkinter as ctk
import threading
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
import pyttsx3
import pyperclip
from tkinter import messagebox

# Ensure consistent language detection results
DetectorFactory.seed = 0

# Language mapping: Display Name -> Language Code
LANGUAGES = {
    "Auto Detect": "auto",
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese (Simplified)": "zh-CN",
    "Chinese (Traditional)": "zh-TW",
    "Arabic": "ar",
    "Hindi": "hi",
    "Bengali": "bn",
    "Dutch": "nl",
    "Swedish": "sv",
    "Turkish": "tr",
    "Polish": "pl",
    "Vietnamese": "vi",
    "Indonesian": "id",
    "Thai": "th",
    "Greek": "el",
    "Hebrew": "he",
    "Ukrainian": "uk",
}


class TranslationApp:
    """Main application class for the Language Translation Tool."""

    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("CodeAlpha - Language Translation Tool")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)

        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.is_translating = False

        self.setup_ui()

    def setup_ui(self):
        """Build and arrange all UI components."""

        # ── Title Bar ───────────────────────────────────────────────────────
        title_frame = ctk.CTkFrame(self.root, fg_color="#1a1a2e", height=70)
        title_frame.pack(fill="x", padx=0, pady=0)
        title_frame.pack_propagate(False)

        ctk.CTkLabel(
            title_frame,
            text="🌐 Language Translation Tool",
            font=ctk.CTkFont(family="Helvetica", size=24, weight="bold"),
            text_color="#00d4ff",
        ).pack(side="left", padx=20, pady=15)

        ctk.CTkLabel(
            title_frame,
            text="Powered by Google Translate | CodeAlpha Internship",
            font=ctk.CTkFont(size=11),
            text_color="#888888",
        ).pack(side="left", padx=5, pady=15)

        # Theme toggle button (stored for later text updates)
        self.theme_var = ctk.StringVar(value="dark")
        self.theme_btn = ctk.CTkButton(
            title_frame,
            text="☀️ Light Mode",
            width=120,
            command=self.toggle_theme,
            fg_color="#16213e",
            hover_color="#0f3460",
        )
        self.theme_btn.pack(side="right", padx=20, pady=15)

        # ── Language Selection Bar ──────────────────────────────────────────
        lang_frame = ctk.CTkFrame(self.root, fg_color="#16213e", height=60)
        lang_frame.pack(fill="x", padx=10, pady=(5, 0))
        lang_frame.pack_propagate(False)

        ctk.CTkLabel(
            lang_frame, text="From:", font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left", padx=(15, 5), pady=15)

        lang_names = list(LANGUAGES.keys())

        self.src_lang = ctk.CTkComboBox(
            lang_frame,
            values=lang_names,
            width=180,
            font=ctk.CTkFont(size=12),
            command=self.on_src_lang_change,
        )
        self.src_lang.set("Auto Detect")
        self.src_lang.pack(side="left", padx=5, pady=15)

        # Label that shows which language was auto-detected
        self.detected_label = ctk.CTkLabel(
            lang_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="#00d4ff",
        )
        self.detected_label.pack(side="left", padx=5)

        # Swap languages button
        ctk.CTkButton(
            lang_frame,
            text="⇄ Swap",
            width=80,
            command=self.swap_languages,
            fg_color="#0f3460",
            hover_color="#533483",
        ).pack(side="left", padx=20, pady=15)

        ctk.CTkLabel(
            lang_frame, text="To:", font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left", padx=(5, 5), pady=15)

        lang_names_no_auto = [l for l in lang_names if l != "Auto Detect"]

        self.tgt_lang = ctk.CTkComboBox(
            lang_frame,
            values=lang_names_no_auto,
            width=180,
            font=ctk.CTkFont(size=12),
        )
        self.tgt_lang.set("Spanish")
        self.tgt_lang.pack(side="left", padx=5, pady=15)

        # ── Main Content Area ───────────────────────────────────────────────
        content_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Left Panel – Source Text ──────────────────────────────────────────
        left_panel = ctk.CTkFrame(content_frame, fg_color="#16213e")
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 5))

        src_header = ctk.CTkFrame(left_panel, fg_color="#0f3460", height=40)
        src_header.pack(fill="x", padx=0, pady=0)
        src_header.pack_propagate(False)

        ctk.CTkLabel(
            src_header, text="Source Text", font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left", padx=15, pady=10)

        self.src_char_count = ctk.CTkLabel(
            src_header,
            text="0 chars",
            font=ctk.CTkFont(size=11),
            text_color="#888888",
        )
        self.src_char_count.pack(side="right", padx=15, pady=10)

        self.src_text = ctk.CTkTextbox(
            left_panel, font=ctk.CTkFont(size=14), fg_color="#1a1a2e", wrap="word"
        )
        self.src_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.src_text.bind("<KeyRelease>", self.update_char_count)

        # Source action buttons
        src_btn_frame = ctk.CTkFrame(left_panel, fg_color="transparent", height=45)
        src_btn_frame.pack(fill="x", padx=10, pady=(0, 10))
        src_btn_frame.pack_propagate(False)

        ctk.CTkButton(
            src_btn_frame,
            text="🔊 Speak",
            width=90,
            command=lambda: self.speak_text(self.src_text.get("1.0", "end-1c")),
            fg_color="#533483",
            hover_color="#7b52ab",
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            src_btn_frame,
            text="🗑️ Clear",
            width=90,
            command=self.clear_all,
            fg_color="#c62a2a",
            hover_color="#e53935",
        ).pack(side="left", padx=2)

        # Middle – Translate Button ─────────────────────────────────────────
        mid_panel = ctk.CTkFrame(content_frame, fg_color="transparent", width=120)
        mid_panel.pack(side="left", fill="y", padx=5)
        mid_panel.pack_propagate(False)

        self.translate_btn = ctk.CTkButton(
            mid_panel,
            text="TRANSLATE\n➡️",
            width=100,
            height=80,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.translate,
            fg_color="#00d4ff",
            text_color="#000000",
            hover_color="#00a8cc",
        )
        self.translate_btn.place(relx=0.5, rely=0.5, anchor="center")

        # Right Panel – Translated Text ─────────────────────────────────────
        right_panel = ctk.CTkFrame(content_frame, fg_color="#16213e")
        right_panel.pack(side="left", fill="both", expand=True, padx=(5, 0))

        tgt_header = ctk.CTkFrame(right_panel, fg_color="#0f3460", height=40)
        tgt_header.pack(fill="x", padx=0, pady=0)
        tgt_header.pack_propagate(False)

        ctk.CTkLabel(
            tgt_header,
            text="Translated Text",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side="left", padx=15, pady=10)

        self.tgt_char_count = ctk.CTkLabel(
            tgt_header,
            text="0 chars",
            font=ctk.CTkFont(size=11),
            text_color="#888888",
        )
        self.tgt_char_count.pack(side="right", padx=15, pady=10)

        self.tgt_text = ctk.CTkTextbox(
            right_panel,
            font=ctk.CTkFont(size=14),
            fg_color="#1a1a2e",
            wrap="word",
            state="disabled",
        )
        self.tgt_text.pack(fill="both", expand=True, padx=10, pady=10)

        # Target action buttons
        tgt_btn_frame = ctk.CTkFrame(right_panel, fg_color="transparent", height=45)
        tgt_btn_frame.pack(fill="x", padx=10, pady=(0, 10))
        tgt_btn_frame.pack_propagate(False)

        ctk.CTkButton(
            tgt_btn_frame,
            text="🔊 Speak",
            width=90,
            command=lambda: self.speak_text(self.tgt_text.get("1.0", "end-1c")),
            fg_color="#533483",
            hover_color="#7b52ab",
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            tgt_btn_frame,
            text="📋 Copy",
            width=90,
            command=self.copy_translation,
            fg_color="#1b5e20",
            hover_color="#2e7d32",
        ).pack(side="left", padx=2)

        # ── Status Bar ──────────────────────────────────────────────────────
        self.status_bar = ctk.CTkLabel(
            self.root,
            text="Ready to translate. Enter text and click TRANSLATE.",
            font=ctk.CTkFont(size=11),
            text_color="#888888",
            fg_color="#0f0f23",
            height=30,
        )
        self.status_bar.pack(fill="x", padx=0, pady=0)

    # ── Translation Logic ────────────────────────────────────────────────────

    def translate(self):
        """Kick off a background translation thread."""
        if self.is_translating:
            return

        source_text = self.src_text.get("1.0", "end-1c").strip()
        if not source_text:
            messagebox.showwarning("Warning", "Please enter text to translate!")
            return

        threading.Thread(
            target=self._translate_thread, args=(source_text,), daemon=True
        ).start()

    def _translate_thread(self, source_text: str):
        """Background worker: calls the translation API and updates the UI."""
        self.is_translating = True
        self.root.after(
            0,
            lambda: self.translate_btn.configure(text="Loading...\n⏳", state="disabled"),
        )
        self.root.after(0, lambda: self.status_bar.configure(text="Translating…"))

        try:
            src_lang_name = self.src_lang.get()
            src_code = LANGUAGES.get(src_lang_name, "auto")

            tgt_lang_name = self.tgt_lang.get()
            tgt_code = LANGUAGES.get(tgt_lang_name, "en")

            # Auto-detect source language and display result
            if src_code == "auto":
                try:
                    detected = detect(source_text)
                    # Fixed f-string: closing brace is inside the string
                    self.root.after(
                        0,
                        lambda d=detected: self.detected_label.configure(
                            text=f"(Detected: {d})"
                        ),
                    )
                except Exception:
                    pass  # Detection failed silently; translation may still work

            translator = GoogleTranslator(source=src_code, target=tgt_code)
            translation = translator.translate(source_text)

            self.root.after(0, self._update_translation, translation)
            self.root.after(
                0,
                lambda: self.status_bar.configure(
                    text=f"✅ Translation complete: {src_lang_name} → {tgt_lang_name}"
                ),
            )

        except Exception as e:
            error_msg = str(e)
            self.root.after(
                0, lambda: messagebox.showerror("Translation Error", error_msg)
            )
            self.root.after(
                0,
                lambda: self.status_bar.configure(text=f"❌ Error: {error_msg}"),
            )
        finally:
            self.is_translating = False
            self.root.after(
                0,
                lambda: self.translate_btn.configure(
                    text="TRANSLATE\n➡️", state="normal"
                ),
            )

    def _update_translation(self, translation: str):
        """Write the translated text to the output box and refresh char count."""
        self.tgt_text.configure(state="normal")
        self.tgt_text.delete("1.0", "end")
        self.tgt_text.insert("1.0", translation)
        self.tgt_text.configure(state="disabled")
        self.tgt_char_count.configure(text=f"{len(translation)} chars")

    # ── UI Helpers ───────────────────────────────────────────────────────────

    def update_char_count(self, event=None):
        """Update the source character counter on each key press."""
        count = len(self.src_text.get("1.0", "end-1c"))
        self.src_char_count.configure(text=f"{count} chars")

    def speak_text(self, text: str):
        """Speak *text* in a background thread to avoid blocking the UI."""
        if text.strip():
            threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    def _speak(self, text: str):
        """TTS worker running on a background thread."""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception:
            pass  # Silently ignore TTS failures

    def copy_translation(self):
        """Copy the translated text to the system clipboard."""
        text = self.tgt_text.get("1.0", "end-1c")
        if text.strip():
            try:
                pyperclip.copy(text)
            except Exception:
                # Fall back to tkinter clipboard if pyperclip is unavailable
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
            self.status_bar.configure(text="✅ Translation copied to clipboard!")

    def clear_all(self):
        """Clear both text boxes and reset status labels."""
        self.src_text.delete("1.0", "end")
        self.tgt_text.configure(state="normal")
        self.tgt_text.delete("1.0", "end")
        self.tgt_text.configure(state="disabled")
        self.src_char_count.configure(text="0 chars")
        self.tgt_char_count.configure(text="0 chars")
        self.detected_label.configure(text="")
        self.status_bar.configure(text="Ready to translate.")

    def swap_languages(self):
        """Swap source ↔ target language selection and their text content."""
        src = self.src_lang.get()
        tgt = self.tgt_lang.get()
        if src == "Auto Detect":
            messagebox.showinfo(
                "Swap Unavailable",
                "Cannot swap while 'Auto Detect' is selected.\nPlease choose a specific source language first.",
            )
            return

        self.src_lang.set(tgt)
        self.tgt_lang.set(src)

        src_text = self.src_text.get("1.0", "end-1c")
        tgt_text = self.tgt_text.get("1.0", "end-1c")

        self.src_text.delete("1.0", "end")
        self.src_text.insert("1.0", tgt_text)

        self.tgt_text.configure(state="normal")
        self.tgt_text.delete("1.0", "end")
        self.tgt_text.insert("1.0", src_text)
        self.tgt_text.configure(state="disabled")

        self.update_char_count()

    def on_src_lang_change(self, value: str):
        """Clear the detected-language label when user picks an explicit language."""
        if value != "Auto Detect":
            self.detected_label.configure(text="")

    def toggle_theme(self):
        """Switch between dark and light appearance modes."""
        if self.theme_var.get() == "dark":
            ctk.set_appearance_mode("light")
            self.theme_var.set("light")
            self.theme_btn.configure(text="🌙 Dark Mode")
        else:
            ctk.set_appearance_mode("dark")
            self.theme_var.set("dark")
            self.theme_btn.configure(text="☀️ Light Mode")

    def run(self):
        """Start the Tkinter main event loop."""
        self.root.mainloop()


if __name__ == "__main__":
    app = TranslationApp()
    app.run()
