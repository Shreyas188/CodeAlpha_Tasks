"""
chatbot.py - Main Application for CodeAlpha FAQ Chatbot (Task 2)
Beautiful dark chat UI built with CustomTkinter.
Author: CodeAlpha Internship Project
"""

import threading
import time
from datetime import datetime
from tkinter import messagebox

import customtkinter as ctk

from faqs import FAQS
from nlp_engine import NLPEngine


# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
BG_DARK      = "#0f0f23"
BG_SIDEBAR   = "#1a1a2e"
BG_HEADER    = "#16213e"
BG_INPUT_BAR = "#16213e"
ACCENT       = "#00d4ff"
ACCENT_HOVER = "#00a8cc"
USER_BUBBLE  = "#0f3460"
BOT_BUBBLE   = "#1e3a1e"
DANGER       = "#c62a2a"
DANGER_HOVER = "#e53935"
TEXT_DIM     = "#888888"
TEXT_MAIN    = "#ffffff"


# ---------------------------------------------------------------------------
# Category metadata
# ---------------------------------------------------------------------------
CATEGORY_EMOJIS = {
    "Python Basics":  "🐍",
    "OOP":            "🔷",
    "Machine Learning": "🤖",
    "Data Science":   "📊",
    "Git/GitHub":     "🐙",
    "General":        "💡",
}


class ChatbotApp:
    """Main application class for the CodeAlpha FAQ Chatbot."""

    def __init__(self):
        # Apply global theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Root window
        self.root = ctk.CTk()
        self.root.title("CodeAlpha — AI FAQ Chatbot  |  Task 2")
        self.root.geometry("1150x720")
        self.root.minsize(900, 600)
        self.root.resizable(True, True)

        # NLP engine (blocking init is fast enough for startup)
        self.nlp = NLPEngine()
        self.message_count = 0
        self.typing_frame = None   # reference so we can destroy it

        self._build_ui()
        self._send_bot_welcome()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Build the entire UI layout."""
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_area()
        self._build_status_bar()

    def _build_sidebar(self):
        """Left sidebar with categories and controls."""
        sidebar = ctk.CTkFrame(
            self.root, width=260,
            fg_color=BG_SIDEBAR, corner_radius=0
        )
        sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        sidebar.grid_propagate(False)

        # Logo / title
        ctk.CTkLabel(
            sidebar, text="🤖  FAQ Chatbot",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=ACCENT
        ).pack(pady=(22, 4), padx=16)

        ctk.CTkLabel(
            sidebar, text="CodeAlpha Internship · Task 2",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_DIM
        ).pack(pady=(0, 18), padx=16)

        # Divider
        ctk.CTkFrame(sidebar, height=1, fg_color="#333355").pack(fill="x", padx=16, pady=4)

        ctk.CTkLabel(
            sidebar, text="📚  Topics Available",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_MAIN
        ).pack(pady=(12, 8), padx=16, anchor="w")

        # One button per category
        categories = sorted(set(f['category'] for f in FAQS))
        for cat in categories:
            emoji = CATEGORY_EMOJIS.get(cat, "📌")
            count = sum(1 for f in FAQS if f['category'] == cat)
            ctk.CTkButton(
                sidebar,
                text=f"{emoji}  {cat}  ({count})",
                anchor="w",
                font=ctk.CTkFont(size=12),
                fg_color="transparent",
                hover_color="#16213e",
                text_color="#cccccc",
                command=lambda c=cat: self._ask_about_category(c)
            ).pack(fill="x", padx=12, pady=2)

        # Stats
        ctk.CTkLabel(
            sidebar,
            text=f"📈  Total FAQs: {len(FAQS)}",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_DIM
        ).pack(pady=(22, 6), padx=16, anchor="w")

        # Clear chat
        ctk.CTkButton(
            sidebar,
            text="🗑️  Clear Chat",
            command=self._clear_chat,
            fg_color=DANGER,
            hover_color=DANGER_HOVER,
            width=220, height=38,
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=(8, 20), padx=18)

    def _build_main_area(self):
        """Central chat area: header, messages scroll, input bar."""
        main = ctk.CTkFrame(self.root, fg_color=BG_DARK, corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)

        # --- Header ---
        header = ctk.CTkFrame(main, fg_color=BG_HEADER, height=62, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)

        ctk.CTkLabel(
            header,
            text="💬  Chat with AI Assistant",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT_MAIN
        ).pack(side="left", padx=20, pady=18)

        self.online_label = ctk.CTkLabel(
            header,
            text="●  Online",
            font=ctk.CTkFont(size=12),
            text_color="#00ff88"
        )
        self.online_label.pack(side="left", padx=4, pady=18)

        # --- Scrollable messages frame ---
        self.chat_frame = ctk.CTkScrollableFrame(
            main, fg_color=BG_DARK, label_text=""
        )
        self.chat_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)
        self.chat_frame.grid_columnconfigure(0, weight=1)

        # --- Input bar ---
        input_bar = ctk.CTkFrame(main, fg_color=BG_INPUT_BAR, height=80, corner_radius=0)
        input_bar.grid(row=2, column=0, sticky="ew")
        input_bar.grid_propagate(False)
        input_bar.grid_columnconfigure(0, weight=1)

        self.input_field = ctk.CTkEntry(
            input_bar,
            placeholder_text="Type your question here…  (Enter to send)",
            font=ctk.CTkFont(size=13),
            height=46,
            fg_color="#1a1a2e",
            border_color=ACCENT,
            border_width=2
        )
        self.input_field.grid(row=0, column=0, padx=(16, 6), pady=17, sticky="ew")
        self.input_field.bind("<Return>", lambda _e: self._on_send())

        ctk.CTkButton(
            input_bar,
            text="Send  ➤",
            width=110, height=46,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._on_send,
            fg_color=ACCENT,
            text_color="#000000",
            hover_color=ACCENT_HOVER
        ).grid(row=0, column=1, padx=(6, 16), pady=17)

    def _build_status_bar(self):
        """Thin status bar at the very bottom."""
        self.status_bar = ctk.CTkLabel(
            self.root,
            text="Ready  —  Ask me anything about Python, ML, or Data Science!",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_DIM,
            fg_color="#0a0a1a",
            height=24,
            anchor="w"
        )
        self.status_bar.grid(row=1, column=1, sticky="ew", padx=8)

    # ------------------------------------------------------------------
    # Message rendering helpers
    # ------------------------------------------------------------------

    def _add_user_bubble(self, text: str):
        """Render a user message bubble (right-aligned, blue)."""
        timestamp = datetime.now().strftime("%H:%M")

        container = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        container.pack(fill="x", pady=5)

        wrapper = ctk.CTkFrame(container, fg_color="transparent")
        wrapper.pack(side="right", padx=12)

        ctk.CTkLabel(
            wrapper,
            text=f"You  •  {timestamp}",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_DIM
        ).pack(anchor="e", pady=(0, 2))

        bubble = ctk.CTkFrame(wrapper, fg_color=USER_BUBBLE, corner_radius=16)
        bubble.pack(anchor="e")

        ctk.CTkLabel(
            bubble,
            text=text,
            font=ctk.CTkFont(size=13),
            wraplength=440,
            justify="left",
            text_color=TEXT_MAIN
        ).pack(padx=16, pady=10)

        self._scroll_bottom()

    def _add_bot_bubble(self, text: str, confidence: float = 1.0, category: str = ""):
        """Render a bot message bubble (left-aligned, dark green)."""
        timestamp = datetime.now().strftime("%H:%M")

        container = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        container.pack(fill="x", pady=5)

        wrapper = ctk.CTkFrame(container, fg_color="transparent")
        wrapper.pack(side="left", padx=12)

        # Avatar row
        ctk.CTkLabel(
            wrapper, text="🤖", font=ctk.CTkFont(size=22)
        ).pack(anchor="w")

        # Meta row (name, time, confidence)
        meta_row = ctk.CTkFrame(wrapper, fg_color="transparent")
        meta_row.pack(anchor="w", pady=(0, 2))

        ctk.CTkLabel(
            meta_row,
            text=f"AI Assistant  •  {timestamp}",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_DIM
        ).pack(side="left")

        if category and confidence < 1.0:
            ctk.CTkLabel(
                meta_row,
                text=f"   {category}  •  {confidence:.0%} match",
                font=ctk.CTkFont(size=10),
                text_color=ACCENT
            ).pack(side="left")

        # Bubble
        bubble = ctk.CTkFrame(wrapper, fg_color=BOT_BUBBLE, corner_radius=16)
        bubble.pack(anchor="w")

        ctk.CTkLabel(
            bubble,
            text=text,
            font=ctk.CTkFont(size=13),
            wraplength=520,
            justify="left",
            text_color=TEXT_MAIN
        ).pack(padx=16, pady=10)

        self._scroll_bottom()

    def _show_typing_indicator(self):
        """Show an animated 'AI is thinking…' indicator."""
        self.typing_frame = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        self.typing_frame.pack(fill="x", pady=5)

        wrapper = ctk.CTkFrame(self.typing_frame, fg_color="transparent")
        wrapper.pack(side="left", padx=12)

        bubble = ctk.CTkFrame(wrapper, fg_color="#1a1a2e", corner_radius=16)
        bubble.pack(anchor="w")

        self.typing_label = ctk.CTkLabel(
            bubble,
            text="🤖  AI is thinking…",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_DIM
        )
        self.typing_label.pack(padx=16, pady=10)
        self._scroll_bottom()

    def _remove_typing_indicator(self):
        """Remove the typing indicator frame."""
        if self.typing_frame and self.typing_frame.winfo_exists():
            self.typing_frame.destroy()
        self.typing_frame = None

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_send(self):
        """Handle send button / Enter key press."""
        text = self.input_field.get().strip()
        if not text:
            return

        self.input_field.delete(0, "end")
        self._add_user_bubble(text)
        self.message_count += 1
        self._show_typing_indicator()

        # Run NLP in background to keep UI responsive
        threading.Thread(
            target=self._fetch_response, args=(text,), daemon=True
        ).start()

    def _fetch_response(self, text: str):
        """Background thread: call NLP, then schedule UI update on main thread."""
        time.sleep(0.75)  # brief pause to simulate "thinking"
        response = self.nlp.get_response(text)

        self.root.after(0, self._remove_typing_indicator)
        self.root.after(
            0, self._add_bot_bubble,
            response['answer'],
            response['confidence'],
            response.get('category', '')
        )
        self.root.after(
            0, self.status_bar.configure,
            {
                "text": (
                    f"Category: {response.get('category', 'N/A')}  •  "
                    f"Confidence: {response['confidence']:.1%}  •  "
                    f"Messages: {self.message_count}"
                )
            }
        )

    def _ask_about_category(self, category: str):
        """Pre-fill the input with a category query and send."""
        self.input_field.delete(0, "end")
        self.input_field.insert(0, f"What is {category}?")
        self._on_send()

    def _clear_chat(self):
        """Clear all messages and restart with a welcome."""
        for widget in self.chat_frame.winfo_children():
            widget.destroy()
        self.message_count = 0
        self.typing_frame = None
        self._add_bot_bubble("Chat cleared! 🔄 How can I help you?", confidence=1.0)
        self.status_bar.configure(
            text="Ready  —  Ask me anything about Python, ML, or Data Science!"
        )

    def _send_bot_welcome(self):
        """Send the initial welcome message."""
        self._add_bot_bubble(
            "Hello! 👋 I'm your AI-powered FAQ Chatbot!\n\n"
            "I can answer questions about:\n"
            "  🐍 Python Basics & OOP\n"
            "  🤖 Machine Learning & Deep Learning\n"
            "  📊 Data Science (Pandas, NumPy, Matplotlib)\n"
            "  🐙 Git & GitHub\n"
            "  💡 General Programming Concepts\n\n"
            "Type a question or click a topic on the left to get started!",
            confidence=1.0
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _scroll_bottom(self):
        """Scroll the chat frame to the latest message."""
        self.root.after(
            120, lambda: self.chat_frame._parent_canvas.yview_moveto(1.0)
        )

    def run(self):
        """Start the Tkinter main loop."""
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = ChatbotApp()
    app.run()
