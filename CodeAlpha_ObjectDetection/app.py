"""
CodeAlpha — Object Detection & Tracking
GUI Launcher (app.py)

Provides a polished CustomTkinter window for configuring and launching
the detection pipeline (detector.py) as a subprocess.
"""
import sys
import threading
import subprocess
from tkinter import filedialog, messagebox

import customtkinter as ctk


class DetectionApp:
    """Main GUI launcher for the CodeAlpha Object Detection system."""

    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("CodeAlpha — Object Detection & Tracking")
        self.root.geometry("520x640")
        self.root.resizable(False, False)

        # ── State variables ────────────────────────────────────────────────────
        self.source_var = ctk.StringVar(value="0")        # "0" = webcam
        self.model_var  = ctk.StringVar(value="yolov8n.pt")
        self.conf_var   = ctk.DoubleVar(value=0.4)
        self.proc       = None   # subprocess handle

        self._build_ui()

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        """Assemble all widgets."""

        # ── Header ────────────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self.root, fg_color="#0d1b2a", corner_radius=0, height=90)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        ctk.CTkLabel(
            hdr, text="🎯  Object Detection & Tracking",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#00d4ff"
        ).pack(pady=(18, 2))

        ctk.CTkLabel(
            hdr, text="YOLOv8  ·  SORT Tracker  ·  CodeAlpha Internship Task 4",
            font=ctk.CTkFont(size=11),
            text_color="#607d8b"
        ).pack()

        # ── Settings card ─────────────────────────────────────────────────────
        card = ctk.CTkFrame(self.root, fg_color="#131f2e", corner_radius=12)
        card.pack(fill="x", padx=20, pady=(18, 10))

        ctk.CTkLabel(card, text="⚙️  Configuration",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="#90caf9").pack(anchor="w", padx=18, pady=(14, 6))

        # Source row
        self._row(card, "Video Source")
        src_btns = ctk.CTkFrame(card, fg_color="transparent")
        src_btns.pack(fill="x", padx=30, pady=(0, 6))
        ctk.CTkRadioButton(src_btns, text="Webcam (index 0)",
                           variable=self.source_var, value="0").pack(side="left", padx=(0, 20))
        ctk.CTkRadioButton(src_btns, text="Video File",
                           variable=self.source_var, value="file").pack(side="left")

        # File picker row (shown when "Video File" selected)
        fp = ctk.CTkFrame(card, fg_color="transparent")
        fp.pack(fill="x", padx=30, pady=(0, 10))
        self.file_entry = ctk.CTkEntry(fp, placeholder_text="Browse for a video file …",
                                       width=310, height=32)
        self.file_entry.pack(side="left", padx=(0, 8))
        ctk.CTkButton(fp, text="Browse", width=90, height=32,
                      command=self._browse_file).pack(side="left")

        # Model row
        self._row(card, "YOLO Model")
        ctk.CTkComboBox(
            card,
            values=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
            variable=self.model_var,
            width=220, height=34,
            dropdown_fg_color="#1e2d3d"
        ).pack(anchor="w", padx=30, pady=(0, 10))

        # Confidence row
        self._row(card, "Confidence Threshold")
        conf_row = ctk.CTkFrame(card, fg_color="transparent")
        conf_row.pack(fill="x", padx=30, pady=(0, 16))
        self.conf_label = ctk.CTkLabel(conf_row, text="0.40", width=44,
                                       font=ctk.CTkFont(size=13, weight="bold"),
                                       text_color="#00d4ff")
        self.conf_label.pack(side="right")
        ctk.CTkSlider(
            conf_row, from_=0.1, to=0.9,
            variable=self.conf_var,
            command=self._update_conf_label,
            width=340, height=18
        ).pack(side="left")

        # ── Feature / controls info box ────────────────────────────────────────
        info = ctk.CTkFrame(self.root, fg_color="#131f2e", corner_radius=12)
        info.pack(fill="x", padx=20, pady=(0, 12))

        ctk.CTkLabel(
            info,
            text=(
                "📌  What you get:\n"
                "   • YOLOv8 detection on every frame\n"
                "   • SORT Kalman-filter object tracking with persistent IDs\n"
                "   • HUD overlay: FPS · frame count · per-class counts\n"
                "   • Corner-accent bounding boxes, colour-coded by track\n"
                "   • Auto CSV log saved to detection_logs/\n\n"
                "🎮  Keyboard controls (in the OpenCV window):\n"
                "   Q / Esc  →  Quit          S  →  Save screenshot\n"
                "   P        →  Pause / Resume"
            ),
            font=ctk.CTkFont(size=12),
            justify="left",
            text_color="#b0bec5"
        ).pack(padx=18, pady=14, anchor="w")

        # ── Start / Stop button ────────────────────────────────────────────────
        self.start_btn = ctk.CTkButton(
            self.root,
            text="🚀  Start Detection",
            font=ctk.CTkFont(size=17, weight="bold"),
            height=58,
            command=self._toggle_detection,
            fg_color="#00b4d8",
            hover_color="#0096b7",
            text_color="#000000",
            corner_radius=10
        )
        self.start_btn.pack(fill="x", padx=20, pady=(0, 6))

        # Status line
        self.status_lbl = ctk.CTkLabel(
            self.root,
            text="Ready — configure settings above then click Start.",
            font=ctk.CTkFont(size=11),
            text_color="#607d8b"
        )
        self.status_lbl.pack(pady=(0, 10))

    def _row(self, parent, label: str):
        """Render a section sub-label."""
        ctk.CTkLabel(parent, text=label,
                     font=ctk.CTkFont(size=12),
                     text_color="#78909c").pack(anchor="w", padx=18, pady=(4, 2))

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                ("All files",   "*.*")
            ]
        )
        if path:
            self.source_var.set("file")
            self.file_entry.delete(0, "end")
            self.file_entry.insert(0, path)

    def _update_conf_label(self, val):
        self.conf_label.configure(text=f"{float(val):.2f}")

    def _toggle_detection(self):
        """Start or stop the detection subprocess."""
        if self.proc and self.proc.poll() is None:
            # Already running → stop
            self.proc.terminate()
            self.proc = None
            self.start_btn.configure(text="🚀  Start Detection",
                                     fg_color="#00b4d8", hover_color="#0096b7")
            self._set_status("Detection stopped.", "#607d8b")
            return

        # Build source argument
        source = self.source_var.get()
        if source == "file":
            source = self.file_entry.get().strip()
            if not source:
                messagebox.showwarning(
                    "No file selected",
                    "Please browse for a video file or switch to Webcam."
                )
                return

        conf  = self.conf_var.get()
        model = self.model_var.get()

        cmd = [
            sys.executable, "detector.py",
            "--source", str(source),
            "--model",  model,
            "--conf",   f"{conf:.2f}"
        ]

        self._set_status(f"Launching detector … (model={model}, conf={conf:.2f})", "#ffb74d")
        self.start_btn.configure(text="⏹  Stop Detection",
                                 fg_color="#e53935", hover_color="#b71c1c")

        def _run():
            import os
            cwd = os.path.dirname(os.path.abspath(__file__))
            self.proc = subprocess.Popen(cmd, cwd=cwd)
            self.proc.wait()
            # Reset button after process exits naturally
            self.root.after(0, self._on_process_done)

        threading.Thread(target=_run, daemon=True).start()
        self.root.after(1500, lambda: self._set_status(
            "Detection window is open — use keyboard controls there.", "#4caf50"))

    def _on_process_done(self):
        """Called when the detector subprocess exits on its own."""
        self.proc = None
        self.start_btn.configure(text="🚀  Start Detection",
                                 fg_color="#00b4d8", hover_color="#0096b7")
        self._set_status("Detection finished.", "#607d8b")

    def _set_status(self, text: str, color: str = "#607d8b"):
        self.status_lbl.configure(text=text, text_color=color)

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    DetectionApp().run()
