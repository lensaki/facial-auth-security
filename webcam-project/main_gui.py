"""Standalone Windows face enrollment/recognition GUI with UDP Wireshark events.
Requires opencv-python (includes FaceDetectorYN/FaceRecognizerSF), numpy, Pillow, yunet.onnx, sface.onnx.
"""
import json
import re
import socket
import time
import uuid
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

BASE = Path(__file__).resolve().parent
# Models live in models/ and enrolled faces in faces/ (shared with face_recognition.py).
MODELS = BASE / "models"
FACES = BASE / "faces"
THRESHOLD = 0.363  # OpenCV's SFace reference cosine threshold (matches webcam-project/face_recognition.py)
UDP_PORT = 5006  # recognition events; face_detection.py uses 5005
UDP_ADDRESS = ("127.0.0.1", UDP_PORT)

# ---- Theme -----------------------------------------------------------------
FONT = "Segoe UI"
MONO = "Consolas"
BG = "#0f1419"          # window background
SURFACE = "#171d26"     # cards
SURFACE_2 = "#1f2733"   # inputs, hover surfaces
BORDER = "#2a3342"
TEXT = "#e6eaf0"
MUTED = "#8b96a8"
ACCENT = "#4f8cff"
ACCENT_HOVER = "#6a9fff"
ACCENT_PRESSED = "#3a74e0"
SUCCESS = "#2ecc71"
SUCCESS_BG = "#12301f"
DANGER = "#ff5c5c"
DANGER_BG = "#3a1818"
WARN = "#f5b942"
NEUTRAL_BG = "#222b38"
VIDEO_BG = "#0a0d11"
SIDE_WIDTH = 360
PAD_S, PAD_M, PAD_L = 8, 12, 16
NAME_HINT = "Enter a name"


def bgr(hex_color):
    """Convert '#rrggbb' to an OpenCV BGR tuple."""
    h = hex_color.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


class FaceGUI:
    def __init__(self, root):
        self.root = root
        root.title("FaceAuth — Facial Authentication")
        root.geometry("1200x760")
        root.minsize(1000, 700)
        root.configure(bg=BG)
        for model in ("yunet.onnx", "sface.onnx"):
            if not (MODELS / model).is_file():
                raise FileNotFoundError(f"Missing {model}; looked in folder: {MODELS}")
        self.detector = cv2.FaceDetectorYN.create(str(MODELS / "yunet.onnx"), "", (320, 320), 0.9)
        self.recognizer = cv2.FaceRecognizerSF.create(str(MODELS / "sface.onnx"), "")
        self.enrolled = {}
        self.load_faces()
        self.cap = None
        self.mode = "stopped"
        self.frame = None
        self.faces = None
        self.results = []
        self.session_id = str(uuid.uuid4())
        self.attempt_id = 0
        self.last_auto_log = 0.0
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver = None
        try:
            self.receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.receiver.bind(UDP_ADDRESS)
            self.receiver.setblocking(False)
        except OSError as exc:
            if self.receiver:
                self.receiver.close()
            self.receiver = None
            messagebox.showwarning("UDP listener", f"Could not bind UDP port {UDP_PORT}: {exc}\nOutgoing UDP logging will still work if another listener is using the port.")

        self.setup_style()
        self.build_ui()
        self.refresh_people()
        self.update_state()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(30, self.tick)

    # ---- Styling -----------------------------------------------------------
    def setup_style(self):
        s = ttk.Style(self.root)
        s.theme_use("clam")
        s.configure(".", background=SURFACE, foreground=TEXT, font=(FONT, 10), bordercolor=BORDER,
                    focuscolor=SURFACE, lightcolor=SURFACE, darkcolor=SURFACE)
        s.configure("TFrame", background=SURFACE)
        s.configure("TLabel", background=SURFACE, foreground=TEXT)
        for name, bg_color, fg_color, border in (
                ("Primary", ACCENT, "#ffffff", ACCENT),
                ("Secondary", SURFACE_2, TEXT, "#3a4558")):
            style = f"{name}.TButton"
            s.configure(style, background=bg_color, foreground=fg_color, bordercolor=border,
                        lightcolor=bg_color, darkcolor=bg_color, focuscolor=bg_color,
                        font=(FONT, 10, "bold"), padding=(PAD_L, 7), relief="flat", borderwidth=1)
        s.map("Primary.TButton",
              background=[("pressed", ACCENT_PRESSED), ("active", ACCENT_HOVER), ("disabled", SURFACE_2)],
              bordercolor=[("pressed", ACCENT_PRESSED), ("active", ACCENT_HOVER)],
              lightcolor=[("pressed", ACCENT_PRESSED), ("active", ACCENT_HOVER)],
              darkcolor=[("pressed", ACCENT_PRESSED), ("active", ACCENT_HOVER)],
              foreground=[("disabled", MUTED)])
        s.map("Secondary.TButton",
              background=[("pressed", BORDER), ("active", "#2a3444")],
              bordercolor=[("pressed", MUTED), ("active", MUTED)],
              lightcolor=[("pressed", BORDER), ("active", "#2a3444")],
              darkcolor=[("pressed", BORDER), ("active", "#2a3444")])
        s.configure("Link.TButton", background=SURFACE, foreground=TEXT, bordercolor=SURFACE,
                    lightcolor=SURFACE, darkcolor=SURFACE, focuscolor=SURFACE, relief="flat",
                    font=(FONT, 11, "bold"), padding=(0, 2), anchor="w")
        s.map("Link.TButton", background=[("active", SURFACE)], foreground=[("active", ACCENT_HOVER)])
        s.configure("TEntry", fieldbackground=SURFACE_2, foreground=TEXT, insertcolor=TEXT,
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, padding=8)
        s.map("TEntry", bordercolor=[("focus", ACCENT)], lightcolor=[("focus", ACCENT)],
              darkcolor=[("focus", ACCENT)])
        s.configure("TCheckbutton", background=SURFACE, foreground=TEXT, focuscolor=SURFACE,
                    indicatorbackground=SURFACE_2, indicatorforeground=ACCENT, bordercolor=BORDER,
                    font=(FONT, 10))
        s.map("TCheckbutton", background=[("active", SURFACE)],
              indicatorbackground=[("selected", ACCENT), ("active", SURFACE_2)],
              indicatorforeground=[("selected", "#ffffff")])
        s.configure("Vertical.TScrollbar", background=SURFACE_2, troughcolor=SURFACE, bordercolor=SURFACE,
                    arrowcolor=MUTED, lightcolor=SURFACE_2, darkcolor=SURFACE_2)

    def card(self, parent, title, **pack):
        """A bordered surface with a title row; returns (outer, body, header_row)."""
        outer = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        outer.pack(fill="x", pady=(0, PAD_M), **pack)
        header = tk.Frame(outer, bg=SURFACE)
        header.pack(fill="x", padx=PAD_L, pady=(PAD_M, PAD_S))
        tk.Label(header, text=title, bg=SURFACE, fg=TEXT, font=(FONT, 11, "bold")).pack(side="left")
        body = tk.Frame(outer, bg=SURFACE)
        body.pack(fill="x", padx=PAD_L, pady=(0, PAD_M))
        return outer, body, header

    def note(self, parent, text="", fg=MUTED, **kw):
        return tk.Label(parent, text=text, bg=SURFACE, fg=fg, font=(FONT, 9), justify="left",
                        anchor="w", wraplength=SIDE_WIDTH - 2 * PAD_L - 2, **kw)

    # ---- Layout ------------------------------------------------------------
    def build_ui(self):
        root = self.root
        # Header bar
        header = tk.Frame(root, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        header.pack(fill="x")
        inner = tk.Frame(header, bg=SURFACE)
        inner.pack(fill="x", padx=PAD_L + 4, pady=PAD_M)
        tk.Label(inner, text="◉", bg=SURFACE, fg=ACCENT, font=(FONT, 18)).pack(side="left")
        tk.Label(inner, text="FaceAuth", bg=SURFACE, fg=TEXT, font=(FONT, 16, "bold")).pack(side="left", padx=(PAD_S, PAD_M))
        tk.Label(inner, text="Facial authentication", bg=SURFACE, fg=MUTED, font=(FONT, 10)).pack(side="left", pady=(4, 0))
        self.pill = tk.Label(inner, text="Camera off", font=(FONT, 9, "bold"), padx=14, pady=4)
        self.pill.pack(side="right")

        main = tk.Frame(root, bg=BG)
        main.pack(fill="both", expand=True, padx=PAD_L, pady=PAD_L)

        # Side panel (fixed width)
        side = tk.Frame(main, bg=BG, width=SIDE_WIDTH)
        side.pack(side="right", fill="y", padx=(PAD_L, 0))
        side.pack_propagate(False)

        # Video card
        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        video_card = tk.Frame(left, bg=VIDEO_BG, highlightbackground=BORDER, highlightthickness=1)
        video_card.pack(side="top", fill="both", expand=True)
        footer = tk.Frame(video_card, bg=SURFACE)
        footer.pack(side="bottom", fill="x")
        self.status = tk.StringVar(value="Ready")
        tk.Label(footer, textvariable=self.status, bg=SURFACE, fg=MUTED, font=(FONT, 9), anchor="w"
                 ).pack(fill="x", padx=PAD_L, pady=PAD_S)
        holder = tk.Frame(video_card, bg=VIDEO_BG)
        holder.pack(fill="both", expand=True)
        self.video = tk.Label(holder, bg=VIDEO_BG, fg=MUTED, font=(FONT, 13), justify="center",
                              text="\U0001F4F7\n\nCamera is off\nStart a scan or register a face to begin")
        self.video.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.build_verify(side)
        self.build_register(side)
        self.build_log(left)   # collapsible strip under the video
        self.build_users(side)

    def build_verify(self, side):
        _, body, _ = self.card(side, "Verify identity")
        row = tk.Frame(body, bg=SURFACE)
        row.pack(fill="x")
        ttk.Button(row, text="Start scan", style="Primary.TButton",
                   command=lambda: self.start("recognize")).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Stop camera", style="Secondary.TButton",
                   command=self.stop).pack(side="left", padx=(PAD_S, 0))
        self.banner = tk.Label(body, text="", font=(FONT, 12, "bold"), padx=PAD_M, pady=PAD_M,
                               wraplength=SIDE_WIDTH - 2 * PAD_L - 30, justify="center")
        self.banner.pack(fill="x", pady=(PAD_M, 0))
        self.verify_hint = self.note(body, "")
        self.verify_hint.pack_opts = dict(fill="x", pady=(PAD_S, 0))

    def build_register(self, side):
        _, body, _ = self.card(side, "Register a new face")
        self.name = tk.StringVar()
        self.name_entry = ttk.Entry(body, textvariable=self.name, font=(FONT, 11))
        self.name_entry.pack(fill="x")
        self.name_placeholder = False
        self.name_entry.bind("<FocusIn>", self._name_focus_in)
        self.name_entry.bind("<FocusOut>", self._name_focus_out)
        self._name_focus_out()
        self.note(body, "Look at the camera and make sure only your face is visible."
                  ).pack(fill="x", pady=(PAD_S, PAD_S))
        row = tk.Frame(body, bg=SURFACE)
        row.pack(fill="x")
        ttk.Button(row, text="Start camera", style="Secondary.TButton",
                   command=lambda: self.start("enroll")).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Save face", style="Primary.TButton",
                   command=self.save_face).pack(side="left", fill="x", expand=True, padx=(PAD_S, 0))
        self.register_msg = self.note(body, "")
        self.register_msg.pack_opts = dict(fill="x", pady=(PAD_S, 0))

    def build_users(self, side):
        outer, body, header = self.card(side, "Registered users")
        outer.pack_configure(fill="both", expand=True)
        body.pack_configure(fill="both", expand=True)
        self.user_count = tk.Label(header, text="0", bg=SURFACE_2, fg=MUTED, font=(FONT, 9, "bold"), padx=8)
        self.user_count.pack(side="right")
        self.user_list = tk.Listbox(body, height=3, bg=SURFACE_2, fg=TEXT, font=(FONT, 10), relief="flat",
                                    highlightthickness=1, highlightbackground=BORDER, highlightcolor=BORDER,
                                    selectbackground=ACCENT, selectforeground="#ffffff", activestyle="none",
                                    borderwidth=0)
        self.user_list.pack(fill="both", expand=True)

    def build_log(self, side):
        outer = tk.Frame(side, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        outer.pack(side="bottom", fill="x", pady=(PAD_M, 0))
        self.log_toggle = ttk.Button(outer, text="▸  Network log", style="Link.TButton", command=self.toggle_log)
        self.log_toggle.pack(fill="x", padx=PAD_L, pady=PAD_S + 2)
        self.log_body = tk.Frame(outer, bg=SURFACE)
        pad = dict(padx=PAD_L)
        top = tk.Frame(self.log_body, bg=SURFACE)
        top.pack(fill="x", **pad)
        ttk.Button(top, text="Log attempt", style="Secondary.TButton",
                   command=self.log_attempt).pack(side="left")
        self.auto_log = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Auto-log visible faces every 2 seconds", variable=self.auto_log,
                        command=lambda: setattr(self, "last_auto_log", 0.0)).pack(side="left", padx=PAD_M)
        self.note(top, f"UDP → 127.0.0.1:{UDP_PORT}").pack(side="right")
        box = tk.Frame(self.log_body, bg=BORDER)
        box.pack(fill="x", padx=PAD_L, pady=(PAD_S, PAD_M))
        self.log_msg = self.note(self.log_body, "")
        self.log_msg.pack_opts = dict(fill="x", pady=(PAD_S, 0), before=box, **pad)
        self.event_box = tk.Text(box, height=5, state="disabled", font=(MONO, 9), bg=VIDEO_BG, fg="#9fe3b5",
                                 relief="flat", borderwidth=0, padx=8, pady=6, wrap="word",
                                 insertbackground=TEXT, highlightthickness=0)
        scroll = ttk.Scrollbar(box, orient="vertical", command=self.event_box.yview)
        self.event_box.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y", padx=(0, 1), pady=1)
        self.event_box.pack(side="left", fill="x", expand=True, padx=(1, 0), pady=1)
        self.log_open = False

    def toggle_log(self):
        self.log_open = not self.log_open
        if self.log_open:
            self.log_body.pack(fill="x")
            self.log_toggle.configure(text="▾  Network log")
        else:
            self.log_body.pack_forget()
            self.log_toggle.configure(text="▸  Network log")

    # ---- Placeholder entry -------------------------------------------------
    def _name_focus_in(self, _event=None):
        if self.name_placeholder:
            self.name_placeholder = False
            self.name_entry.delete(0, "end")
            self.name_entry.configure(foreground=TEXT)

    def _name_focus_out(self, _event=None):
        if not self.name.get().strip():
            self.name_entry.delete(0, "end")
            self.name_entry.insert(0, NAME_HINT)
            self.name_entry.configure(foreground=MUTED)
            self.name_placeholder = True

    def get_name(self):
        return "" if self.name_placeholder else self.name.get().strip()

    # ---- State display -----------------------------------------------------
    def show_msg(self, label, text="", fg=MUTED):
        """Show inline text in a note label; hide the label entirely when empty."""
        if text:
            label.configure(text=text, fg=fg)
            if not label.winfo_ismapped():
                label.pack(**label.pack_opts)
        else:
            label.pack_forget()

    def set_pill(self, text, fg, bg):
        self.pill.configure(text=text, fg=fg, bg=bg)

    def set_banner(self, text, fg, bg):
        self.banner.configure(text=text, fg=fg, bg=bg)

    def update_state(self):
        """Refresh the status pill and the result banner from self.mode / self.results."""
        if self.mode == "stopped":
            self.set_pill("Camera off", MUTED, NEUTRAL_BG)
            self.set_banner("Ready to verify", MUTED, NEUTRAL_BG)
            return
        if self.mode == "enroll":
            self.set_pill("● Enrolling", WARN, NEUTRAL_BG)
            self.set_banner("Enrollment in progress", MUTED, NEUTRAL_BG)
            return
        self.set_pill("● Scanning", SUCCESS, SUCCESS_BG)
        results = self.results
        if not results:
            self.set_banner("No face detected", MUTED, NEUTRAL_BG)
        elif len(results) > 1:
            known = sum(1 for r in results if r["result"] == "recognized")
            self.set_banner(f"Multiple faces ({len(results)}) — {known} recognized", WARN, NEUTRAL_BG)
        elif results[0]["result"] == "recognized":
            r = results[0]
            self.set_banner(f"✓ Access granted — {r['identity']} ({r['similarity']:.2f})", SUCCESS, SUCCESS_BG)
        else:
            self.set_banner(f"✗ Not recognized ({results[0]['similarity']:.2f})", DANGER, DANGER_BG)

    def load_faces(self):
        self.enrolled.clear()
        for path in sorted(FACES.glob("*_face.npy")):
            name = path.stem.removesuffix("_face").replace("_", " ").title()
            self.enrolled[name] = np.load(path, allow_pickle=False)

    def refresh_people(self):
        self.user_list.delete(0, "end")
        for name in self.enrolled:
            self.user_list.insert("end", "  " + name)
        self.user_count.configure(text=str(len(self.enrolled)))

    def append_event(self, message):
        self.event_box.configure(state="normal")
        self.event_box.insert("end", message + "\n")
        self.event_box.see("end")
        self.event_box.configure(state="disabled")

    # ---- Camera control ----------------------------------------------------
    def start(self, mode):
        self.show_msg(self.verify_hint)
        self.show_msg(self.register_msg)
        if mode == "recognize" and not self.enrolled:
            self.show_msg(self.verify_hint, "No registered faces yet. Register a face first.", WARN)
            return
        if self.cap is None:
            # CAP_DSHOW avoids some Windows camera-backend issues; fall back to default.
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = None
                messagebox.showerror("Webcam", "Cannot access webcam. Close other apps that may be using it.")
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.mode = mode
        self.last_auto_log = 0.0
        self.results = []
        self.status.set("Enrollment: show exactly one face" if mode == "enroll" else "Scanning for faces")
        self.update_state()

    def stop(self):
        self.mode = "stopped"
        self.frame = None
        self.faces = None
        self.results = []
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video.configure(image="", text="\U0001F4F7\n\nCamera is off\nStart a scan or register a face to begin")
        self.video.image = None
        self.status.set("Camera stopped")
        self.update_state()

    def save_face(self):
        self.show_msg(self.register_msg)
        if self.mode != "enroll" or self.frame is None or self.faces is None or len(self.faces) != 1:
            self.show_msg(self.register_msg, "Start the camera and show exactly one face before saving.", WARN)
            return
        name = self.get_name()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,49}", name):
            messagebox.showwarning("Name", "Enter a name using letters, numbers, spaces, _ or - (max 50 chars).")
            return
        filename = re.sub(r"[ -]+", "_", name).strip("_") + "_face.npy"
        target = FACES / filename
        if target.exists() and not messagebox.askyesno("Overwrite", f"Replace enrollment for {name}?"):
            return
        try:
            aligned = self.recognizer.alignCrop(self.frame, self.faces[0])
            features = self.recognizer.feature(aligned)
            FACES.mkdir(parents=True, exist_ok=True)
            np.save(target, features)
            self.load_faces()
            self.refresh_people()
            self.append_event(f"Enrolled {name}: {target.name}")
            self.status.set(f"Enrolled {name}")
            self.show_msg(self.register_msg, f"✓ Saved {name}.", SUCCESS)
        except cv2.error as exc:
            messagebox.showerror("Enrollment error", str(exc))

    def log_attempt(self):
        if self.mode != "recognize" or not self.results:
            self.show_msg(self.log_msg, "No visible faces to log. Start a scan first.", WARN)
            self.status.set("No visible recognized/unknown faces to log")
            return
        self.attempt_id += 1
        timestamp = datetime.now().astimezone().isoformat()
        for result in self.results:
            event = {
                "session_id": self.session_id,
                "attempt_id": self.attempt_id,
                "timestamp": timestamp,
                "event": "recognition_attempt",
                "face_count": len(self.results),
                "threshold": THRESHOLD,
                **result,
            }
            payload = json.dumps(event, separators=(",", ":"))
            try:
                self.sender.sendto(payload.encode("utf-8"), UDP_ADDRESS)
                self.append_event(f"#{self.attempt_id} {result['result']} "
                                  f"{result['identity'] or 'Unknown'} score={result['similarity']:.3f}")
            except OSError as exc:
                self.append_event(f"UDP send failed: {exc}")
        # Drain packets so loopback listener does not accumulate unread datagrams.
        if self.receiver:
            while True:
                try:
                    self.receiver.recvfrom(65535)
                except (BlockingIOError, OSError):
                    break
        self.show_msg(self.log_msg, f"Logged attempt #{self.attempt_id}", SUCCESS)
        self.status.set(f"Logged attempt #{self.attempt_id} via UDP :{UDP_PORT}")

    # ---- Video -------------------------------------------------------------
    @staticmethod
    def draw_face(img, x, y, w, h, label, color):
        """Corner-bracket box with a filled label strip above it."""
        t = max(2, img.shape[1] // 320)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)
        c = max(12, min(w, h) // 4)
        for px, py, dx, dy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
            cv2.line(img, (px, py), (px + dx * c, py), color, t + 2, cv2.LINE_AA)
            cv2.line(img, (px, py), (px, py + dy * c), color, t + 2, cv2.LINE_AA)
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.6
        (tw, th), base = cv2.getTextSize(label, font, scale, 2)
        pad = 6
        top = y - th - base - 2 * pad
        if top < 0:
            top = y + 2
        cv2.rectangle(img, (x, top), (x + tw + 2 * pad, top + th + base + 2 * pad), color, -1)
        cv2.putText(img, label, (x + pad, top + pad + th), font, scale, bgr(VIDEO_BG), 2, cv2.LINE_AA)

    def show_frame(self, preview):
        """Display a BGR OpenCV frame in the video card, scaled to fit."""
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        available_w = max(320, self.video.winfo_width() - 8)
        available_h = max(240, self.video.winfo_height() - 8)
        img.thumbnail((available_w, available_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.video.configure(image=photo, text="")
        self.video.image = photo

    def tick(self):
        try:
            if self.cap is not None and self.mode != "stopped":
                ok, frame = self.cap.read()
                if ok:
                    self.frame = frame
                    h, w = frame.shape[:2]
                    self.detector.setInputSize((w, h))
                    _, self.faces = self.detector.detect(frame)
                    preview = frame.copy()
                    self.results = []
                    if self.faces is not None:
                        for index, face in enumerate(self.faces, start=1):
                            x, y, fw, fh = map(int, face[:4])
                            if self.mode == "enroll":
                                label, color = "Face detected", bgr(SUCCESS)
                            else:
                                aligned = self.recognizer.alignCrop(frame, face)
                                features = self.recognizer.feature(aligned)
                                scores = {name: float(self.recognizer.match(ref, features,
                                    cv2.FaceRecognizerSF_FR_COSINE)) for name, ref in self.enrolled.items()}
                                best_name = max(scores, key=scores.get)
                                score = scores[best_name]
                                recognized = score >= THRESHOLD
                                identity = best_name if recognized else None
                                label = f"{identity or 'Unknown'}  {score:.2f}"
                                color = bgr(SUCCESS) if recognized else bgr(DANGER)
                                self.results.append({"face_index": index,
                                                     "result": "recognized" if recognized else "unknown",
                                                     "identity": identity, "similarity": round(score, 4),
                                                     "bbox": [x, y, fw, fh]})
                            self.draw_face(preview, x, y, fw, fh, label, color)
                    if self.mode == "recognize" and self.auto_log.get() and self.results:
                        now = time.monotonic()
                        if now - self.last_auto_log >= 2.0:
                            self.last_auto_log = now
                            self.log_attempt()
                    self.update_state()
                    if self.mode == "enroll":
                        count = 0 if self.faces is None else len(self.faces)
                        self.set_banner("Face ready to save" if count == 1 else
                                        ("No face detected" if count == 0 else f"Multiple faces ({count})"),
                                        SUCCESS if count == 1 else MUTED,
                                        SUCCESS_BG if count == 1 else NEUTRAL_BG)
                    self.show_frame(preview)
                else:
                    self.status.set("Could not read webcam frame")
        except (cv2.error, ValueError) as exc:
            self.status.set(f"Processing error: {exc}")
        self.root.after(30, self.tick)

    def close(self):
        self.stop()
        self.sender.close()
        if self.receiver:
            self.receiver.close()
        self.root.destroy()


if __name__ == "__main__":
    app = tk.Tk()
    try:
        FaceGUI(app)
        app.mainloop()
    except Exception as exc:
        messagebox.showerror("Startup error", str(exc))
        app.destroy()
