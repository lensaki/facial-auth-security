"""Standalone Windows face enrollment/recognition GUI with UDP Wireshark events.
Requires opencv-contrib-python, numpy, Pillow, yunet.onnx, sface.onnx.
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
THRESHOLD = 0.30
UDP_ADDRESS = ("127.0.0.1", 5005)


class FaceGUI:
    def __init__(self, root):
        self.root = root
        root.title("Facial Authentication & Network Logging")
        root.geometry("1100x790")
        root.minsize(860, 650)
        self.detector = cv2.FaceDetectorYN.create(str(BASE / "yunet.onnx"), "", (320, 320), 0.9)
        self.recognizer = cv2.FaceRecognizerSF.create(str(BASE / "sface.onnx"), "")
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
            messagebox.showwarning("UDP listener", f"Could not bind UDP port 5005: {exc}\nOutgoing UDP logging will still work if another listener is using the port.")

        top = ttk.Frame(root, padding=12)
        top.pack(fill="x")
        ttk.Label(top, text="Facial Authentication System", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(top, text="Local webcam  •  YuNet / SFace  •  UDP 127.0.0.1:5005").pack(anchor="w")

        controls = ttk.Frame(root, padding=(12, 0, 12, 10))
        controls.pack(fill="x")
        ttk.Label(controls, text="Person's name:").pack(side="left")
        self.name = tk.StringVar()
        ttk.Entry(controls, textvariable=self.name, width=18).pack(side="left", padx=6)
        ttk.Button(controls, text="Enroll Mode", command=lambda: self.start("enroll")).pack(side="left", padx=3)
        ttk.Button(controls, text="Save Face", command=self.save_face).pack(side="left", padx=3)
        ttk.Button(controls, text="Recognition Mode", command=lambda: self.start("recognize")).pack(side="left", padx=3)
        ttk.Button(controls, text="Log Attempt", command=self.log_attempt).pack(side="left", padx=3)
        ttk.Button(controls, text="Stop Camera", command=self.stop).pack(side="left", padx=3)

        options = ttk.Frame(root, padding=(12, 0, 12, 10))
        options.pack(fill="x")
        self.auto_log = tk.BooleanVar(value=False)
        ttk.Checkbutton(options, text="Automatically log visible faces every 2 seconds", variable=self.auto_log,
                        command=lambda: setattr(self, "last_auto_log", 0.0)).pack(side="left")
        self.status = tk.StringVar(value="Ready. Select Enroll Mode or Recognition Mode.")
        ttk.Label(options, textvariable=self.status).pack(side="right")

        self.video = ttk.Label(root, anchor="center", text="Webcam preview", background="#18212c", foreground="white")
        self.video.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        bottom = ttk.Frame(root, padding=(12, 0, 12, 12))
        bottom.pack(fill="x")
        self.people = tk.StringVar()
        self.refresh_people()
        ttk.Label(bottom, textvariable=self.people).pack(anchor="w")
        ttk.Label(bottom, text="Recent events:").pack(anchor="w")
        self.event_box = tk.Text(bottom, height=7, state="disabled", font=("Consolas", 9))
        self.event_box.pack(fill="x")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(30, self.tick)

    def load_faces(self):
        self.enrolled.clear()
        for path in sorted(BASE.glob("*_face.npy")):
            name = path.stem.removesuffix("_face").replace("_", " ").title()
            self.enrolled[name] = np.load(path, allow_pickle=False)

    def refresh_people(self):
        self.people.set("Enrolled: " + (", ".join(self.enrolled) or "none"))

    def append_event(self, message):
        self.event_box.configure(state="normal")
        self.event_box.insert("end", message + "\n")
        self.event_box.see("end")
        self.event_box.configure(state="disabled")

    def start(self, mode):
        if mode == "recognize" and not self.enrolled:
            messagebox.showinfo("No faces", "Please enroll at least one face first.")
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
        self.status.set("Enrollment: show exactly one face" if mode == "enroll" else "Recognition running")

    def stop(self):
        self.mode = "stopped"
        self.frame = None
        self.faces = None
        self.results = []
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video.configure(image="", text="Webcam stopped")
        self.video.image = None
        self.status.set("Camera stopped")

    def save_face(self):
        if self.mode != "enroll" or self.frame is None or self.faces is None or len(self.faces) != 1:
            messagebox.showinfo("Enrollment", "Select Enroll Mode and show exactly one face before saving.")
            return
        name = self.name.get().strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,49}", name):
            messagebox.showwarning("Name", "Enter a name using letters, numbers, spaces, _ or - (max 50 chars).")
            return
        filename = re.sub(r"[ -]+", "_", name).strip("_") + "_face.npy"
        target = BASE / filename
        if target.exists() and not messagebox.askyesno("Overwrite", f"Replace enrollment for {name}?"):
            return
        try:
            aligned = self.recognizer.alignCrop(self.frame, self.faces[0])
            features = self.recognizer.feature(aligned)
            np.save(target, features)
            self.load_faces()
            self.refresh_people()
            self.append_event(f"Enrolled {name}: {target.name}")
            self.status.set(f"Enrolled {name}")
        except cv2.error as exc:
            messagebox.showerror("Enrollment error", str(exc))

    def log_attempt(self):
        if self.mode != "recognize" or not self.results:
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
        self.status.set(f"Logged attempt #{self.attempt_id} via UDP :5005")

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
                                label, color = "Face detected", (0, 255, 0)
                            else:
                                aligned = self.recognizer.alignCrop(frame, face)
                                features = self.recognizer.feature(aligned)
                                scores = {name: float(self.recognizer.match(ref, features,
                                    cv2.FaceRecognizerSF_FR_COSINE)) for name, ref in self.enrolled.items()}
                                best_name = max(scores, key=scores.get)
                                score = scores[best_name]
                                recognized = score >= THRESHOLD
                                identity = best_name if recognized else None
                                label = f"{identity or 'Unknown'} | {score:.3f}"
                                color = (0, 255, 0) if recognized else (0, 0, 255)
                                self.results.append({"face_index": index,
                                                     "result": "recognized" if recognized else "unknown",
                                                     "identity": identity, "similarity": round(score, 4),
                                                     "bbox": [x, y, fw, fh]})
                            cv2.rectangle(preview, (x, y), (x + fw, y + fh), color, 2)
                            cv2.putText(preview, label, (max(0, x), max(20, y - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                    if self.mode == "recognize" and self.auto_log.get() and self.results:
                        now = time.monotonic()
                        if now - self.last_auto_log >= 2.0:
                            self.last_auto_log = now
                            self.log_attempt()
                    # Display BGR OpenCV frames in Tkinter via Pillow.
                    rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    available_w = max(320, self.video.winfo_width() - 12)
                    available_h = max(240, self.video.winfo_height() - 12)
                    img.thumbnail((available_w, available_h), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self.video.configure(image=photo, text="")
                    self.video.image = photo
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
