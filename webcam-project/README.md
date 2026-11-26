current files that i totally made here

# Webcam project

## Layout

- `models/`: YuNet face detector and SFace face recognizer ONNX models.
- `faces/`: enrolled face feature arrays (`*_face.npy`).
- Python scripts stay in this directory; paths are resolved relative to the script, so the project can be launched from any working directory.

## Which script to use

- `main_gui.py` is the primary app. It combines enrollment, recognition, and optional UDP event logging.
- `main_gui_account.py` is a WIP of the `main_gui.py` app, including a login for admin and also account creation. 
- `enroll_face.py` is a small standalone enrollment demo. Its enrollment flow overlaps with the GUI.
- `face_recognition.py` is a standalone recognition window with manual UDP event logging. It overlaps with the GUI's recognition mode.
- `face_detection.py` is a separate detection-only demo that emits a face-present event; it does not identify people.
- `webcam_test.py` only checks that the webcam opens and displays frames. Keep it as a camera troubleshooting utility; it does not use either model.

The repository history uses generic upload commits and does not reliably establish which script is oldest. For normal use, start `main_gui.py`; keep the other scripts when you want their narrower standalone behavior.

## Run

Run a script from this directory, for example:

```powershell
python main_gui.py
```

The GUI and standalone recognition script read enrollments from `faces/`. Both enrollment scripts save there. The YuNet and SFace models are loaded from `models/`.
