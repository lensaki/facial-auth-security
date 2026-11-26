# Webcam project

## Layout

- `models/`: YuNet face detector and SFace face recognizer ONNX models.
- `faces/`: enrolled face feature arrays (`*_face.npy`).
- Python scripts stay in this directory; paths are resolved relative to the script, so the project can be launched from any working directory.

## Install

From the repo root:

    pip install -r requirements.txt

## Which script to use

- `main_gui.py` is the primary app. It combines enrollment, recognition, and optional UDP event logging.
- `main_gui_account.py` is a WIP of the `main_gui.py` app, including a login for admin and also account creation.
- `enroll_face.py` is a small standalone enrollment demo. Its enrollment flow overlaps with the GUI.
- `face_recognition.py` is a standalone recognition window with manual UDP event logging. It overlaps with the GUI's recognition mode.
- `face_detection.py` is a separate detection-only demo that emits a face-present event; it does not identify people.
- `webcam_test.py` only checks that the webcam opens and displays frames. Keep it as a camera troubleshooting utility; it does not use either model.

For normal use, start `main_gui.py`; keep the other scripts when you want their narrower standalone behavior.

## Run

Run a script from this directory, for example:

```powershell
python main_gui.py
```

The GUI and standalone recognition script read enrollments from `faces/`. Both enrollment scripts save there. The YuNet and SFace models are loaded from `models/`.

### GUI (`main_gui.py`)

To register someone, type a name under **Register a new face**, click
**Start camera**, then **Save face**. To verify, click **Start scan**; the
banner shows *Access granted* or *Not recognized*. The collapsible
**Network log** under the video has **Log attempt** and auto-log; recognition
events are sent over UDP on port 5006.

### Standalone scripts

- Enroll a face (saves `faces/<Name>_face.npy`; asks before overwriting):

      python enroll_face.py <Name>

  Press S to enroll, Q to quit.

- Recognize enrolled faces: `python face_recognition.py`. Keys: A logs a
  recognition attempt for all visible faces, Q quits. Events are sent over UDP
  on port 5006.

- Detect faces only: `python face_detection.py`. Events are sent over UDP on
  port 5005.

The detection and recognition scripts use different ports, so they can run at
the same time.

Recognition uses cosine similarity with a threshold of 0.363 (the OpenCV SFace
reference value).
