import re
import sys
import cv2
import numpy as np
from pathlib import Path

folder = Path(__file__).resolve().parent

# Name comes from the first argument, else we ask for it.
name = sys.argv[1] if len(sys.argv) > 1 else input("Name to enroll: ")
name = re.sub(r"\s+", "_", name.strip())
if not name or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
    sys.exit("Invalid name: use only letters, digits, '_' and '-'.")

destination = folder / f"{name}_face.npy"
if destination.exists():
    answer = input(f"{destination.name} already exists. Overwrite? [y/N]: ")
    if answer.strip().lower() != "y":
        sys.exit("Enrollment cancelled.")

detector = cv2.FaceDetectorYN.create(
    str(folder / "yunet.onnx"), "", (320, 320), 0.9
)
recognizer = cv2.FaceRecognizerSF.create(
    str(folder / "sface.onnx"), ""
)

camera = cv2.VideoCapture(0)

try:
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam.")

    while True:
        success, frame = camera.read()
        if not success:
            print("Could not read the webcam.")
            break

        height, width = frame.shape[:2]
        detector.setInputSize((width, height))
        _, faces = detector.detect(frame)

        preview = frame.copy()
        count = 0 if faces is None else len(faces)

        if count == 1:
            x, y, w, h = map(int, faces[0][:4])
            cv2.rectangle(
                preview, (x, y), (x + w, y + h),
                (0, 255, 0), 2
            )
            instruction = "Press S to enroll your face"
        else:
            instruction = "Show exactly one face"

        cv2.putText(
            preview, instruction, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )

        cv2.imshow("Enroll face - Q to quit", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") and count == 1:
            aligned = recognizer.alignCrop(frame, faces[0])
            features = recognizer.feature(aligned)
            np.save(destination, features)
            print(f"Face enrolled! Saved to: {destination}")
            break

        if key == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()