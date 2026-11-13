import cv2
import numpy as np
from pathlib import Path

folder = Path(__file__).resolve().parent

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
            destination = folder / "Len_face.npy"
            np.save(destination, features)
            print(f"Face enrolled! Saved to: {destination}")
            break

        if key == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()