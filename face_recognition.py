import cv2
import numpy as np
import json
import socket
import uuid
from datetime import datetime
from pathlib import Path

folder = Path(__file__).resolve().parent
THRESHOLD = 0.30

detector = cv2.FaceDetectorYN.create(
    str(folder / "yunet.onnx"), "", (320, 320), 0.9
)
recognizer = cv2.FaceRecognizerSF.create(
    str(folder / "sface.onnx"), ""
)

# Example: elijah_face.npy becomes the name "elijah".
enrolled = {}
for path in sorted(folder.glob("*_face.npy")):
    name = path.stem.removesuffix("_face").replace("_", " ").title()
    enrolled[name] = np.load(path, allow_pickle=False)

if not enrolled:
    raise RuntimeError("No enrolled faces found. Run enrollment first.")

print("Enrolled people:", ", ".join(enrolled))

receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receiver.bind(("127.0.0.1", 5005))
receiver.settimeout(1)

sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
camera = cv2.VideoCapture(0)
cv2.namedWindow("Multi-person recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Multi-person recognition", 960, 720)

session_id = str(uuid.uuid4())
attempt_id = 0

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
        results = []

        if faces is not None:
            for face_index, face in enumerate(faces, start=1):
                aligned = recognizer.alignCrop(frame, face)
                features = recognizer.feature(aligned)

                scores = {
                    name: float(recognizer.match(
                        reference, features,
                        cv2.FaceRecognizerSF_FR_COSINE
                    ))
                    for name, reference in enrolled.items()
                }

                best_name = max(scores, key=scores.get)
                best_score = scores[best_name]
                recognized = best_score >= THRESHOLD

                identity = best_name if recognized else None
                result = "recognized" if recognized else "unknown"
                label = identity if recognized else "Unknown"
                color = (0, 255, 0) if recognized else (0, 0, 255)

                x, y, w, h = map(int, face[:4])

                results.append({
                    "face_index": face_index,
                    "result": result,
                    "identity": identity,
                    "similarity": round(best_score, 4),
                    "bbox": [x, y, w, h]
                })

                cv2.rectangle(
                    preview, (x, y), (x + w, y + h), color, 2
                )
                cv2.putText(
                    preview,
                    f"{label} | {best_score:.3f}",
                    (max(0, x), max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

        cv2.putText(
            preview,
            f"Faces: {len(results)} | A: log all | Q: quit",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 2
        )
        cv2.imshow("Multi-person recognition", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("a"):
            if not results:
                print("No faces visible; no attempt logged.")
                continue

            attempt_id += 1
            timestamp = datetime.now().astimezone().isoformat()

            for face_result in results:
                event = {
                    "session_id": session_id,
                    "attempt_id": attempt_id,
                    "timestamp": timestamp,
                    "event": "recognition_attempt",
                    "face_count": len(results),
                    "threshold": THRESHOLD,
                    **face_result
                }

                message = json.dumps(event)
                sender.sendto(
                    message.encode("utf-8"), ("127.0.0.1", 5005)
                )

                try:
                    receiver.recvfrom(65535)
                except socket.timeout:
                    print("Local receiver timed out; check capture.")

                print(message)

        if key == ord("q"):
            break
finally:
    camera.release()
    sender.close()
    receiver.close()
    cv2.destroyAllWindows()
