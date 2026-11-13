import cv2
import json
import socket
import time
from datetime import datetime
from pathlib import Path

model = Path(__file__).with_name("yunet.onnx")
detector = cv2.FaceDetectorYN.create(
    str(model), "", (320, 320), 0.9
)

# Receive our local messages so the destination port is open.
receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receiver.bind(("127.0.0.1", 5005))
receiver.settimeout(1)

sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
camera = cv2.VideoCapture(0)

event_id = 0
face_present = False
last_seen = 0

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
        now = time.monotonic()

        if faces is not None and len(faces) > 0:
            if not face_present:
                event_id += 1
                event = {
                    "event_id": event_id,
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "event": "face_detected",
                    "face_count": len(faces)
                }

                message = json.dumps(event)
                sender.sendto(
                    message.encode("utf-8"), ("127.0.0.1", 5005)
                )
                receiver.recvfrom(65535)
                print(message)

            face_present = True
            last_seen = now

            for face in faces:
                x, y, w, h = map(int, face[:4])
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h),
                    (0, 255, 0), 2
                )

        elif face_present and now - last_seen >= 2:
            face_present = False

        cv2.putText(
            frame, f"Events sent: {event_id}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        cv2.imshow("Face detection - Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    camera.release()
    sender.close()
    receiver.close()
    cv2.destroyAllWindows()