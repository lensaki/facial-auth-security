import cv2

camera = cv2.VideoCapture(0)

try:
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read an image from the webcam.")
            break

        cv2.imshow("Webcam test - press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()