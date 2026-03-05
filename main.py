import cv2
import numpy as np

# Webcam + face detection setup
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

SAFFRON = (51, 153, 255)
WHITE = (255, 255, 255)
GREEN = (8, 136, 8)
NAVY_BLUE = (0, 0, 128)

def create_tricolor_overlay(width, height):

    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    third = height // 3

    overlay[:third, :] = (*SAFFRON, 180)
    overlay[third:2*third, :] = (*WHITE, 150)
    overlay[2*third:, :] = (*GREEN, 180)

    center_x = width // 2
    center_y = third + third // 2
    radius = min(width, height) // 8

    cv2.circle(overlay, (center_x, center_y), radius, (*NAVY_BLUE, 200), -1)

    for i in range(24):
        angle = i * (360 / 24)
        radian = np.deg2rad(angle)
        x = int(center_x + radius * np.cos(radian))
        y = int(center_y + radius * np.sin(radian))
        cv2.line(overlay, (center_x, center_y), (x, y), (*WHITE, 220), 2)

    return overlay

def apply_tricolor_effect(frame, faces):

    result = frame.copy()

    for (x, y, w, h) in faces:
        overlay = create_tricolor_overlay(w, h)

        bgr = overlay[:, :, :3]
        alpha = overlay[:, :, 3] / 255.0

        for c in range(3):
            result[y:y+h, x:x+w, c] = (
                alpha * bgr[:, :, c] +
                (1 - alpha) * result[y:y+h, x:x+w, c]
            )

    return result

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    result = apply_tricolor_effect(frame, faces)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Happy Independence Day"
    text_size = cv2.getTextSize(text, font, 1.2, 3)[0]
    text_x = (result.shape[1] - text_size[0]) // 2
    text_y = 50

    cv2.rectangle(result, (text_x - 10, text_y - text_size[1] - 10),
                 (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)

    cv2.putText(result, text, (text_x, text_y), font, 1.2, (255, 255, 255), 3)

    cv2.imshow("Indian Independence Day Face Filter", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
