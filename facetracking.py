import cv2
import numpy as np
import pyfirmata
from cvzone.FaceDetectionModule import FaceDetector


# Camera setup
camera = cv2.VideoCapture(0)

frame_width = 1280
frame_height = 720

camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

if not camera.isOpened():
    print("Unable to open camera")
    raise SystemExit


# Arduino and servo setup
arduino = pyfirmata.Arduino("COM3")

horizontal_servo = arduino.get_pin("d:9:s")
vertical_servo = arduino.get_pin("d:10:s")

servo_angles = [90, 90]


# Face detector
face_detector = FaceDetector()


while True:

    camera_ok, frame = camera.read()

    if not camera_ok:
        print("Failed to read camera frame")
        break

    frame, faces = face_detector.findFaces(frame, draw=False)

    if len(faces) > 0:

        # Center position of the first detected face
        face_x, face_y = faces[0]["center"]

        # Map camera coordinates to servo angles
        x_angle = np.interp(
            face_x,
            (0, frame_width),
            (180, 0)
        )

        y_angle = np.interp(
            face_y,
            (0, frame_height),
            (0, 180)
        )

        # Keep both servos inside their valid range
        x_angle = np.clip(x_angle, 0, 180)
        y_angle = np.clip(y_angle, 0, 180)

        servo_angles = [x_angle, y_angle]

        # Draw targeting display
        target = (face_x, face_y)

        cv2.line(frame, (0, face_y), (frame_width, face_y),
                 (0, 0, 0), 2)

        cv2.line(frame, (face_x, 0), (face_x, frame_height),
                 (0, 0, 0), 2)

        cv2.circle(frame, target, 80, (0, 0, 255), 2)
        cv2.circle(frame, target, 15, (0, 0, 255), -1)

        cv2.putText(
            frame,
            f"({face_x}, {face_y})",
            (face_x + 15, face_y - 15),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            "TARGET LOCKED",
            (850, 50),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 0, 255),
            3
        )

    else:

        center_x = frame_width // 2
        center_y = frame_height // 2
        screen_center = (center_x, center_y)

        cv2.putText(
            frame,
            "NO TARGET",
            (880, 50),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (0, 0, 255),
            3
        )

        cv2.line(frame, (0, center_y), (frame_width, center_y),
                 (0, 0, 0), 2)

        cv2.line(frame, (center_x, 0), (center_x, frame_height),
                 (0, 0, 0), 2)

        cv2.circle(frame, screen_center, 80, (0, 0, 255), 2)
        cv2.circle(frame, screen_center, 15, (0, 0, 255), -1)


    # Display current servo angles
    cv2.putText(
        frame,
        f"Servo X: {int(servo_angles[0])} deg",
        (50, 50),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Servo Y: {int(servo_angles[1])} deg",
        (50, 100),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (255, 0, 0),
        2
    )


    # Move servos
    horizontal_servo.write(servo_angles[0])
    vertical_servo.write(servo_angles[1])


    cv2.imshow("Face Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
