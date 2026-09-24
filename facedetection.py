import cv2

# Load the face detection model
face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# Start webcam
camera = cv2.VideoCapture(0)

if camera.isOpened() == False:
    print("Could not connect to webcam")
    quit()

running = True

while running:

    success, frame = camera.read()

    if success:

        # Convert the camera image to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Search for faces
        faces = face_detector.detectMultiScale(
            gray_frame,
            1.3,
            2
        )

        # Draw a box around each detected face
        for face in faces:
            x, y, width, height = face

            top_left = (x, y)
            bottom_right = (x + width, y + height)

            cv2.rectangle(
                frame,
                top_left,
                bottom_right,
                (0, 255, 0),
                2
            )

        # Show how many faces were found
        face_count = len(faces)
        message = f"Faces Detected: {face_count}"

        cv2.putText(
            frame,
            message,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            1
        )

        cv2.imshow("Face Detector", frame)

    # Press Q to close the program
    key = cv2.waitKey(1)

    if key & 0xFF == ord("q"):
        running = False


camera.release()
cv2.destroyAllWindows()
