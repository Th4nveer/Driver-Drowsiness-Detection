import cv2
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from modules.ear import calculate_ear
from modules.mar import calculate_mar
from modules.headtilt import calculate_head_tilt

#MediaPipe Setup
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

#Video Capture
cap = cv2.VideoCapture(0)

#Recording variable
record_label = None  #None = not recording
frame_count = 0

print("Press A for ALERT recording")
print("Press D for DROWSY recording")
print("Press S to STOP recording")
print("Press Q to QUIT")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    if result.face_landmarks:
        frame_height, frame_width, _ = frame.shape

        for face_landmarks in result.face_landmarks:

            #Feature Extraction
            ear = calculate_ear(face_landmarks, frame_width, frame_height)
            mar = calculate_mar(face_landmarks, frame_width, frame_height)
            head_tilt = calculate_head_tilt(face_landmarks, frame_width, frame_height)

            #Save to Dataset
            if record_label is not None and frame_count % 5 == 0:
                with open("dataset.csv", "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([ear, mar, head_tilt, record_label])

            #UI 
            FONT = cv2.FONT_HERSHEY_COMPLEX

            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 40),
                        FONT, 0.8, (0, 0, 255), 2)

            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 80),
                        FONT, 0.8, (150, 0, 0), 2)

            cv2.putText(frame, f"TILT: {head_tilt:.2f}", (30, 120),
                        FONT, 0.8, (34, 139, 34), 2)

            #Recording status
            if record_label == 0:
                cv2.putText(frame, "Recording: ALERT",
                            (300, 40), cv2.FONT_HERSHEY_COMPLEX,
                            0.7, (0, 255, 0), 2)

            elif record_label == 1:
                cv2.putText(frame, "Recording: DROWSY",
                            (300, 40), cv2.FONT_HERSHEY_COMPLEX,
                            0.7, (0, 0, 255), 2)

            #landmarks
            for landmark in face_landmarks:
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('a'):
        record_label = 0
        print("Recording ALERT samples...")

    elif key == ord('d'):
        record_label = 1
        print("Recording DROWSY samples...")

    elif key == ord('s'):
        record_label = None
        print("Recording stopped.")

    elif key == ord('q'):
        break

    cv2.imshow("ML Dataset Collection - Drowsiness Detection", frame)

cap.release()
cv2.destroyAllWindows()