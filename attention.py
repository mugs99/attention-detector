import cv2
import mediapipe as mp
import numpy as np
import sqlite3
from datetime import datetime

# Constants
KNOWN_DISTANCE = 76.2
KNOWN_WIDTH = 14.3
EYE_DISTANCE_THRESHOLD = 80
EYEBROW_DISTANCE_THRESHOLD = 140
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONTS = cv2.FONT_HERSHEY_COMPLEX

# Face mesh detector object
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Connect to SQLite database
conn = sqlite3.connect('attention.db')
cursor = conn.cursor()

# Create a table to store face data
cursor.execute('''
    CREATE TABLE IF NOT EXISTS face_data (
        timestamp TEXT,
        face_angle REAL,
        zone_text TEXT,
        attention_text TEXT
    )
''')
conn.commit()

# Store face data in the database
def store_face_data(timestamp, face_angle, zone_text, attention_text):
    cursor.execute('''
        INSERT INTO face_data
        VALUES (?, ?, ?, ?)
    ''', (timestamp, face_angle, zone_text, attention_text))
    conn.commit()

# Focal length finder function
def focal_length_finder(measured_distance, real_width, width_in_rf_image):
    return (width_in_rf_image * measured_distance) / real_width

# Face orientation estimation function
def face_orientation(face_landmarks):
    left_eye = (int(face_landmarks.landmark[159].x * frame.shape[1]),
                int(face_landmarks.landmark[159].y * frame.shape[0]))
    right_eye = (int(face_landmarks.landmark[386].x * frame.shape[1]),
                 int(face_landmarks.landmark[386].y * frame.shape[0]))

    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    nose_tip = (int(face_landmarks.landmark[5].x * frame.shape[1]),
                int(face_landmarks.landmark[5].y * frame.shape[0]))

    # Calculate the vector from eye center to nose tip
    vector = np.array(nose_tip) - np.array(eye_center)

    # Calculate the angle of the vector
    angle = np.degrees(np.arctan2(vector[1], vector[0]))
    return angle

# Distance estimation function
def distance_finder(focal_length, face_landmarks, face_width, face_size_factor):
    if face_landmarks.landmark[0].z == 0 or face_landmarks.landmark[5].z == 0:
        return None

    nose_z = (face_landmarks.landmark[5].z + face_landmarks.landmark[0].z) / 2
    adjusted_known_width = KNOWN_WIDTH * face_size_factor
    return (adjusted_known_width * focal_length) / face_width, nose_z

# Face data function
def face_data(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = face_mesh.process(image)
    face_width = 0
    face_size_factor = 1.0  # Initialize face_size_factor

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        left_eye_outer = (int(landmarks.landmark[33].x * image.shape[1]),
                          int(landmarks.landmark[33].y * image.shape[0]))
        right_eye_outer = (int(landmarks.landmark[263].x * image.shape[1]),
                           int(landmarks.landmark[263].y * image.shape[0]))
        face_width = abs(right_eye_outer[0] - left_eye_outer[0])
        face_size_factor = face_width / KNOWN_WIDTH

    return results, face_width, face_size_factor

# Reading reference image from directory
ref_image = cv2.imread("reference.png")
ref_results, ref_image_face_width, _ = face_data(ref_image)

# Get the focal length
focal_length_found = focal_length_finder(KNOWN_DISTANCE, KNOWN_WIDTH, ref_image_face_width)
print(f"Focal Length: {focal_length_found}")

# Show the reference image
cv2.imshow("Reference Image", ref_image)

# Initialize the camera object
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    # Calling face_data function to find the width of face(pixels) in the frame
    results, face_width_in_frame, face_size_factor = face_data(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if face_width_in_frame != 0:
        distance, nose_z = distance_finder(focal_length_found, results.multi_face_landmarks[0], face_width_in_frame,
                                           face_size_factor)

        for face_landmarks in results.multi_face_landmarks:
            left_eye = (int(face_landmarks.landmark[159].x * frame.shape[1]),
                        int(face_landmarks.landmark[159].y * frame.shape[0]))
            right_eye = (int(face_landmarks.landmark[386].x * frame.shape[1]),
                         int(face_landmarks.landmark[386].y * frame.shape[0]))
            left_eyebrow = (int(face_landmarks.landmark[145].x * frame.shape[1]),
                            int(face_landmarks.landmark[145].y * frame.shape[0]))
            right_eyebrow = (int(face_landmarks.landmark[377].x * frame.shape[1]),
                             int(face_landmarks.landmark[377].y * frame.shape[0]))
            left_eye_outer = (int(results.multi_face_landmarks[0].landmark[33].x * frame.shape[1]),
                              int(results.multi_face_landmarks[0].landmark[33].y * frame.shape[0]))
            right_eye_outer = (int(results.multi_face_landmarks[0].landmark[263].x * frame.shape[1]),
                               int(results.multi_face_landmarks[0].landmark[263].y * frame.shape[0]))
            face_size = abs(right_eye_outer[0] - left_eye_outer[0])

            distance, nose_z = distance_finder(focal_length_found, results.multi_face_landmarks[0], face_width_in_frame,
                                               face_size_factor)

            # Calculate the distance between eyes
            eye_distance = abs(right_eye[0] - left_eye[0])

            # Calculate the distance between eyebrows
            eyebrow_distance = abs(right_eyebrow[1] - left_eyebrow[1])

            face_angle = face_orientation(face_landmarks)

        # Draw line as background of text
        cv2.line(frame, (30, 30), (230, 30), RED, 32)
        cv2.line(frame, (30, 30), (230, 30), BLACK, 28)

        # Drawing Text on the screen
        cv2.putText(frame, f"face size: {face_size}", (30, 35), FONTS, 0.6, GREEN, 2)
        cv2.putText(frame, f"Eye Distance: {eye_distance}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
        cv2.putText(frame, f"Eyebrow Distance: {eyebrow_distance}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
        cv2.putText(frame, f"Face Orientation: {face_angle:.2f} degrees", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    WHITE, 2)

        # Face Zone check
        zone_text = "Zone: Subject not detected"

        if face_size > 230:
            zone_text = "Zone: Subject too close"
        elif 150 < face_size < 230:
            zone_text = "Zone: Close"
        elif 100 < face_size < 150:
            zone_text = "Zone: Normal"
        elif 70 < face_size < 100:
            zone_text = "Zone: Far"

        if zone_text == "Zone: Subject too close":
            attention_text = "Not Attentive"
        elif 70 < face_angle < 110:
            attention_text = "Attentive"
        else:
            attention_text = "Not Attentive"

        # Timestamp for the current frame
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Store data in the database
        store_face_data(timestamp, face_angle, zone_text, attention_text)

        cv2.putText(frame, zone_text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    RED if ("too" or "not") in zone_text else GREEN, 2)
        cv2.putText(frame, attention_text, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    RED if "Not" in attention_text else GREEN, 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Close the SQLite connection
conn.close()
