import cv2
import face_recognition
import os
import numpy as np

# Load images and encode faces
def load_and_encode_images(image_folder):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, filename)
            image = face_recognition.load_image_file(image_path)
            
            # Get face encodings
            encodings = face_recognition.face_encodings(image)
            if encodings:  # Check if at least one face was found
                known_face_encodings.append(encodings[0])  # Use the first encoding
                known_face_names.append(filename)  # Save the file name
            else:
                print(f"Warning: No face detected in {filename}. Skipping this image.")
    
    return known_face_encodings, known_face_names

# Initialize known faces
IMAGE_FOLDER = "faces"  # Path to the folder containing reference images
known_face_encodings, known_face_names = load_and_encode_images(IMAGE_FOLDER)

# Check if known faces are loaded
if not known_face_encodings:
    print("Error: No valid face encodings found in the 'faces' folder. Exiting.")
    exit()

# Start video capture
video_capture = cv2.VideoCapture(0)

print("Starting live face recognition. Press 'q' to quit.")
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Ensure the frame is converted to RGB format
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find faces in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)

    # Check if face locations are valid before encoding
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            if any(matches):  # Check if there's at least one match
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    print(f"Matched with: {name}")  # Print the file name of the matched face
                else:
                    name = "Unknown"
            else:
                name = "Unknown"

            # Scale face locations back to original frame size
            top, right, bottom, left = face_location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the video feed
    cv2.imshow("Live Face Recognition", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
