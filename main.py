import cv2
import mediapipe as mp

# Initialize MediaPipe Pose and drawing utilities.
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Open the input video.
cap = cv2.VideoCapture('pose.mp4')
if not cap.isOpened():
    print("Error opening pose.mp4")
    exit()

# Retrieve video properties for output writer.
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('annotated.mp4', fourcc, fps, (width, height))

# Prepare a list of landmark names from the PoseLandmark enum.
landmark_names = [landmark.name for landmark in mp_pose.PoseLandmark]

# Initialize the Pose estimator.
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB before processing.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # If pose landmarks are detected, draw them.
        if results.pose_landmarks:
            # Draw the skeleton with connections.
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            # Label each landmark with its name.
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.putText(frame, landmark_names[idx], (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Write the annotated frame to the output video.
        out.write(frame)

# Release all resources.
cap.release()
out.release()
print("Annotated video saved as annotated.mp4")

