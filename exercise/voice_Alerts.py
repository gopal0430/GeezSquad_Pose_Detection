import cv2
import numpy as np
import mediapipe as mp
import math
import time
import pyttsx3

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Initialize variables
squat_count = 0
squat_position = False
valid_position = False
prev_foot_pos = None
frame_count = 0
show_instructions = True
start_time = time.time()
last_feedback = ""
last_feedback_time = 0

# Thresholds
HIP_ANGLE_THRESHOLD = 140
KNEE_ANGLE_THRESHOLD = 130
TORSO_LEAN_THRESHOLD = 20
SHOULDER_ANGLE_THRESHOLD = 35
FOOT_MOVEMENT_THRESHOLD = 0.05
MIN_FRAMES_STATIONARY = 10

def speak(message):
    global last_feedback, last_feedback_time
    current_time = time.time()
    if message != last_feedback or (current_time - last_feedback_time) > 3:
        last_feedback = message
        last_feedback_time = current_time
        engine.say(message)
        engine.runAndWait()

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return angle if angle <= 180 else 360 - angle

def is_torso_upright(shoulder, hip, knee):
    return calculate_angle(shoulder, hip, knee) > (180 - TORSO_LEAN_THRESHOLD)

def are_arms_down(shoulder, elbow, wrist, hip):
    return calculate_angle(elbow, shoulder, hip) < SHOULDER_ANGLE_THRESHOLD

def are_feet_stationary(current, previous, threshold):
    if previous is None:
        return False
    movement = math.sqrt((current[0] - previous[0])**2 + (current[1] - previous[1])**2)
    return movement < threshold

def draw_instructions(image):
    instructions = [
        "## Usage Tips:",
        "- Stand 2-3 meters from the camera",
        "- Keep entire body visible in frame",
        "- Maintain upright torso position",
        "- Keep arms at your sides",
        "- Perform squats slowly",
        "- Avoid walking/shifting feet",
        "",
        "Press any key to continue..."
    ]
    overlay = image.copy()
    cv2.rectangle(overlay, (50, 50), (600, 350), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    for i, text in enumerate(instructions):
        y = 100 + i * 40
        color = (255, 255, 255) if i != 0 else (0, 255, 255)
        scale = 0.7 if i != 0 else 0.9
        thick = 1 if i != 0 else 2
        cv2.putText(image, text, (100, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if show_instructions:
        draw_instructions(image)
        cv2.imshow('Enhanced Squat Counter', image)
        if cv2.waitKey(1) != -1 or (time.time() - start_time) > 5:
            show_instructions = False
        continue

    try:
        lm = results.pose_landmarks.landmark

        left_shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x, lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        left_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        right_shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        right_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        foot_pos = [(left_ankle[0] + right_ankle[0])/2, (left_ankle[1] + right_ankle[1])/2]

        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

        torso_upright = is_torso_upright(left_shoulder, left_hip, left_knee) and is_torso_upright(right_shoulder, right_hip, right_knee)
        arms_down = are_arms_down(left_shoulder, left_elbow, left_wrist, left_hip) and are_arms_down(right_shoulder, right_elbow, right_wrist, right_hip)
        stationary = are_feet_stationary(foot_pos, prev_foot_pos, FOOT_MOVEMENT_THRESHOLD)

        if stationary:
            frame_count += 1
        else:
            frame_count = 0
            prev_foot_pos = foot_pos

        valid_position = torso_upright and arms_down and frame_count > MIN_FRAMES_STATIONARY

        knee_angle = (left_knee_angle + right_knee_angle) / 2
        hip_angle = (left_hip_angle + right_hip_angle) / 2

        if valid_position and knee_angle < KNEE_ANGLE_THRESHOLD and hip_angle < HIP_ANGLE_THRESHOLD:
            if not squat_position:
                squat_count += 1
                squat_position = True
                cv2.putText(image, "GOOD FORM!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                speak("Good squat")
        else:
            squat_position = False

        if not valid_position:
            issues = []
            if not torso_upright:
                issues.append("Keep your torso upright.")
            if not arms_down:
                issues.append("Lower your arms.")
            if frame_count < MIN_FRAMES_STATIONARY:
                issues.append("Keep your feet steady.")

            for i, txt in enumerate(issues):
                cv2.putText(image, txt, (50, 180 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if issues:
                speak(" ".join(issues))

        # Show angles
        cv2.putText(image, f"Knee: {int(knee_angle)}", tuple(np.multiply(left_knee, [image.shape[1], image.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(image, f"Hip: {int(hip_angle)}", tuple(np.multiply(left_hip, [image.shape[1], image.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    except Exception as e:
        print("Error:", e)

    cv2.rectangle(image, (0, 0), (300, 100), (245, 117, 16), -1)
    cv2.putText(image, 'SQUATS', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(image, str(squat_count), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    cv2.imshow('Enhanced Squat Counter', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
