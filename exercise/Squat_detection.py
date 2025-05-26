import cv2  
import numpy as np  
import mediapipe as mp  
import math  
import time  
  
# Initialize MediaPipe Pose  
mp_pose = mp.solutions.pose  
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)  
mp_drawing = mp.solutions.drawing_utils  
  
# Initialize variables  
squat_count = 0  
squat_position = False  
valid_position = False  
prev_foot_pos = None  
frame_count = 0  
show_instructions = True  
start_time = time.time()  
  
# Thresholds  
HIP_ANGLE_THRESHOLD = 140  # Degrees to consider as squat  
KNEE_ANGLE_THRESHOLD = 130  
TORSO_LEAN_THRESHOLD = 20  # Max degrees torso can lean forward  
SHOULDER_ANGLE_THRESHOLD = 35  # Max arm raise angle  
FOOT_MOVEMENT_THRESHOLD = 0.05  # Max foot movement allowed  
MIN_FRAMES_STATIONARY = 10  # Min frames feet must be stationary  
  
def calculate_angle(a, b, c):  
    """Calculate the angle between three points"""  
    a = np.array(a)  # First point  
    b = np.array(b)  # Mid point  
    c = np.array(c)  # End point  
      
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])  
    angle = np.abs(radians*180.0/np.pi)  
      
    return angle if angle <= 180 else 360 - angle  
  
def is_torso_upright(shoulder, hip, knee):  
    """Check if torso is not leaning too far forward"""  
    torso_angle = calculate_angle(shoulder, hip, knee)  
    return torso_angle > (180 - TORSO_LEAN_THRESHOLD)  
  
def are_arms_down(shoulder, elbow, wrist, hip):  
    """Check if arms are not raised"""  
    arm_angle = calculate_angle(elbow, shoulder, hip)  
    return arm_angle < SHOULDER_ANGLE_THRESHOLD  
  
def are_feet_stationary(current_foot_pos, prev_foot_pos, threshold):  
    """Check if feet haven't moved significantly"""  
    if prev_foot_pos is None:  
        return False  
      
    movement = math.sqrt((current_foot_pos[0] - prev_foot_pos[0])**2 +   
                         (current_foot_pos[1] - prev_foot_pos[1])**2)  
    return movement < threshold  
  
def draw_instructions(image):  
    """Draw the usage tips on the image"""  
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
      
    # Draw semi-transparent background  
    overlay = image.copy()  
    cv2.rectangle(overlay, (50, 50), (600, 350), (50, 50, 50), -1)  
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)  
      
    # Draw instructions  
    for i, text in enumerate(instructions):  
        y_position = 100 + i * 40  
        color = (255, 255, 255) if i != 0 else (0, 255, 255)  
        font_scale = 0.7 if i != 0 else 0.9  
        thickness = 1 if i != 0 else 2  
        cv2.putText(image, text, (100, y_position),   
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)  
  
cap = cv2.VideoCapture(0)  
  
while cap.isOpened():  
    ret, frame = cap.read()  
    if not ret:  
        break  
      
    frame = cv2.flip(frame, 1)  # Mirror image  
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    image.flags.writeable = False  
    results = pose.process(image)  
    image.flags.writeable = True  
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
      
    # Show instructions for first 5 seconds or until key press  
    if show_instructions:  
        draw_instructions(image)  
        cv2.imshow('Enhanced Squat Counter', image)  
          
        # Check for key press or timeout  
        if cv2.waitKey(1) != -1 or (time.time() - start_time) > 5:  
            show_instructions = False  
        continue  
      
    try:  
        landmarks = results.pose_landmarks.landmark  
          
        # Get left side coordinates  
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,   
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]  
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,  
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]  
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,  
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]  
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,  
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]  
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,  
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]  
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,  
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]  
          
        # Get right side coordinates  
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,  
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]  
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,  
                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]  
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,  
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]  
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,  
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]  
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,  
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]  
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,  
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]  
          
        # Current foot position (average of both ankles)  
        current_foot_pos = [(left_ankle[0] + right_ankle[0]) / 2,   
                             (left_ankle[1] + right_ankle[1]) / 2]  
          
        # Calculate angles  
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)  
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)  
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)  
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)  
          
        # Check form requirements  
        torso_upright = (is_torso_upright(left_shoulder, left_hip, left_knee) and   
                        is_torso_upright(right_shoulder, right_hip, right_knee))  
          
        arms_down = (are_arms_down(left_shoulder, left_elbow, left_wrist, left_hip) and   
                    are_arms_down(right_shoulder, right_elbow, right_wrist, right_hip))  
          
        feet_stationary = are_feet_stationary(current_foot_pos, prev_foot_pos, FOOT_MOVEMENT_THRESHOLD)  
          
        if feet_stationary:  
            frame_count += 1  
        else:  
            frame_count = 0  
            prev_foot_pos = current_foot_pos  
          
        # Only count if all form requirements are met  
        valid_position = (torso_upright and arms_down and   
                        frame_count >= MIN_FRAMES_STATIONARY)  
          
        # Squat detection  
        knee_angle = (left_knee_angle + right_knee_angle) / 2  
        hip_angle = (left_hip_angle + right_hip_angle) / 2  
          
        if valid_position and knee_angle < KNEE_ANGLE_THRESHOLD and hip_angle < HIP_ANGLE_THRESHOLD:  
            if not squat_position:  
                squat_count += 1  
                squat_position = True  
                cv2.putText(image, "GOOD FORM!", (50, 150),   
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)  
        else:  
            squat_position = False  
              
        # Visual feedback  
        if not valid_position:  
            feedback = []  
            if not torso_upright:  
                feedback.append("Keep torso upright")  
            if not arms_down:  
                feedback.append("Keep arms down")  
            if frame_count < MIN_FRAMES_STATIONARY:  
                feedback.append("Stay stationary")  
              
            for i, text in enumerate(feedback):  
                cv2.putText(image, text, (50, 180 + i * 30),   
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)  
                  
        # Display angles  
        cv2.putText(image, f"Knee: {int(knee_angle)}",   
                   tuple(np.multiply(left_knee, [image.shape[1], image.shape[0]]).astype(int)),   
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)  
          
        cv2.putText(image, f"Hip: {int(hip_angle)}",   
                   tuple(np.multiply(left_hip, [image.shape[1], image.shape[0]]).astype(int)),   
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)  
          
    except Exception as e:  
        print(f"Error: {e}")  
        pass  
      
    # Display squat count  
    cv2.rectangle(image, (0, 0), (300, 100), (245, 117, 16), -1)  
    cv2.putText(image, 'SQUATS', (20, 30),   
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)  
    cv2.putText(image, str(squat_count), (20, 80),   
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)  
      
    # Render pose landmarks  
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,  
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),   
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))  
      
    cv2.imshow('Enhanced Squat Counter', image)  
      
    if cv2.waitKey(10) & 0xFF == ord('q'):  
        break  
  
cap.release()  
cv2.destroyAllWindows()
