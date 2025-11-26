import cv2
import mediapipe as mp
import math
import ikpy.chain
import ikpy.utils.plot as plot_utils
import numpy as np
import matplotlib.pyplot as plt
import serial
from collections import deque
import time


# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# -------------------------
# ★ ADD: SMOOTHING BUFFERS
# -------------------------
SMOOTHING_WINDOW = 5
# Z-axis scaling constants
Z_SCALE_FACTOR = 0.5  # Scale depth sensitivity
Z_OFFSET = 0.3        # Base height offset

smooth_buffers = {
    "thumb": deque(maxlen=SMOOTHING_WINDOW),
    "finger_center": deque(maxlen=SMOOTHING_WINDOW)
}

def smooth_point(buffer, new_point):
    """Average last N points for smoother tracking."""
    buffer.append(np.array(new_point))
    return tuple(np.mean(buffer, axis=0).astype(int))
# -----------------------------------------------------

# Initialize webcam
cap = cv2.VideoCapture(0)

# Window setup - position on LEFT
cv2.namedWindow('Arm & Claw Gripper Tracking', cv2.WINDOW_NORMAL)
cv2.moveWindow('Arm & Claw Gripper Tracking', 0, 0)  # Top-left corner
cv2.resizeWindow('Arm & Claw Gripper Tracking', 640, 480)


# Arm landmarks
arm_landmarks = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_WRIST,
]

# Arm connections
arm_connections = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
]

def calculate_distance(point1, point2):
    """Calculate distance between two points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Configure MediaPipe
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose, \
     mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

        my_chain = ikpy.chain.Chain.from_urdf_file("actual_arm_urdf.urdf", active_links_mask=[False, True, True, True, True, True])

        target_position = [0, 0.2, 0.1]

        target_orientation = [-1, 0, 0]

        ik = my_chain.inverse_kinematics(target_position, target_orientation, orientation_mode="Y")
        print("The angles of each joints are: ", list(map(lambda r: math.degrees(r), ik.tolist())))
        computed_position = my_chain.forward_kinematics(ik)
        print("Computed position:", [f"{val:.2f}" for val in computed_position[:3, 3]])
        
        fig, ax = plot_utils.init_3d_figure()
        fig.set_figheight(18)  
        fig.set_figwidth(26)  
        my_chain.plot(ik, ax, target=target_position)
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(650, 0, 800, 600)
        plt.xlim(-0.5, 0.5)
        plt.ylim(-0.5, 0.5)
        ax.set_zlim(0, 0.6)
        plt.ion()
        plt.show(block= False)

        last_update_time = 0

        def doIK():
            global ik
            old_position = ik.copy()
            ik = my_chain.inverse_kinematics(target_position, target_orientation, orientation_mode="Y", initial_position=old_position)

        def updatePlot():
            ax.cla()
            my_chain.plot(ik, ax, target=target_position)
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_zlim(0, 0.6)
            fig.canvas.draw()
            plt.pause(0.001)
            
        def move(x,y,z): #Fix 2: Move function to match coordinate system
            global target_position, last_update_time
            x = max(-0.5, min(0.5, x))
            y = max(-0.5, min(0.5, y))
            z = max(0.0, min(0.6, z))
            current_time = time.time()
            if current_time - last_update_time > 0.1:  
                target_position = [x,y,z]
                doIK()
                updatePlot()
                last_update_time = current_time
    
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            
            # Flip for selfie-view
            image = cv2.flip(image, 1)
            h, w, c = image.shape
            
            # Single RGB conversion
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process
            pose_results = pose.process(image_rgb)
            hand_results = hands.process(image_rgb)
            
            neck_ref = None

            # Draw ARM tracking
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                # Calculate neck reference FIRST
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                neck_ref = {
                    'x': (left_shoulder.x + right_shoulder.x) / 2,
                    'y': (left_shoulder.y + right_shoulder.y) / 2,
                    'z': (left_shoulder.z + right_shoulder.z) / 2
                }
                
                # Draw neck point
                neck_px = (int(neck_ref['x'] * w), int(neck_ref['y'] * h))
                cv2.circle(image, neck_px, 10, (255, 255, 0), -1)
                cv2.circle(image, neck_px, 12, (255, 255, 255), 2)
                cv2.putText(image, "NECK (0,0,0)", (neck_px[0] + 15, neck_px[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Draw arm connections
                for connection in arm_connections:
                    start = landmarks[connection[0]]
                    end = landmarks[connection[1]]
                    start_px = (int(start.x * w), int(start.y * h))
                    end_px = (int(end.x * w), int(end.y * h))
                    cv2.line(image, start_px, end_px, (0, 255, 0), 3)
                
                # Draw arm landmarks
                for landmark_idx in arm_landmarks:
                    landmark = landmarks[landmark_idx]
                    px = (int(landmark.x * w), int(landmark.y * h))
                    cv2.circle(image, px, 8, (0, 0, 255), -1)
                    cv2.circle(image, px, 10, (255, 255, 255), 2)
                    
            # Draw HAND as simple claw
            if hand_results.multi_hand_landmarks and neck_ref:
                for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    # Draw basic hand skeleton (subtle)
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(180, 180, 180), thickness=1, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(120, 120, 120), thickness=1))

                    # Get handedness
                    handedness = hand_results.multi_handedness[hand_idx].classification[0].label
                    
                    # Get fingertips
                    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    ring = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                    
                    # Convert to pixels
                    thumb_px = (int(thumb.x * w), int(thumb.y * h))
                    index_px = (int(index.x * w), int(index.y * h))
                    middle_px = (int(middle.x * w), int(middle.y * h))
                    ring_px = (int(ring.x * w), int(ring.y * h))
                    pinky_px = (int(pinky.x * w), int(pinky.y * h))
                    
                    # Calculate center of 4 fingers (claw jaw 2)
                    fingers_center_norm = {
                        'x': (index.x + middle.x + ring.x + pinky.x) / 4,
                        'y': (index.y + middle.y + ring.y + pinky.y) / 4,
                        'z': (index.z + middle.z + ring.z + pinky.z) / 4
                    }
                    
                    fingers_center = (
                        int(fingers_center_norm['x'] * w),
                        int(fingers_center_norm['y'] * h)
                    )
                    
                    # Calculate relative coordinates from neck
                    thumb_rel = {
                        'x': thumb.x - neck_ref['x'],
                        'y': neck_ref['y'] - thumb.y, # Fix 1: Invert Y for intuitive up-down
                        'z': thumb.z - neck_ref['z']
                    }
                    
                    fingers_rel = {
                        'x': fingers_center_norm['x'] - neck_ref['x'],
                        'y': neck_ref['y'] - fingers_center_norm['y'],
                        'z': fingers_center_norm['z'] - neck_ref['z']
                    }
                
                    # Calculate midpoint between thumb and fingers for better tracking
                    mid_x = (thumb_rel['x'] + fingers_rel['x']) / 2
                    mid_y = (thumb_rel['y'] + fingers_rel['y']) / 2
                    mid_z = (thumb_rel['z'] + fingers_rel['z']) / 2

                    # Apply smoothing
                    if len(smooth_buffers["finger_center"]) > 0:
                        smooth_buffers["finger_center"].append([mid_x, mid_y, mid_z])
                        smoothed = np.mean(smooth_buffers["finger_center"], axis=0)
                        smooth_x, smooth_y, smooth_z = smoothed[0], smoothed[1], smoothed[2]
                    else:
                        smooth_buffers["finger_center"].append([mid_x, mid_y, mid_z])
                        smooth_x, smooth_y, smooth_z = mid_x, mid_y, mid_z

                    z_scaled = -smooth_z * Z_SCALE_FACTOR + Z_OFFSET  # Fix 3: Scale and offset Z-axis
                    move(smooth_x, z_scaled, smooth_y)

                    # Calculate claw opening distance
                    claw_distance = calculate_distance(thumb_px, fingers_center)
                    # Map claw distance to gripper angle (0-180°)
                    MIN_DISTANCE = 50
                    MAX_DISTANCE = 200
                    gripper_angle = np.clip((claw_distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE) * 180, 0, 180)
                    
                    # Draw claw jaws
                    # Jaw 1: Thumb (RED)
                    cv2.circle(image, thumb_px, 15, (0, 0, 255), -1)
                    cv2.circle(image, thumb_px, 17, (255, 255, 255), 2)
                    
                    # Jaw 2: Fingers center (BLUE)
                    cv2.circle(image, fingers_center, 15, (255, 0, 0), -1)
                    cv2.circle(image, fingers_center, 17, (255, 255, 255), 2)
                    
                    # Draw connection line
                    cv2.line(image, thumb_px, fingers_center, (0, 255, 255), 3)
                    
                    # Display distance
                    mid_point = (
                        int((thumb_px[0] + fingers_center[0]) / 2),
                        int((thumb_px[1] + fingers_center[1]) / 2)
                    )
                    cv2.putText(image, f"{claw_distance:.0f}px", 
                            (mid_point[0] + 10, mid_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Display coordinates relative to neck
                    panel_x = 10 if handedness == "Left" else w - 400
                    panel_y = h - 180
                    
                    cv2.putText(image, f"{handedness} Hand - Relative to Neck", 
                            (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    cv2.putText(image, f"Jaw 1 (Thumb):", 
                            (panel_x, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(image, f"  X: {thumb_rel['x']:+.3f}  Y: {thumb_rel['y']:+.3f}  Z: {thumb_rel['z']:+.3f}", 
                            (panel_x, panel_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    
                    cv2.putText(image, f"Jaw 2 (Fingers):", 
                            (panel_x, panel_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(image, f"  X: {fingers_rel['x']:+.3f}  Y: {fingers_rel['y']:+.3f}  Z: {fingers_rel['z']:+.3f}", 
                            (panel_x, panel_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    
                    cv2.putText(image, f"Distance: {claw_distance:.0f}px", 
                            (panel_x, panel_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Display
            cv2.imshow('Arm & Claw Gripper Tracking', image)

            
            # Exit on 'q'
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
