import cv2
import mediapipe as mp
import math
import ikpy.chain
import ikpy.utils.plot as plot_utils
import numpy as np
import os
import matplotlib.pyplot as plt

# -------------------------
# MediaPipe
# -------------------------
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# -------------------------
# Camera
# -------------------------
cap = cv2.VideoCapture(0)
cv2.namedWindow("Arm & Claw Gripper Tracking")

# -------------------------
# Load URDF
# -------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(script_dir, "actual_arm_urdf.urdf")

print("\nURDF path:", urdf_path)
print("URDF exists:", os.path.exists(urdf_path), "\n")

my_chain = ikpy.chain.Chain.from_urdf_file(
    urdf_path,
    active_links_mask=[False, True, True, True, True, True, False]
 
    # last three True = wrist_joint_2 + finger_left + finger_right
)

my_chain._base_pose = np.eye(4)

# -------------------------
# 3D Robot Viewer
# -------------------------
fig, ax = plot_utils.init_3d_figure()
fig.set_figwidth(8)
fig.set_figheight(6)
plt.ion()

def center_axes():
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 0.6)
    try:
        ax.set_box_aspect([1,1,1])
    except:
        pass

center_axes()

# -------------------------
# Robot target
# -------------------------
target_position = [0, 0.2, 0.1]
target_orientation = [-1, 0, 0]

ik = my_chain.inverse_kinematics(target_position, target_orientation, orientation_mode="Y")

# -------------------------
# Gripper + wrist state
# -------------------------
grip_value = 0.0
wrist_twist = 0.0

# -------------------------
# Helpers
# -------------------------
def calculate_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def clamp(x, a, b):
    return max(a, min(b, x))

def remap(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# -------------------------
# IK + Plot update
# -------------------------
def doIK():
    global ik, grip_value, wrist_twist

    ik = my_chain.inverse_kinematics(
        target_position,
        target_orientation,
        orientation_mode="Y",
        initial_position=ik
    )

    # Apply wrist rotation
    ik[-3] = wrist_twist

    # Apply gripper fingers
    ik[-2] = grip_value  # left finger
    ik[-1] = grip_value  # right finger

    ax.cla()
    center_axes()
    my_chain.plot(ik, ax, target=target_position)
    plt.pause(0.001)

def move(x, y, z):
    global target_position
    target_position = [x, y, z]
    doIK()

# -------------------------
# Models
# -------------------------
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# -------------------------
# Main loop
# -------------------------
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(rgb)
    hand_results = hands.process(rgb)

    # -------------------------
    # Neck reference
    # -------------------------
    neck_ref = None
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark
        ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        neck_ref = {
            "x": (ls.x + rs.x) / 2,
            "y": (ls.y + rs.y) / 2,
            "z": (ls.z + rs.z) / 2
        }

        px = (int(neck_ref["x"] * w), int(neck_ref["y"] * h))
        cv2.circle(image, px, 10, (255, 255, 0), -1)

    # -------------------------
    # Hand tracking
    # -------------------------
    if hand_results.multi_hand_landmarks and neck_ref:
        hand = hand_results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

        thumb = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring = hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky = hand.landmark[mp_hands.HandLandmark.PINKY_TIP]

        # Gripper
        pinch = calculate_distance(
            (thumb.x, thumb.y),
            (index.x, index.y)
        )
        grip_value = remap(pinch, 0.02, 0.12, 0.0, 1.0)
        grip_value = clamp(grip_value, 0.0, 1.0)


        # Wrist rotation
        ix, iy = index.x, index.y
        px, py = pinky.x, pinky.y
        hand_angle = math.atan2(py - iy, px - ix)
        wrist_twist = clamp(hand_angle, -1.5, 1.5)

        # End-effector XYZ
        fingers_x = (index.x + middle.x + ring.x + pinky.x) / 4
        fingers_y = (index.y + middle.y + ring.y + pinky.y) / 4
        fingers_z = (index.z + middle.z + ring.z + pinky.z) / 4

        fingers_rel = {
            "x": fingers_x - neck_ref["x"],
            "y": neck_ref["y"] - fingers_y,
            "z": fingers_z - neck_ref["z"]
        }
        z_scaled = -fingers_rel["z"] * 0.5 + 0.3

        move(fingers_rel["x"], z_scaled, fingers_rel["y"])

    cv2.imshow("Arm & Claw Gripper Tracking", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
