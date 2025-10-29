import cv2
import mediapipe as mp

# Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Tên các ngón tay và điểm quan trọng
finger_tips = {
    4: "Thumb",      # Ngón cái
    8: "Index",      # Ngón trỏ
    12: "Middle",    # Ngón giữa
    16: "Ring",      # Ngón áp út
    20: "Pinky"      # Ngón út
}

cap = cv2.VideoCapture(0)

print("Camera is opened. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Lật frame (như gương)
    frame = cv2.flip(frame, 1)
    
    # Chuyển sang RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hands
    results = hands.process(rgb_frame)
    
    # Nếu phát hiện bàn tay
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Vẽ landmarks
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )
            
            # Lấy kích thước frame
            h, w, c = frame.shape
            
            # Hiển thị tọa độ các đầu ngón tay
            for id, finger_name in finger_tips.items():
                # Lấy landmark
                landmark = hand_landmarks.landmark[id]
                
                # Chuyển đổi tọa độ chuẩn hóa (0-1) sang pixel
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                
                # Vẽ điểm lớn hơn cho các đầu ngón
                cv2.circle(frame, (cx, cy), 8, (255, 0, 255), cv2.FILLED)
                
                # Hiển thị tên và tọa độ
                text = f"{finger_name}"
                cv2.putText(frame, text, (cx + 10, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Hiển thị tọa độ chi tiết ở góc màn hình
            y_offset = 30
            cv2.putText(frame, "Finger Coordinates:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            y_offset += 30
            for id, finger_name in finger_tips.items():
                landmark = hand_landmarks.landmark[id]
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                
                coord_text = f"{finger_name}: ({cx}, {cy})"
                cv2.putText(frame, coord_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
    
    # Hiển thị hướng dẫn
    cv2.putText(frame, "Press ESC to exit", (10, frame.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    cv2.imshow('Camera', frame)
    
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
hands.close()