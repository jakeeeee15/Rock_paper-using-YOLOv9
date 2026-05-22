import cv2
import os
import time

# 1. Setup the output directory
output_dir = r"Data\Raw_images\stone"


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. Initialize webcam (0 is usually the built-in default webcam)
cap = cv2.VideoCapture(0)

print("=== Webcam Data Collection Script ===")
print("Press 'SPACE' to capture a frame.")
print("Press 'q' to quit.")
print("-------------------------------------")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Make a copy to display text on screen without ruining the saved image
    display_frame = frame.copy()
    cv2.putText(
        display_frame, 
        f"Captured: {count}", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0), 
        2
    )
    
    cv2.imshow("Data Collection - Hold Sign and Press Space", display_frame)

    # Listen for key presses
    key = cv2.waitKey(1) & 0xFF
    
    # Spacebar pressed -> Save the raw frame
    if key == ord(' '):
        img_name = os.path.join(output_dir, f"frame_{count}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
        count += 1
        
    # 'q' pressed -> Exit loop
    elif key == ord('q'):
        break

# Clean up window and camera stream
cap.release()
cv2.destroyAllWindows()
print(f"\nCollection finished! Total images saved in '{output_dir}': {count}")