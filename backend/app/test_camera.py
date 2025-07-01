# ~/test_camera.py
import cv2
import sys

def test_camera(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open camera {source}", file=sys.stderr)
        return False
    
    print(f"Camera {source} opened successfully")
    print(f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame", file=sys.stderr)
        cap.release()
        return False
    
    print("Frame read successfully")
    cap.release()
    return True

if __name__ == "__main__":
    for source in [0, 1, 2, "/dev/video0", "/dev/video1"]:
        print(f"\nTesting camera source: {source}")
        if test_camera(source):
            break