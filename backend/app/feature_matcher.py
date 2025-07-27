# matcher.py
import cv2
import numpy as np

class ObjectMatcher:
    def __init__(self, ref_img_path: str):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
        self.kp1, self.des1 = self.orb.detectAndCompute(self.ref_img, None)
        self.match_result = False

    def compare_frame(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = self.orb.detectAndCompute(gray_frame, None)

        if des2 is not None and self.des1 is not None:
            matches = self.bf.match(self.des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 50]

            if len(good_matches) > 15:
                self.match_result = True
                print(f"[MATCH ✅] Good Matches: {len(good_matches)}")
            else:
                self.match_result = False
                print(f"[NO MATCH ❌] Good Matches: {len(good_matches)}")
        else:
            self.match_result = False
            print("[WARNING ⚠️] Feature detection failed")

    def get_result(self):
        return self.match_result
