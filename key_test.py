"""
key_test.py

Tiny diagnostic tool - opens a plain OpenCV window and prints the raw key
code every time you press a key. Use this if collect_sequence_data.py's
Enter/Backspace/Esc handling doesn't seem to register correctly, to find
out what key codes your system/keyboard actually sends.

Press 'x' to quit this test.
"""

import cv2
import numpy as np

print(__doc__)

blank = np.zeros((200, 500, 3), dtype="uint8")
cv2.putText(blank, "Press keys here (x to quit)", (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

while True:
    cv2.imshow("Key Test", blank)
    key = cv2.waitKey(0) & 0xFF  # waits indefinitely for a keypress
    print(f"Key pressed: {key}")
    if key == ord("x"):
        break

cv2.destroyAllWindows()
