import cv2
import numpy as np
a = np.zeros((1000,1000,3), dtype = np.uint8)
cv2.putText(a, 'Forward direction', (
                0,500
            ),
        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255),2)
cv2.imshow('navigation viewer',
           a)
k = cv2.waitKey(0)
