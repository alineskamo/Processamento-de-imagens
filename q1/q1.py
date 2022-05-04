#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2

cap = cv2.VideoCapture("q1.mp4")


while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    # Seu código aqui. 

    # Exibe resultado
    cv2.imshow("Feed", frame)

    # Wait for key 'ESC' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# That's how you exit
cap.release()
cv2.destroyAllWindows()