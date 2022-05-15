#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

#Carrega o vídeo
video = cv2.VideoCapture("q1.mp4")
if not video.isOpened():
    print("Cannot open video")
    exit()

while True:
    ret, frame = video.read()

    #Se o frame é lido corretamente, ret é true
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #Converte vídeo para HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    #Definindo intervalo de cor
    image_lower_hsv = np.array([0, 0, 0])  
    image_upper_hsv = np.array([255, 255, 80])

    mask_hsv = cv2.inRange(frame_hsv, image_lower_hsv, image_upper_hsv)

    #Encontra contornos
    contornos, _ = cv2.findContours(mask_hsv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame, contornos, -1, [0, 255, 0], 1);

    frame = cv2.putText(frame, 'Naipe: Espadas', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)


    # Mostra o frame final
    cv2.imshow('img', frame)
    if cv2.waitKey(20) == ord('q'):
        break

# That's how you exit
video.release()
cv2.destroyAllWindows()