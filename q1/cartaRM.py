#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


#Criação do sift
sift = cv2.SIFT_create()


######## iMAGEM DE REFERÊNCIA ########

#Carrega imagem de referência
imgRef = cv2.imread("cartaAs.png")

#Ajusta tamanho da imagem de referência
scale_percent = 150
height = int(imgRef.shape[0] * scale_percent / 100)
width = int(imgRef.shape[1] * scale_percent / 100)
dim = (width, height)

imgRef = cv2.resize(imgRef, dim, interpolation = cv2.INTER_AREA)

#Detecta os keypoints e calcula os descritores
imgKp, imgDes = sift.detectAndCompute(imgRef, None)

######## VIDEO PARA ANALISE ########

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
    
    #Detecta os keypoints e calcula os descritores do frame
    frameKp, frameDes = sift.detectAndCompute(frame, None)


    ######## MATCH ########

    #fazendo o match, encontrando as referencias
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(imgDes, frameDes, k=2)

    # Varre as relações encontradas e cria a lista good com as melhores 
    good = []
    for m,n in matches:
        if m.distance < 0.6 * n.distance:
            good.append([m])

    #Se tiver mais de 20 matchs bons
    if(len(good) > 20):
        print("encontrei bons matchs")

        # Extrai a localizaão dos bons matches
        pointsImg = np.zeros((len(good), 2), dtype=np.float32)
        pointsFrame = np.zeros((len(good), 2), dtype=np.float32)

        for i, match in enumerate(good):
            pointsImg[i, :] = imgKp[match[0].queryIdx].pt
            pointsFrame[i, :] = frameKp[match[0].trainIdx].pt

        # Acha homography
        h, mask = cv2.findHomography(pointsImg, pointsFrame, cv2.RANSAC)
        
        # Usa homography
        height, width, channels = imgRef.shape
        pts = np.float32([ [0,0],[0,height],[width,height],[width,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,h)

        frame = cv2.polylines(frame,[np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)


        #Adiciona texto
        frame = cv2.putText(frame, 'Carta encontrada', (width, height), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, 'Numero de bons pontos: ' + str(len(good)), (width, height + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        #Gera o conjuntos das operações no frame
        result = cv2.drawMatchesKnn(imgRef, imgKp, frame, frameKp, good[:20], None, flags=2)

    #Se não tiver mais de 20 matchs bons
    else:
        print("não encontrei bons matchs")
        frame = cv2.putText(frame, 'Numero de bons pontos: ' + str(len(good)), (width, height + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        result = cv2.drawMatchesKnn(imgRef, imgKp, frame, frameKp, good[:0], None, flags=2)


    # Mostra o frame final
    cv2.imshow('frame', result)
    if cv2.waitKey(1) == ord('q'):
        break

# That's how you exit
video.release()
cv2.destroyAllWindows()