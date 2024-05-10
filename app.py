import cv2 as cv 
import sys


# ccharger le classificateur en cascade pre-entraine
face_cascade= cv.CascadeClassifier('haarcascade_frontalface_default.xml')


#charger l'image
img = cv.imread('brad-angelina.jpg')
img_gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#executer la detection de visage
faces= face_cascade.detectMultiScale(img_gray, 1.1, 8)

#Verifier le nombre de visages

if len(faces) != 2:
    sys.exit('la photo doit avoir xactement 2 visages , reessayer...')
    
# recuperation des dimensions de chaque visage
x1, y1, w1, h1 =faces[0]
x2, y2, w2, h2 =faces[1]

#extraction des deux visages de l'image

face1= img[y1:y1+h1, x1:x1+w1]    
face2= img[y2:y2+h2, x2:x2+w2] 

#redimensionner face2 aux dimensions de face1 et vice versa

face2= cv.resize(face2,(w1,h1) )
face1= cv.resize(face1, (w2,h2))
#remplacer face2 par face1
img[y2:y2+h2, x2:x2+w2]  = face1

#remplacer face1 par face2

img[y1:y1+h1, x1:x1+w1]= face2

#afficher l#echange d'image
cv.imshow('echange', img)
cv.waitKey(0)
cv.destroyAllWindows()