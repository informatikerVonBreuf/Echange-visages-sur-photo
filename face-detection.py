import cv2 as cv

#charger les classificateur sen cascade 
face_cascade =cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# charger les images 
img = cv.imread('obama.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# execution de la detection de vsage
faces= face_cascade.detectMultiScale(gray, 1.1, 8)
# afficher les visages 
i=0
for face in faces :
  x , y , w, h = face
  # dessinner le rectangle sur l'image principale

  cv.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

  # etraire les visages de l'image principale
  # opencv et Numpy : y<-> row et x<-> col
  face = img[y:y+h, x:x+w]


  # affiche l'image principale
  cv.imshow('face{}'.format(i), face)
  i+=1

cv.imshow('image principale', img)
cv.waitKey(0)
cv.destroyAllWindows()