import numpy as np
import cv2

#Creamos un objeto que iniciará la captura de video en la primera entrada disponible (la 0)

cap = cv2.VideoCapture(0)
#Iniciamos un bucle continua
while(True):
    #Dos objetos irán grabando las imágenes del objeto cap
    ret, frame = cap.read()

    # Especificamos el tipo de color (existen decenas) que queremos mostrar de la captura de frame y lo asignamos a un objeto que llamamos marco
    marco = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

    # Mostramos el marco de ventana con el título "Hola Mundo" con el objeto "marco" dentro
    cv2.imshow('Hola Mundo',marco)

    #el procedimiento waitKey comprueba si se ha pulsado la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Si se ha roto el bucle, procedemos a destruir la ventana y finalizar el programa
cap.release()
cv2.destroyAllWindows()