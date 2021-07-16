from PIL import Image, ImageFilter, ImageDraw
import turtle
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL.ImageFilter import (BLUR, CONTOUR, DETAIL, EDGE_ENHANCE,
                             EDGE_ENHANCE_MORE, EMBOSS, FIND_EDGES,
                             SMOOTH, SMOOTH_MORE, SHARPEN)

def negative(im, h, w):
    height= h
    width = w
    for row in range(height):
        for col in range(width):
            red = 255- (im.getpixel((row, col))[0]) 
            green = 255- (im.getpixel((row, col))[1]) 
            blue = 255- (im.getpixel((row, col))[2]) 
            im.putpixel((row,col),(red,green,blue))
    return im

def center(centro,centro2,w,h):
    acumula=[]
    color=(51,255,251)
    color2=(111,100,251)
    orig=centro2
    dato=centro.getextrema()# pixel mas obscuro 
    dato2=dato[0]
    dato3=dato2[1]
    print(dato3,'pixel de mayor rango')   
    for m in range(centro.height):
        for n in range(centro.width):  # extrae los valores de todos los pixel
            d1=centro.getpixel((n,m))
            d2=d1[0]
            if d2 > (dato3-120):
                centro.putpixel((n,m), color)

    draw = ImageDraw.Draw(centro)
    draw.rectangle((0, 0, w-1, h-1),
                   fill=None,
                   outline=(255, 0, 0))
    for x in range(centro.height):
        for y in range(centro.width):
            verdes=centro.getpixel((x,y))
            if verdes == color:
                #print('detectó diferente verde',x,y)
                p_ct=centro.getpixel((x,y))
                p1=centro.getpixel((x-1,y-1))
                p2=centro.getpixel((x,y-1))
                p3=centro.getpixel((x+1,y-1))
                p4=centro.getpixel((x-1,y))
                p6=centro.getpixel((x+1,y))
                p7=centro.getpixel((x-1,y+1))
                p8=centro.getpixel((x,y+1))
                p9=centro.getpixel((x+1,y+1))
                vecinos=(p1,p2,p3,p4,p6,p7,p8,p9)
                if p_ct == color:
                    if (any([d != color for d in vecinos]))== True:
                        orig.putpixel((x,y), color2)
                        acumula.append((x,y))
    #orig.save('circ_central.bmp', "BMP", quality=300)
    #orig.save('circ_centralp.png', "png", quality=300)
    #centro.save('circ_central2.png', "png", quality=300)    
    return (centro, orig)
####### se abre la imagen y obtienen los bordes 
imorig = Image.open("Ag_Nano1.bmp")
im = Image.open("Ag_Nano1.bmp")
px = im.load()
w= im.size[0]
h= im.size[1]
pix_central=center(imorig,im,w,h)
bordes=(pix_central[1])
#plt.imshow(pix_central[0])
#plt.show()
#plt.imshow(pix_central[1])
#plt.show()

####### Se cambia a blancos los bordes y fuera de ahi negro
color=(51,255,251)
color2=(111,100,251)
blanco=(255,255,255)
negro=(0,0,0)
for m in range(h):
    for n in range(w):
        d1=bordes.getpixel((m,n))   
        if d1 == color2:
            bordes.putpixel((m,n),blanco)
        else:
            bordes.putpixel((m,n),negro)
#plt.imshow(bordes)
#plt.show()

####### negativo
negativo = negative(bordes,h,w)
negativo.save('negativo.jpg',"JPEG")
#plt.imshow(negativo)
#plt.show()

###### escala de grises
negativo2 =cv2.imread('negativo.jpg')
gray= cv2.cvtColor(negativo2, cv2.COLOR_BGR2GRAY)
#plt.imshow(gray,cmap='gray')
#plt.title('negativo B/N')
#plt.show()

########### ruido gaussiano 
blur = cv2.GaussianBlur(gray,(11,11), 0)
#plt.imshow(blur,cmap='gray')
#plt.title('gaussian blur')
#plt.show()

######### canny contorno
canny= cv2.Canny(blur,30,150,3)
#plt.imshow(canny,cmap='gray')
#plt.title('canny')
#plt.show()

######### Detección contornos
contornos=[]
(cnt, heirarchy)= cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(canny, cv2.COLOR_BGR2RGB)
for i in range(len(cnt)):
    contornos.append([len(cnt[i]),i])
    
mayor=max(contornos)
borde=mayor[0]
pos=mayor[1]
print(contornos,'tamaños de contornos')
print(borde,'máximo número de pixels en contornos')
print(pos,'numero de contorno')
color_cont=(0,255,0)
cv2.drawContours(rgb,cnt, pos, color_cont, 2)
cv2.imwrite('borde.png',rgb)
plt.imshow(rgb)
plt.title('Contornos')
plt.show()

#############################################################################
new = Image.open("borde.png")
pxborde= []
pyborde= []
for m in range(h):
    for n in range(w):
        b1=new.getpixel((m,n))   
        if b1 != color_cont:
            new.putpixel((m,n),negro)
        if b1 == color_cont:
            pxborde.append(m)
            pyborde.append(n)
xmin=min(pxborde)
xmax=max(pxborde)
ymin=min(pyborde)
ymax=max(pyborde)
plt.imshow(new)
plt.title('Centro')
plt.show()
#### corta la seccion a encontrar circunferencia
corte=new.crop((xmin, ymin, xmax, ymax))
corte.save('corte.png', "png", quality=300)
plt.imshow(corte)
plt.title('Corte')
plt.show()
print(corte.size,'tamaño de imagen de contorno')
xy=corte.size
X=xy[0]
Y=xy[1]
#############################################################################


cen_circ= cv2.imread('corte.png',0)
plt.imshow(cen_circ)
plt.title('primera para hough')
plt.show()
cen_circ = cv2.medianBlur(cen_circ,5)
plt.imshow(cen_circ)
plt.title('median blur hough')
plt.show()
cimg1 = cv2.cvtColor(cen_circ,cv2.COLOR_GRAY2BGR)
cimg = cv2.cvtColor(cen_circ,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(cen_circ,cv2.HOUGH_GRADIENT,1,20,
          param1=150,param2=24,minRadius=(round(X/2))-50,maxRadius=(round(X/2))+50)#80,20

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    #dibuja fuera del circulo
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),1)
    #dibuja el centro del circulo
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),-1)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.imshow(cimg)
plt.show()






