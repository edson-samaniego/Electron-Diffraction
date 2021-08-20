from PIL import Image, ImageFilter, ImageDraw
import turtle
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL.ImageFilter import (BLUR, CONTOUR, DETAIL, EDGE_ENHANCE,
                             EDGE_ENHANCE_MORE, EMBOSS, FIND_EDGES,
                             SMOOTH, SMOOTH_MORE, SHARPEN)

############# cambia a escala de grises y saca histograma
#imagen = cv2.imread("Ag_Nano1.bmp",cv2.IMREAD_GRAYSCALE)
#imagen2 = cv2.imread("Ag_Nano1.bmp")
#plt.imshow(imagen)
#plt.show()
#plt.hist(imagen.ravel(), 256, [0,256])
#plt.show()
#######################################
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

def draw_rct(dato1, dato2, imagen):
    draw = ImageDraw.Draw(imagen)
    draw.rectangle((dato1, dato2, (dato1*3), (dato2*3)),
                   fill=None,
                   outline=(0, 175, 120))
    return imagen

def cut_mid(dato1, dato2, im):
    corte=im.crop((dato1, dato2, (dato1*3), (dato2*3)))
    corte.save('Corte.bmp', "BMP", quality=300)
    return corte

imorig = Image.open("Fe3O4_Nano1.bmp")
im = Image.open("Fe3O4_Nano1.bmp")#Ag_Nano1.bmp Ag_Nano2.bmp Au_Nano_1.bmp
                               #Au_Nano_2.bmp  Fe3O4_Nano1.bmp
print('imagen 1',im.size, im.mode, im.format)
px = im.load()
w= im.size[0]
h= im.size[1]
imagen2 = cv2.imread("Fe3O4_Nano1.bmp")
plt.hist(imagen2.ravel(), 256, [0,256])
counts, bins, bars = plt.hist(imagen2.ravel(), 256, [0,256])
#print(bins)
#print(counts)
#print(counts)
plt.show()
####################################
color=(51,255,251)
color2=(111,100,251)
blanco=(255,255,255)
negro=(50,50,50)
for m in range(h):
    for n in range(w):
        d1=imorig.getpixel((m,n))   
        if d1 <= negro:
            imorig.putpixel((m,n),blanco)
        else:
            imorig.putpixel((m,n),(0,0,0))
plt.imshow(imorig)
plt.show()

########################################
for x in range(h):
    for y in range(w):
        if (imorig.getpixel((x,y))) != blanco:
            if 0 < y < (w-1):
                if 0 < x < (h-1):
                    p_ct=imorig.getpixel((x,y))
                    p1=imorig.getpixel((x-1,y-1))
                    p2=imorig.getpixel((x,y-1))
                    p3=imorig.getpixel((x+1,y-1))
                    p4=imorig.getpixel((x-1,y))
                    p6=imorig.getpixel((x+1,y))
                    p7=imorig.getpixel((x-1,y+1))
                    p8=imorig.getpixel((x,y+1))
                    p9=imorig.getpixel((x+1,y+1))
                    vecinos=(p1,p2,p3,p4,p6,p7,p8,p9)
                    if (any([d == blanco for d in vecinos]))== True:
                        imorig.putpixel((x,y), color2)
plt.imshow(imorig)
plt.show()
######################BORDES SOLOS##################
negro=(0,0,0)
for m in range(h):
    for n in range(w):
        d1=imorig.getpixel((m,n))   
        if d1 == color2:
            imorig.putpixel((m,n),negro)
        else:
            imorig.putpixel((m,n),blanco)
plt.imshow(imorig)
plt.show()
################### junta los vecinos cercanos
im_1=imorig
for m in range(h):
    for n in range(w):
        d2=im_1.getpixel((m,n))   
        if d2 == negro:
            if 0 < m < (h-1):
                if 0 < n < (w-1):
                    matriz=9
                    for p in range(-5,6):
                        for q in range(-5,6):
                            p1=im.putpixel((m+p,n+q),negro)
                    #p1=im.putpixel((m-1,n-1),negro)
                    #p2=im.putpixel((m,n-1),negro)
                    #p3=im.putpixel((m+1,n-1),negro)
                    #p4=im.putpixel((m-1,n),negro)
                    #p6=im.putpixel((m+1,n),negro)
                    #p7=im.putpixel((m-1,n+1),negro)
                    #p8=im.putpixel((m,n+1),negro)
                    #p9=im.putpixel((m+1,n+1),negro)
        else:
            im.putpixel((m,n),blanco)
plt.imshow(im)
plt.show()
###############################################################################
####### negativo
negativo = negative(im,h,w)
negativo.save('negativo.jpg',"JPEG")
plt.imshow(negativo)
plt.show()

###### escala de grises
negativo2 =cv2.imread('negativo.jpg')
gray= cv2.cvtColor(negativo2, cv2.COLOR_BGR2GRAY)
plt.imshow(gray,cmap='gray')
plt.title('negativo B/N')
plt.show()

########### ruido gaussiano 
blur = cv2.GaussianBlur(gray,(11,11), 0)
plt.imshow(blur,cmap='gray')
plt.title('gaussian blur')
plt.show()

######### canny contorno
#canny= cv2.Canny(blur,30,150,3)
canny= cv2.Canny(blur,30,150,3)
plt.imshow(canny,cmap='gray')
plt.title('canny')
plt.show()
#################################################################################
cv2.imwrite('canny.png',canny)
canny = Image.open("canny.png")
canny2 = Image.open("canny.png")

for b in range(h):
    for c in range(w):
        d2=canny.getpixel((b,c))
        if d2 >= 200:
            if 0 < b < (h-1):
                if 0 < c < (w-1):
                    matriz=9
                    for s in range(-2,3):
                        for t in range(-2,3):
                            p2=canny2.putpixel((b+s,c+t),(255))
                else:
                    canny2.putpixel((b,c),(0))
canny2.save('canny2.png',"png")
canny2=cv2.imread('canny2.png')
cv2.imshow('canny2.png', canny2)
cv2.waitKey(0)  
cv2.destroyAllWindows()
canny= cv2.Canny(canny2,30,150,3)
#################################################################################
######### Detección contornos
contornos=[]
color_cont=(0,255,0)
(cnt, heirarchy)= cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(canny, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb,cnt, -1, color_cont, 2)
plt.imshow(rgb)
plt.title('Contornos')
plt.show()
######### contorno mayor
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
new.save('new.png',"png")
#### corta la seccion a encontrar circunferencia
#corte=new.crop((xmin, ymin, xmax, ymax))
#corte.save('corte.png', "png", quality=300)
#plt.imshow(corte)
#plt.title('Corte')
#plt.show()
#print(corte.size,'tamaño de imagen de contorno')
#xy=corte.size
#X=xy[0]
#Y=xy[1]
#############################################################################

X=xmax-xmin
Y=ymax-ymin
print(X,Y,'para limites del centro')
#cen_circ= cv2.imread('corte.png',0)
cen_circ= cv2.imread('new.png',0)
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
          param1=250,param2=20,minRadius=(round(X/2))-10,maxRadius=(round(X/2))+20)#80,20
          #param1=250,param2=20,minRadius=(round(Y/2))-10,maxRadius=(round(Y/2))+20)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    #dibuja fuera del circulo
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),1)
    #dibuja el centro del circulo
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),-1)

#cv2.imshow('detected circles',cimg)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
plt.imshow(cimg)
plt.show()









