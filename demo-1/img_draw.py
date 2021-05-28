from PIL import Image, ImageFilter, ImageDraw, ImageFont
import turtle
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL.ImageFilter import (BLUR, CONTOUR, DETAIL, EDGE_ENHANCE,
                             EDGE_ENHANCE_MORE, EMBOSS, FIND_EDGES,
                             SMOOTH, SMOOTH_MORE, SHARPEN)


imorig = Image.open("Ag_Nano1.bmp")
im = Image.open("Ag_Nano1.bmp").convert('RGBA')
### dibuja elipse
draw = ImageDraw.Draw(im)
draw.ellipse((320, 370, 620, 670), fill=None, outline=(255, 0, 0))
### dibuja rectangulo
draw.rectangle(
               (630, 350, 660, 375),
               fill=None,
               outline=(255, 255, 0))
draw.rectangle(
               (370, 75, 395, 100),
               fill=None,
               outline=(255, 255, 0))

### polygon
draw.polygon(
   ((630, 350), (590, 330), (600, 315)),
   fill=(0, 0, 255),
   outline=(0, 0, 0))

####### text
txt = Image.new('RGBA', im.size, (255,255,255,0))
d = ImageDraw.Draw(txt)
# draw text, full opacity
font = ImageFont.truetype(r'C:\Users\System-Pc\Desktop\arial.ttf', 30) 
d.text((100,100), "Muestra de plata",font=font, fill=(0,255,0,255))
font = ImageFont.truetype(r'C:\Users\System-Pc\Desktop\arial.ttf', 20)
d.text((490,310), "plano_crist",font=font, fill=(0,255,0,255))
out = Image.alpha_composite(im, txt)
#Show image
out.show()
