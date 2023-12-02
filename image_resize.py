import cv2
import os
from PIL import Image
import numpy as np

    
directory = 'train/'
prefix = '0'
extension = 'jpg'

for i, filename in enumerate(os.listdir(directory)):
    img = cv2.imread (f'{directory}/{filename}')
    resized = cv2.resize(img, (600, 400))
    print(resized.shape)
    cv2.imwrite(directory+"/"+filename,resized)