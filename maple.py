# maple 모듈은 정말 메이플이라서 필요한 기능을 저장한다.
import numpy as np
import cv2
from PIL import ImageGrab

character = cv2.imread('char.png')
w, h = character.shape[1::-1]

# (x, y) 튜플을 반환한다.
def getPos():
    printscreen = np.array(ImageGrab.grab(bbox=(0,0,300,300)).convert('RGB'))
    printscreen = cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB)
    res = cv2.matchTemplate(printscreen,character,cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return min_loc
