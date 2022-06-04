import time

import keyboard
from PIL import ImageGrab

import tensorflow as tf
import numpy as np
import say
import maple
from agent import Agent

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)

PERIOD = 0.1
NUM_BUFF = 20
DOWNSCALE = 15
IMG_WIDTH   = 64 # 1.6 sec

start = time.time()

agt = Agent(img_width=IMG_WIDTH)

say.say("로딩이 완료되었습니다.")
keyboard.wait('a')
say.say("시작합니다.")

cnt = 0
start = time.time()
prev_pred = 'none'
pressed = {
    'up': False,
    'down': False,
    'left': False,
    'right': False
}

def press_and_release(key):
    keyboard.press(key)
    time.sleep(0.02)
    keyboard.release(key)

while True:
    cnt += 1

    screen = ImageGrab.grab()
    if prev_pred != 'none' and prev_pred != '' and prev_pred != '[UNK]':
        name, event_type = prev_pred.split(' ')
        if name in ['up', 'down', 'left', 'right']:
            if event_type == 'up' :
                pressed[name] = False
                keyboard.release(name)
            else :
                pressed[name] = True
                keyboard.press(name)
        else :
            press_and_release(name)
    
    image = screen.resize((screen.width//DOWNSCALE, screen.height//DOWNSCALE))
    image = np.array(image)

    prev_pred = agt.run(image, pressed['up'], pressed['down'], pressed['left'], pressed['right'])
    if not prev_pred :
        prev_pred = 'none'
    
    print(f"[{cnt*0.1:.1f}]{(int(pressed['up']), int(pressed['down']), int(pressed['left']), int(pressed['right']))}->{prev_pred}")

    to_wait = 0.1*cnt - (time.time()-start)
    if to_wait > 0:
        time.sleep(to_wait)
    else :
        print(f"시간이 밀립니다. {(-1)*to_wait}")