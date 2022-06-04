from mykeyboard import PressKey, ReleaseKey
import random
import time

def sleep(t) :
    rand = 2*(random.random()-0.5)
    factor = 1+0.05*rand
    time.sleep(t*factor)

def press(key) :
    PressKey(key)
    sleep(0.04)
    ReleaseKey(key)
