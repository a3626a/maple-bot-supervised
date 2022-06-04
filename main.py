import keyboard
import pickle
import time
import random

from say import say
import maple

RECORDS = []
for i in range(5):
    with open('record'+str(i), 'rb') as f:
        RECORDS.append(pickle.load(f))

PICK = None
with open('pick0', 'rb') as f:
    PICK = pickle.load(f)

BUFF = None
with open('buff0', 'rb') as f:
    BUFF = pickle.load(f)

PREV_PICK = 0
def check_and_pick():
    global PREV_PICK
    current = time.time()
    left = 110 - (current-PREV_PICK)
    if left <= 0 :
        print("PICK")
        keyboard.play(PICK)
        PREV_PICK = current
        time.sleep(0.5)
    else :
        print("PICK TIME LEFT : "+str(left))

PREV_BUFF = 0
def check_and_buff():
    global PREV_BUFF
    current = time.time()
    left = 230 - (current-PREV_BUFF)
    if left <= 0 :
        print("BUFF")
        keyboard.play(BUFF)
        PREV_BUFF = current
        time.sleep(0.5)
    else :  
        print("BUFF TIME LEFT : "+str(left))


INITIAL_LEFT_WARN = 0
PREV_WARN = time.time() - 60*(15-INITIAL_LEFT_WARN)
def check_and_warn():
    global PREV_WARN
    current = time.time()
    left = 950 - (current-PREV_WARN)
    if left <= 0 :
        print("RUNE")
        say("룬을 확인해주세요.")
        keyboard.wait('a')
        PREV_WARN = current
    else :
        print("RUNE TIME LEFT : "+str(left))

def loop():
    i = random.randint(0, len(RECORDS)-1)
    r = RECORDS[i]
    print("Play record"+str(i))
    keyboard.play(r)

IS_STOPPED = False
def on_F8_press(_):
    say("정지합니다.")
    global IS_STOPPED
    IS_STOPPED = not IS_STOPPED

def recovery():
    if maple.getPos() != (16, 85) :
        say("예외 상황이 발생하여 복원을 시도합니다.")
        while maple.getPos() != (16, 85):
            keyboard.press('down arrow')
            time.sleep(0.1)
            keyboard.press_and_release('d')
            time.sleep(0.1)
            keyboard.release('down arrow')
            time.sleep(1)

say("자동 사냥 시스템을 가동합니다. F7로 시작할 수 있고 F8로 일시정지합니다.")

keyboard.on_press_key('F8', on_F8_press)
keyboard.wait('F7')

while True:
    if IS_STOPPED:
        keyboard.wait('F8')

    time.sleep(0.2)
    recovery()
    check_and_pick()
    check_and_buff()
    check_and_warn()
    loop()