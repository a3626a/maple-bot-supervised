# 데이터셋을 수집한다.
# 
# 다음과 같은 요소를 기록한다.
# - 화면을 1/10 으로 downscale 한 이미지
# - 버프 영역을 자른 이미지
# - keyboard 로 수집되는 입력
# - 각 방향키가 down 된 상태인지
#
# 사냥 행위만을 저장하기 위하여 A 키로 켜고 끌 수 있게 한다.
# 이 과정에서 로그가 출력되며 동시에 음성으로 어떤 상태인지 알려준다.
# 
import os
import time
import pickle
import threading

import numpy as np
import keyboard
from PIL import ImageGrab

from say import say

# A를 누르고 다음 A를 누르기까지 데이터를 주기적으로 수집합니다.
# PERIOD 주기마다 위치를 기록합니다.
PERIOD      = 0.1 
OUTPUT      = 'dataset/raw/06'
DOWNSCALE   = 5
TARGETS     = ['grind', 'collect']
DUMP_SIZE   = 3000 # 5분

def print_and_say(s):
    print(s)
    say(s)

def dump_async(key_events, features):
    def dump():
        with open(f'{OUTPUT}/{str(int(time.time()))}', 'wb') as f:
            pickle.dump([key_events, features], f)
    threading.Thread(target=dump).start()

def collect() :
    running = True
    target  = TARGETS[0]

    key_events = []
    def record(event):
        nonlocal target
        nonlocal running
        if event.name == '4' and event.event_type == 'down':
            print_and_say("사냥 녹화로 변경.")
            target = TARGETS[0]
        elif event.name == '5' and event.event_type == 'down' :
            print_and_say("수집 녹화로 변경.")
            target = TARGETS[1]
        elif event.name == '6' and event.event_type == 'down' :
            print_and_say("녹화 종료.")
            running = False
        else :
            key_events.append(event)
    keyboard.hook(record)

    features = []
    start = time.time()
    cnt = 0
    while running :
        cnt += 1

        recorded_at = time.time()
        image = ImageGrab.grab()
        
        down_img = image.resize((image.width//DOWNSCALE, image.height//DOWNSCALE))
        down_img = np.array(down_img)

        up = keyboard.is_pressed('up')
        down = keyboard.is_pressed('down')
        left = keyboard.is_pressed('left')
        right = keyboard.is_pressed('right')

        features.append(
            {
                'image': down_img,
                'recorded_at': recorded_at,
                'up': up,
                'down': down,
                'left': left,
                'right': right,
                'target': target
            }
        )

        # if len(features) >= DUMP_SIZE :
        #     dumping_features        = features[:DUMP_SIZE]
        #     features                = features[DUMP_SIZE:]
        #     dumping_key_events_len  = len(key_events)
        #     dumping_key_events      = key_events[:dumping_key_events_len]
        #     key_events              = key_events[dumping_key_events_len:]
        #     print_and_say("데이터를 저장합니다.")
        #     dump_async(dumping_key_events, dumping_features)

        to_wait = cnt*PERIOD - (time.time()-start)
        if to_wait > 0:
            time.sleep(to_wait)
        else :
            print_and_say("녹화가 밀리고 있습니다.")
    keyboard.unhook_all()

    return key_events, features

def main() :
    print_and_say("데이터 수집 프로그램을 가동합니다. 7 키를 눌러 수집을 시작해주십시오.")
    os.makedirs(OUTPUT, exist_ok=True)
    while True :
        keyboard.wait('7')
        print_and_say("사냥 녹화 시작.")

        key_events, features = collect()
        dump_async(key_events, features)
        print_and_say("데이터를 저장하고 수집을 종료했습니다.")

if __name__ == "__main__":
    main()