import time
import pickle

import matplotlib.pyplot as plt
from matplotlib import animation, rc
import cv2

DOWNSCALE   = 15

with open('dataset/raw/06/1609768411', 'rb') as f:
    key_events, features = pickle.load(f)

# for key_event in key_events :
#     print(f'Keyboard Event')
#     print(f'::time:         {key_event.time}')
#     print(f'::event_type:   {key_event.event_type}')
#     print(f'::name:         {key_event.name}')
#     print(f'::modifiers:    {key_event.modifiers}')
#     print(f'::scan_code:    {key_event.scan_code}')

start = time.time()
cnt = 0
for feature in features :
    cnt += 1
    img = feature['image']
    img = cv2.resize(img, dsize=(1920//DOWNSCALE, 1080//DOWNSCALE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('image', img)
    print(f"up    : {feature['up']}")
    print(f"down  : {feature['down']}")
    print(f"left  : {feature['left']}")
    print(f"right : {feature['right']}")
    to_wait = cnt*100-1000*(time.time()-start)
    if to_wait > 0:
        cv2.waitKey(int(to_wait))

# window = 3
# for i in range(len(labels)-(window-1)):
#     rdown = [labels[j] == 'r down' for j in range(i, i+window)]
    
#     if np.array(rdown).all() :
#         print(i)
#         print(labels[i-KEY_WIDTH:i+KEY_WIDTH])