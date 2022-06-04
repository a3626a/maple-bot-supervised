import keyboard
import pickle
import time

keyboard.wait('a')

records = []

for i in range(5):
    with open('record'+str(i), 'rb') as f:
        records.append(pickle.load(f))

for r in records:
    keyboard.play(r)
    time.sleep(0.7)