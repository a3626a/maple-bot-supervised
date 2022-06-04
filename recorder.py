import keyboard
import time
import pickle

# Record events until 'esc' is pressed.
NUM = 5
NAME = 'record'
keyboard.wait('a')
for i in range(NUM):
    recorded = keyboard.record(until='a')
    with open(NAME+str(i), 'wb') as f:
        pickle.dump(recorded,f)