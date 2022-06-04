# 입력되는 데이터는 pickle 로 직렬화되었으며
# [key_events, images]
# key_event = [keyboard.KeyEvent]
# images = [{
#   'image': np.array(108, 192, 3),
#   'recorded_at': Float
# }]
# 목표하는 형태는 다음과 같다.
#
# {OUTPUT_DIR}/XX/{label}/{file}
#   tf.Tensor(shape=(FILE_BATCH, WIDTH, 108, 192, 3), dtype=tf.float16)
#   0.5 초 간격으로 촬영된 이미지 4개
#
# PARAMETERS
# INPUT_DIR   : 입력 RAW 데이터가 저장된 디렉토리이다.
# OUTPUT_DIR  : 출력 RAW 데이터가 저장된 디렉토리이다.
# FILE_BATCH  : 한 파일에 몇 개의 Example 을 저장할 지 결정한다. 한 파일의 크기가 100MB 내외임이 적당하다.
#               현재 자료 구조에서는, FILE_BATCH * 4 * 108 * 192 * 3 * 2B = 100MB 이므로, FILE_BATCH ~= 200 이다.
# LABELS      : 학습에 사용할 Label 의 목록이다. 이외의 입력은 무시된다.
# WIDTH       : 
# SAMPLE_RATE : 
import os
import numpy as np
import time
import pickle
import multiprocessing
from multiprocessing import Process
from multiprocessing import Queue

import tensorflow as tf

# PARAMETERS
INPUT_DIR   = 'dataset/raw/06'
OUTPUT_DIR  = 'dataset/preprocessed/14'
FILE_BATCH  = 1000
LABELS      = ['d down', 'down down', 'down up', 'f down', 'left down', 'left up', 'r down', 'right down', 'right up', 'up down', 'up up']
DOWNSCALE   = 15

# CONSTANTS
def make_list_from_record(key_events, features) :
    dropped = 0
    shifted = 0
    shifted_cnt = 0
    shifted_max = 0
    ignored = 0
    labels = ['none'] * len(features)

    period = (features[-1]['recorded_at']-features[0]['recorded_at'])/(len(features)-1)
    recorded_at_start = features[0]['recorded_at']
    recorded_at_end = features[-1]['recorded_at'] + period

    down_time = {}
    for key_idx, key_event in enumerate(key_events): 
        if key_event.time < recorded_at_start:
            continue

        if key_event.time >= recorded_at_end:
            break

        # 키를 꾹누르고 있는 경우 down 이벤트가 빠르게 반복 발생한다.
        # 따라서 up 없이 down 이 0.1초 이하 간격으로 반복되는 경우 모두 무시한다.
        if key_event.event_type == 'up' :
            down_time[key_event.name] = None
        else :
            if key_event.name in down_time :
                if down_time[key_event.name] :
                    if key_event.time - down_time[key_event.name] < 0.1:
                        down_time[key_event.name] = key_event.time
                        ignored+= 1
                        continue
            down_time[key_event.name] = key_event.time


        label = f'{key_event.name} {key_event.event_type}'
        if label not in LABELS :
            continue

        idx = int((key_event.time - recorded_at_start)/period)
        # 이미 같은 시간 범위에 값이 있는 경우
        # 다음 idx 에 입력합니다.
        idx_prev = idx
        try :
            idx = next(i+idx for i,v in enumerate(labels[idx:]) if v == 'none')
            shifted += (idx-idx_prev)
            shifted_cnt += 1
            shifted_max = max((idx-idx_prev), shifted_max)
            labels[idx] = label
            # if (idx-idx_prev) > 3 :
            #     print(key_events[key_idx+1-(idx-idx_prev):key_idx+1])
            #     print(labels[idx_prev:idx+1])
        except StopIteration:
            # 만약 빈 공간이 존재하지 않으면 이 이벤트는 건너뜁니다.
            dropped += 1
            break

    # Visualization for strange data
    # window = 3
    # for i in range(len(labels)-(window-1)):
    #     rdown = [labels[j] == 'r down' for j in range(i, i+window)]
        
    #     if np.array(rdown).all() :
    #         for j in range(i-KEY_WIDTH, i+KEY_WIDTH) :
    #             cv2.imshow('image', cv2.cvtColor(features[j]['image'], cv2.COLOR_BGR2RGB))
    #             cv2.waitKey(100)

    features = [{ 'image': f['image'], 'up': f['up'], 'down':f['down'], 'left':f['left'], 'right':f['right'], 'target':f['target'] } for f in features]
    print(f'Dropped : {dropped}')
    print(f'Shifted Average : {shifted/shifted_cnt}')
    print(f'Shifted Maximum : {shifted_max}')
    print(f'Ignored : {ignored}')
    
    assert len(features) == len(labels)
    return features, labels

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def export(features, labels) :
    images = [f['image'] for f in features]
    ups = [f['up'] for f in features]
    downs = [f['down'] for f in features]
    lefts = [f['left'] for f in features]
    rights = [f['right'] for f in features]
    targets = [f['target'] for f in features]

    images = tf.convert_to_tensor(np.asarray(images), dtype=tf.float16)
    images = tf.image.resize(images, (1080//DOWNSCALE, 1920//DOWNSCALE))
    images = tf.cast(images, dtype=tf.float16)
    ups = tf.convert_to_tensor(np.asarray(ups), dtype=tf.float16)
    downs = tf.convert_to_tensor(np.asarray(downs), dtype=tf.float16)
    lefts = tf.convert_to_tensor(np.asarray(lefts), dtype=tf.float16)
    rights = tf.convert_to_tensor(np.asarray(rights), dtype=tf.float16)
    targets = tf.convert_to_tensor(np.asarray(targets), dtype=tf.string)
    labels = tf.convert_to_tensor(np.asarray(labels), dtype=tf.string)

    serialized_images = tf.io.serialize_tensor(images)
    serialized_ups = tf.io.serialize_tensor(ups)
    serialized_downs = tf.io.serialize_tensor(downs)
    serialized_lefts = tf.io.serialize_tensor(lefts)
    serialized_rights = tf.io.serialize_tensor(rights)
    serialized_targets = tf.io.serialize_tensor(targets)
    serialized_labels = tf.io.serialize_tensor(labels)

    features = {
        'images': _bytes_feature(serialized_images),
        'ups': _bytes_feature(serialized_ups),
        'downs': _bytes_feature(serialized_downs),
        'lefts': _bytes_feature(serialized_lefts),
        'rights': _bytes_feature(serialized_rights),
        'targets': _bytes_feature(serialized_targets),
        'labels': _bytes_feature(serialized_labels),
    }

    features = tf.train.Example(features=tf.train.Features(feature=features))
    serialized_features = features.SerializeToString()
    os.makedirs(f'{OUTPUT_DIR}', exist_ok=True)
    tf.io.write_file(f'{OUTPUT_DIR}/{str(int(time.time()*1000))}', serialized_features)

def load_datum(path):
    with open(f'{INPUT_DIR}/{path}', 'rb') as f:
        result = pickle.load(f)
    return result

def mp_produce(q, paths) :
    stack_merged = {}
    for path in paths :
        key_events, features = load_datum(path)
        features, labels = make_list_from_record(key_events, features)

        for i in range((len(features)-1)//FILE_BATCH+1) :
            q.put((features[i*FILE_BATCH:(i+1)*FILE_BATCH], labels[i*FILE_BATCH:(i+1)*FILE_BATCH]))

    # End sign
    q.put(None)

def mp_export(q):
    while True :
        none_or_data = q.get()
        if none_or_data is None :
            q.put(None)
            break
        
        features, labels = none_or_data
        export(features, labels)

def preprocess_and_export(paths) :
    queue = Queue(10)
    prod = Process(target=mp_produce, args=(queue, paths))
    consumers = [Process(target=mp_export, args=(queue, )) for _ in range(2)]

    prod.start()
    [c.start() for c in consumers]
    
    prod.join()
    [c.join() for c in consumers]

def main():
    paths = os.listdir(INPUT_DIR)
    preprocess_and_export(paths)

if __name__ == '__main__':
    main()