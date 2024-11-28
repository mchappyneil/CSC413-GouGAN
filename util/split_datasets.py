import os
import shutil
from random import randint


DATAPATH = "data/"
SUBSET_PATH = "subset/"

TRAIN_COUNTER = 0
VAL_COUNTER = 0
TEST_COUNTER = 0

def flatten(dir_name, src):
    if src[-3:] == 'jpg':
        head, tail = os.path.split(src)
        os.rename(src, 'data/' + dir_name + '/' + tail)
        return None
    elif os.path.isfile(src) and not (src[-3:] == 'jpg'):
        return None
    else:
        dirs = os.listdir(src)
        for d in dirs:
            flatten(dir_name, src + '/' + d)
        return None

def grab_random(dir, train, val, test):
    global TRAIN_COUNTER
    global VAL_COUNTER
    global TEST_COUNTER

    # if not os.path.isdir(SUBSET_PATH + dir):
    #     os.mkdir(SUBSET_PATH + dir)
    #     print(SUBSET_PATH + dir)

    gen_files = set(os.listdir(DATAPATH + dir))

    for _ in range(train):
        rand = gen_files.pop()
        shutil.copyfile(DATAPATH + dir + '/' + rand, SUBSET_PATH + 'train/' + str(TRAIN_COUNTER) + '.jpg')
        TRAIN_COUNTER += 1

    for _ in range(val):
        rand = gen_files.pop()
        shutil.copyfile(DATAPATH + dir + '/' + rand, SUBSET_PATH + 'val/' + str(VAL_COUNTER) + '.jpg')
        VAL_COUNTER += 1

    for _ in range(test):
        rand = gen_files.pop()
        shutil.copyfile(DATAPATH + dir + '/' + rand, SUBSET_PATH + 'test/' + str(TEST_COUNTER) + '.jpg')
        TEST_COUNTER += 1

if __name__ == "__main__":
    # files = os.listdir('assets/')

    # for f in files:
    #     # os.mkdir('data/' + f)
    #     print('starting', f)
    #     flatten(f, 'assets/' + f)
    #     print('finished', f)

    dirs = os.listdir(DATAPATH)

    if not os.path.isdir(SUBSET_PATH):
        os.mkdir(SUBSET_PATH)
        os.mkdir(SUBSET_PATH + 'train/')
        os.mkdir(SUBSET_PATH + 'val/')
        os.mkdir(SUBSET_PATH + 'test/')


    for dir in dirs:
        print(dir + " in progress...")
        if dir != ".DS_Store":
            grab_random(dir, train=300, val=100, test=100)
        print(dir + " done!")