import os
import shutil
from glob import glob
from pathlib import Path
import numpy as np
import cv2 as cv
import pandas as pd
from sklearn.cluster import KMeans
from joblib import dump, load
import time
import pickle
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

def get_files(root, types):
    res : list = []
    for type in types:
        res += glob(root + '*' + type)
    return res
    
def calculate_descriptors(img_path, save_path):
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    
    # Draw Keypoints and save img
    img = cv.drawKeypoints(gray, kp, img)
    file_name = Path(img_path).stem
    cv.imwrite(save_path + file_name + '.jpg', img)
    
    return des
    
def train(data):
    print('training...', data.shape)
    model = KMeans(n_clusters=2048, n_init=10).fit(data)
    #model = MiniBatchKMeans(n_clusters=2048, n_init='auto', verbose=1)
    model.fit(data)
    print('training completed!')
    return model


if __name__ == '__main__':
    img_paths = get_files('training_data_set/', ['.jpeg', 'jpg'])
    
    if os.path.exists('sift_keypoints'):
        shutil.rmtree('sift_keypoints')
    os.mkdir('sift_keypoints')

    descriptors = pd.DataFrame(columns=range(128), index=range(0))
    for img_path in tqdm(img_paths, desc='image processing'):
        des = calculate_descriptors(img_path, 'sift_keypoints/')
        des = pd.DataFrame(des)
        descriptors = pd.concat([descriptors, des])
    
    start_training_time = time.time()
    model = train(descriptors)
    training_time = time.time() - start_training_time
    dump(model, 'model.joblib')
    print('the model is saved as \"model.joblib\"')
    print('training time: {} seconds'.format(training_time))
    # clf = load('filename.joblib')
    
    # s = pickle.dumps(model)
    # print(s)
   







def df_test():
    df1 = pd.DataFrame({'a':range(6),
                    'b':[5,3,6,9,2,4]}, index=list('abcdef'))

    df2 = pd.DataFrame({'a':range(4),
                        'b':[10,20,30, 40]}, index=list('abhi'))


    print('df1', df1.shape)
    print('df2', df2.shape)
    print('df3', pd.concat([df1, df2]).shape)
