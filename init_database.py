import os
import shutil
from glob import glob
from pathlib import Path
import numpy as np
import cv2 as cv
import pandas as pd
from sklearn.cluster import KMeans
from joblib import dump, load
import pickle
from tqdm import tqdm


MODEL = load('model2048')
MODEL = load('model.joblib')


def encode_to_vector(gray_img, model):
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(gray_img, None)
    if des is None:
        return None
    classes = model.predict(des)
    hist = np.histogram(classes, model.n_clusters, density=True)
    if hist is None:
        return None
    return hist[0]

	
if __name__ == '__main__':
    print(MODEL)
    paths : list = []
    vectors : list = []
    for img_path in tqdm(glob('image_database/*'), desc='database image processing'):
        img = cv.imread(img_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        vector = encode_to_vector(gray, MODEL)
        if vector is None:
            continue
        paths.append(img_path)
        vectors.append(vector)
    db = pd.DataFrame({'paths' : paths, 'vectors' : vectors})
    db.to_pickle('db.csv')
    print('Done!')
        
    
