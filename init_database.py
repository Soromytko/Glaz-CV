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
from PIL import Image, ImageOps
import torch
import open_clip


MODEL, _, PREPROCESS = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e32')

def create_model():
    model, val, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e32')
    return model, val, preprocess


def encode_to_vector(gray_img, model):
    return model.encode_image(gray_img).cpu().detach().numpy()

	
if __name__ == '__main__':
    #print(MODEL)
    paths = []
    vectors = []
    with torch.no_grad():
        for img_path in tqdm(glob('image_database/*'), desc='database image processing'):
            img = PREPROCESS(Image.open(img_path)).unsqueeze(0)
            vector = encode_to_vector(img, MODEL)[0]
            if vector is None:
                continue
            paths.append(img_path)
            vectors.append(vector)
            
    db = pd.DataFrame({'paths' : paths, 'vectors' : vectors})
    db.to_pickle('db_net.csv')
    print('Done!')
        
    
