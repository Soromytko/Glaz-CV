import streamlit as st
import numpy as np
import pandas as pd
import cv2
from io import StringIO
import shutil
import glob
from pathlib import Path
from joblib import dump, load
from init_database import encode_to_vector
from init_database import create_model
from sklearn.neighbors import NearestNeighbors

from PIL import Image, ImageOps
import torch
import open_clip

FILE_FORMATS = ['jpg', 'png', 'jpeg']

if __name__ == '__main__':
    uploaded_file = st.file_uploader("Choose a image file", type=FILE_FORMATS)
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        img = ImageOps.exif_transpose(img)
        
        db = pd.read_pickle('db_net.csv')
        nbrs = NearestNeighbors(n_neighbors=5, metric='cosine', n_jobs=-1)
        nbrs.fit(np.stack(db['vectors'].to_numpy()))
        neighbours = nbrs

        #model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e32')
        model, val, preprocess = create_model()
        
        vector = None
        with torch.no_grad():
            image = preprocess(img).unsqueeze(0)
            vector = encode_to_vector(image, model)
            
        if vector is not None:
            indices = neighbours.kneighbors(vector, return_distance=False)[0]
            paths = np.hstack(db.loc[indices, ['paths']].values)

            for path in paths:
                st.image(path, caption=path)
        else:
            print('vector is None!', vector)    
    else:
        st.write("Make sure you image is in JPG/PNG Format.")

 
    
