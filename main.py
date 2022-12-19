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
from sklearn.neighbors import NearestNeighbors


FILE_FORMATS = ['jpg', 'png', 'jpeg']


if __name__ == '__main__':
    uploaded_file = st.file_uploader("Choose a image file", type=FILE_FORMATS)
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
      
        # Now do something with the image! For example, let's display it:
        # st.write("Loaded image:")
        st.image(opencv_image, channels="BGR", caption='Loaded image')


        db = pd.read_pickle('db.csv')
        model = load('model.joblib')
        
        db = pd.read_pickle('db2048.csv')
        model = load('model2048')
        
        # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        neighbours = NearestNeighbors(n_neighbors=5, metric='cosine', n_jobs=-1)
        neighbours.fit(np.stack(db['vectors'].to_numpy()))
        vector = encode_to_vector(gray, model)
        indices = neighbours.kneighbors(vector.reshape(1, -1), return_distance=False)[0]
        paths = np.hstack(db.loc[indices, ['paths']].values)
        for path in paths:
            #path = path.replace('\\', '/') #for Linux
            print(path)
            st.image(path, caption=path)
    else:
        st.write("Make sure you image is in JPG/PNG Format.")

 
    
