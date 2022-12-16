# import streamlit as st
# import numpy as np

# dataframe = np.random.randn(10, 20)
# st.dataframe(dataframe)

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from io import StringIO

FILE_FORMATS = ['jpg', 'png', 'jpeg']

uploaded_file = st.file_uploader("Choose a image file", type=FILE_FORMATS)
if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
    st.write("Loaded image:")
    st.image(opencv_image, channels="BGR")
else:
    st.write("Make sure you image is in JPG/PNG Format.")
    
    
    
    
    
    