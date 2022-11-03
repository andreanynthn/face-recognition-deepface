import os
import string
import numpy as np
from PIL import Image
import streamlit as st
from deepface import DeepFace



def main():
    st.title("Face Recognition")
    image_file = st.file_uploader(
        "Upload image", type = ["jpg", "jpeg"]
    )


    wait = 0

    # update every 10 minutes
    # if wait % 600000 == 0:
    #     img = np.array(image)
    #     if 'representations_facenet512.pkl' in os.listdir('database'):
    #         os.remove('database/representations_facenet512.pkl')
    #         DeepFace.find(img, os.path.join(os.getcwd(), "database"), enforce_detection = False,
    #                       detector_backend='ssd', model_name = 'Facenet512')

    if image_file is not None:
        image = Image.open(image_file)

        if st.button("Process"):

            # process image

            img = np.array(image)

            predict = DeepFace.find(img, os.path.join(os.getcwd(), "database"), enforce_detection = False,
                            detector_backend='ssd', model_name = 'Facenet512')


            if predict.empty == False:
                # name = names[face_id]
                name = predict['identity'][0].split('/')[-1].split('.')[1]
                st.success("Face found!")
                st.markdown(f'<h2 style="text-align:center">{string.capwords(name)}</h2>', unsafe_allow_html=True)
                st.image(img)
            else:
                st.error("Face not found!")
    # wait += 1000

if __name__ == '__main__':
    main()
