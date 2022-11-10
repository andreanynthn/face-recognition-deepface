import mysql
import string
import pymysql
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import mysql.connector
from deta import Deta
from sqlalchemy import text
from deepface import DeepFace
from sqlalchemy import create_engine
from deepface.commons import functions



# setup database
# def init_connection():
#     return mysql.connector.connect(**st.secrets["mysql"])
#
# db = init_connection()
# cursor = db.cursor()
# engine = create_engine("mysql://root:""@localhost:3306/image", pool_pre_ping=True)
# cursor = engine.connect()
deta = Deta(st.secrets["project_key"])
db = deta.Base("Image")


# model
model = DeepFace.build_model("Facenet512")

# for face registration
# @st.cache(allow_output_mutation=True)
def inputImage(image, name):
    #     workdir = os.getcwd()
    facial_img = functions.preprocess_face(image, target_size = (160, 160), detector_backend = 'ssd')

    # embedding
    embedding = model.predict(facial_img)[0]

    # face name
    img_name = name
    key = name
    insert_query = "INSERT INTO face_image (face_name, embedding) VALUES (:face_name, :embedding)"
    db.put({
        "img_name" : name,
        "embedding" : embedding.tolist()
    }, key=name)


# retrieve image
def retrieveImage():
    retrieve = db.fetch()
    results = retrieve.items

    instances = []
    for i in range(len(results)):
        img_name = results[i]["img_name"]
        embedding_bytes = np.array(results[i]["embedding"])
        embedding = np.array(embedding_bytes)

        instance = []
        instance.append(img_name)
        instance.append(embedding)
        instances.append(instance)

    result_df = pd.DataFrame(instances, columns = ["img_name", "embedding"])

    return result_df


# cosine distance
def findCosineDistance(df):
    vector_1 = df['embedding']
    vector_2 = df['target']
    a = np.matmul(np.transpose(vector_1), vector_2)

    b = np.matmul(np.transpose(vector_1), vector_1)
    c = np.matmul(np.transpose(vector_2), vector_2)

    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


# face recognition
def faceRecognition(image):
    image_preprocess = functions.preprocess_face(image, target_size = (160, 160), detector_backend='ssd')
    image_target = model.predict(image_preprocess)[0].tolist()
    result = retrieveImage()
    image_target_duplicated = np.array([image_target]*result.shape[0])
    result['target'] = image_target_duplicated.tolist()

    # calculate distance and store to result_df
    result['distance'] = result.apply(findCosineDistance, axis = 1)
    result = result.sort_values(by = ['distance']).reset_index(drop = True)
    result = result.drop(columns = ["embedding", "target"])

    # get name
    name = result[result['distance'] == min(result['distance'])]['img_name']
    name = name.values[0].split('.')[0]
    # print(min(result['distance']))

    return name, min(result['distance'])


def main():

    st.title("Face Recognition")

    menus = ["Face Registration", "Face Recognition"]
    select = st.sidebar.selectbox("Menu", menus)

    if select == "Face Registration":
        st.subheader("Face Registration")

        image_file = st.file_uploader("Upload image", type = ["jpg", "jpeg"])
        username = st.text_input("Insert name")

        if image_file is not None:
            image = Image.open(image_file)

            if st.button("Process"):
                img = np.array(image)
                inputImage(img, username)
                st.success("Face registered successfully!")


    if select == "Face Recognition":
        st.subheader("Face Recognition")
        image_file = st.file_uploader(
            "Upload image", type = ["jpg", "jpeg"]
        )

        if image_file is not None:
            image = Image.open(image_file)
            if st.button("Process"):
                img = np.array(image)
                try:
                    face_detection = DeepFace.detectFace(img_path = img,
                                                         target_size = (224, 224),
                                                         detector_backend = 'ssd'
                                                         )
                except:
                    st.error("Face not detected!")
                else:
                    st.success("Face Detected!")
                    predict = faceRecognition(img)[0]

                    if predict is not None:
                        st.success("Face is successfully recognized.")
                        st.markdown(f'<h2 style="text-align:center">{string.capwords(predict)}</h2>', unsafe_allow_html=True)
                        st.text(faceRecognition(img)[1])
                        st.image(img)
                    else:
                        st.error("Face not registered.")

if __name__ == '__main__':
    main()
