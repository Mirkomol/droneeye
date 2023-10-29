import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.PureWindowsPath
import pandas as pd


st.title("Military Drone Data Classification ")


st.write("""

#  UAV_EYE(1_Part(Classifcation))

    Project is called UAV_EYE(Not Segmented), Author:Mirkamol_Rakhimov ,  Github: 'https://www.github.com/Mirkomol

Drones themselves don't classify data; rather, they are equipped with sensors and cameras that can capture data.
In a military context, drones can be used for reconnaissance, surveillance, intelligence gathering, and even targeted strikes.
The data collected by military drones may include imagery, video footage, and other sensor data. 
This project was developed using deep learning technics, using computer vision model, it was used for UAV competiton Technofest_2024 , Turkey/Istanbul
""")


with st.sidebar:
        st.image('https://www.airforce-technology.com/wp-content/uploads/sites/6/2020/11/Feature-Image-Russias-top-long-range-attack-drones.jpg')
        st.title("Military Drone")
        st.subheader("Accurate detection of weapons, person and vehicle using deep learning. This helps an user to easily detect the weapons and person using UAV.")

st.write("""
         # Military Weapon Detection
         """
         )



def user_input_features():
    type = st.sidebar.selectbox('Category',('Weapon','Vehicle','Person'))

    data = {'Category':[Category]}

    features = pd.DataFrame(data)

    return features


input_df = user_input_features()


# widget for loading
st.info("Please upload photo of following categories, 'Weapon',\n 'Vehicle', \n 'People'")

genre = st.radio(
    "What's do you want classify",
    ["Weapon", "Vehicle", "People"])


if genre == 'Weapon':
    st.write('You selected weapon.')
elif genre == 'Vehicle':
    st.write('You selected Vehicle.')
elif genre == 'People':
    st.write('You selected People.')
else:
    st.write("You didn\'t select anything.")


file = st.file_uploader("Upload Photo",type = ['png','jpeg','gif','svg'])

if file:
#Conver Pil

    st.image(file)
    img = PILImage.create(file)

    model = load_learner('drone_data.pkl')

    # prediction


    pred, pred_id, probs = model.predict(img)
    st.success(f'Prediction: {pred}')
    st.info(f"Probability: {probs[pred_id]*100:.1f}%")

    st.markdown("Graph According To Prediction")
    fig = px.bar(x=probs*100,y=model.dls.vocab)
    st.plotly_chart(fig)


    #plot a mask
  
    if pred == 'Weapon':
        st.markdown("## Weapon")
        st.info("any instrument or device for use in attack or defense in combat, fighting, or war, as a sword, rifle, or cannon anything used against an opponent, adversary, or victim")

    elif pred == 'Vehicle':
        st.markdown("## Vehicle")
        st.info("a machine, usually with wheels and an engine, used for transporting people or goods, especially on land ")

    elif pred == 'Person':
        st.markdown("## Person")
        st.info("a human being regarded as an individual ")
