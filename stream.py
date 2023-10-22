import pickle
import streamlit as st 
import setuptools
from PIL import Image


# membaca model
ins_model = pickle.load(open('estimasi_harga_asuransi.sav','rb'))
image = Image.open('banner.jpeg')

#judul web
st.image(image, caption='')
st.title('Aplikasi Prediksi Harga Asuransi')

col1, col2,col3=st.columns(3)
with col1:
    Age = st.number_input('Input Umur :')
with col2:
    Diabetes  = st.number_input('Input Riwayat Diabetes :')
with col3:
    BloodPressureProblems  = st.number_input('Riwayat Penyakit Tekanan Darah :')
with col1:
    AnyTransplants = st.number_input('Input Riwayat Operasi transplasi :')
with col2:
    AnyChronicDiseases = st.number_input('Input Riwayat Penyakit Kronis :')
with col3:
    Height = st.number_input('Input Tinggi :')
with col1:
    Weight = st.number_input('Input Berat Badan :')
with col2:
    KnownAllergies = st.number_input('Input Riwayat Alergi :')
with col3:
    HistoryOfCancerInFamily = st.number_input('Riwayat Kanker(Pribadi/Keluarga) :')
with col1:
    NumberOfMajorSurgeries = st.number_input('Input riwayat operasi besar :')

#code untuk estimasi
ins_est=''

#membuat button
with col1:
    if st.button('Estimasi Harga'):
        ins_pred = ins_model.predict([[Age,Diabetes,BloodPressureProblems,AnyTransplants,AnyChronicDiseases,Height,Weight,KnownAllergies,HistoryOfCancerInFamily,NumberOfMajorSurgeries]])

        st.success(f'Estimasi Harga : {ins_pred[0]:.2f}')