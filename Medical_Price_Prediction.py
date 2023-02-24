# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 10:00:25 2023

@author: Asus
"""

import numpy as np 
import pickle 
import streamlit as st

loaded_model = pickle.load(open(r"C:\Users\Asus\Downloads\Cost_pred.sav","rb"))

def health_insurance_cost(input_data):
    input_data_as_np_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_np_array.reshape(1,-1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    return prediction
    

def main():
    st.title("Medical Insurance Cost")
    
    # getting inputs
    
    age = st.text_input('Age')
    sex = st.text_input('Sex (1->Male, 0->Female)')
    bmi = st.text_input("BMI")
    children = st.text_input("No of children")
    region = st.text_input("Region( 0->northwest, 1->northeast, 2->southeast,3->southwest")
    smoker = st.text_input("Smoker or not (0->non smoker, 1 -> non smoker")
 
    # Prediction 
    
    Charges = ""
    
    if st.button("Test Results : "):
        Charges = health_insurance_cost([age,sex,bmi,children,region,smoker])
    
    st.success(Charges)

if __name__ == "__main__":
    main()
