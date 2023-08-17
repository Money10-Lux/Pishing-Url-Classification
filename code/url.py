import streamlit as st
import pandas as pd
import os
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew
from keras.layers import *
import altair as alt
from PIL import Image
import plotly.express as px #for visualization
import pickle
from streamlit_pandas_profiling import st_profile_report  #render the profile report insisde the streamlit app
from pycaret.classification import setup, compare_models, pull, save_model, load_model #ML libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc,roc_curve
from sklearn.metrics import mean_squared_error,confusion_matrix
from streamlit_option_menu import option_menu
from math import sqrt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential

#body
# page configuration to wide
st.set_page_config(layout="wide") #use full page instead of a narrow centrall column
st.title('Automated Cyber Security Tool (URL Phishing)') #Titl of the page
# information of the app
st.info("This is a Web App to Allow Exploration of various URL using Streamlit, Pandas, pyCaret, numpy, tensorflow among other python libraries.")
    
#APP SIDEBAR
with st.sidebar:
    st.title("Cyber Security")
    # st.image("Phish.jpg")
    #st.image("learn.jpg")
    choice = st.radio("NAVIGATION", ["Prediction of URLs"])

#PREDICTING DATA
if choice == "Prediction of URLs":
    filename = 'model2_final.sav'
    #loading and reading the saved model
    if os.path.exists(filename): #if the file exists
        st.success("Model Exists")
        best_model = pickle.load(open(filename, 'rb'))
        
        if best_model != 0:
            st.success("and the Model can be accessed")
            st.write(best_model)
        else:
            st.warning("and the Model can NOT be Red! Check a proper way to read the Model")
    
    else:
        st.warning("Model Does Not Exists")

    def main():
        
        #Setting Application title
        st.title('URL Link Phishing Prediction')
        st.markdown("<h3></h3>", unsafe_allow_html=True)

        #Setting Application sidebar default
        add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", ("Online", "Batch"))
        # st.sidebar.image("Phish1.jpg")
        st.sidebar.info('This App Can Predict a Single URL or Batch of URLs')

        if add_selectbox == "Online":
            #with col1:
            st.subheader("Input The URL Below:")
            URL = st.text_input( label = "Enter URL", value="", max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")
            URL2 = {'url': URL}    
            #pd.DataFrame()
            df2 = pd.DataFrame(URL2, index=[1])
            #df2 = pd.DataFrame(list(URL), index=URL.keys())
            
            if st.button('Submit for Feature Extraction and Prediction'):
                st.subheader('URL Summary' )
                st.write("Enterd url: ", df2)

                import urllib
                from urllib.parse import urlparse

                # Code adapted from https://stackoverflow.com/questions/48927719/parse-split-urls-in-a-pandas-dataframe-using-urllib
                #separate the URL in different parts
                urls = [URL2 for URL2 in df2['url']]
                df2['protocol'],df2['domain'],df2['path'],df2['query'],df2['fragment'] = zip(*[urllib.parse.urlsplit(x) for x in urls])
                st.write("URL and its Parts : ", df2)
                #function to get URL feature from its parts 
                def get_features(df2):
                    needed_cols = ['url', 'domain', 'path', 'query', 'fragment']
                    for col in needed_cols:
                        df2[f'{col}_length']=df2[col].str.len()
                        df2[f'qty_dot_{col}'] = df2[[col]].applymap(lambda x: str.count(x, '.'))
                        df2[f'qty_hyphen_{col}'] = df2[[col]].applymap(lambda x: str.count(x, '-'))
                        df2[f'qty_slash_{col}'] = df2[[col]].applymap(lambda x: str.count(x, '/'))
                        df2[f'qty_questionmark_{col}'] = df2[[col]].applymap(lambda x: str.count(x, '?'))
                        df2[f'qty_equal_{col}'] = df2[[col]].applymap(lambda x: str.count(x, '='))
                        df2[f'qty_at_{col}'] = df2[[col]].applymap(lambda x: str.count(x, '@'))
                        df2[f'qty_and_{col}'] = df2[[col]].applymap(lambda x: str.count(x, '&'))
                        df2[f'qty_exclamation_{col}'] = df2[[col]].applymap(lambda x: str.count(x, '!'))
                        df2[f'qty_space_{col}'] = df2[[col]].applymap(lambda x: str.count(x, ' '))
                        df2[f'qty_tilde_{col}'] = df2[[col]].applymap(lambda x: str.count(x, '~'))
                        df2[f'qty_comma_{col}'] = df2[[col]].applymap(lambda x: str.count(x, ','))
                        df2[f'qty_plus_{col}'] = df2[[col]].applymap(lambda x: str.count(x, '+'))
                        df2[f'qty_asterisk_{col}'] = df2[[col]].applymap(lambda x: str.count(x, '*'))
                        df2[f'qty_hashtag_{col}'] = df2[[col]].applymap(lambda x: str.count(x, '#'))
                        df2[f'qty_dollar_{col}'] = df2[[col]].applymap(lambda x: str.count(x, '$'))
                        df2[f'qty_percent_{col}'] = df2[[col]].applymap(lambda x: str.count(x, '%'))
                
                # Applying function to extract features
                get_features(df2)

                # printing data with extracted feature
                st.write("URL, its' Parts and Generated Feature: ", df2)

                #then we drop the URL and its PARTS and reataining Features except features with null correlation 
                df3 = df2.drop(columns=['url', 'protocol', 'domain', 'path', 'query', 'fragment']) 
                null_feat_corr= ['qty_slash_domain', 'qty_questionmark_domain','qty_equal_domain', 'qty_at_domain',
                                'qty_and_domain', 'qty_exclamation_domain', 'qty_space_domain', 'qty_tilde_domain',
                                'qty_comma_domain', 'qty_plus_domain', 'qty_asterisk_domain','qty_hashtag_domain',
                                'qty_dollar_domain', 'qty_percent_domain', 'qty_questionmark_path', 'qty_hashtag_path',
                                'qty_hashtag_query', 'qty_at_fragment','qty_tilde_fragment', 'qty_plus_fragment']
                df3.drop(columns = null_feat_corr, inplace=True)

                st.write("The Overview of Input Data Features to be Predicted (After Droping its' Parts and None Impportant Features): ", df3)
                st.write("Number of Rows and Columns: ", df3.shape)
            
                #making prediction
                st.subheader('Prediction')
                prediction = best_model.predict(df3)
                if prediction == True:
                    st.info("Model Can Predict")
                else:
                    #st.warning("Can Not Predict, Check the loaded Model and also DataTypes")
                    pass

                prediction_proba = best_model.predict_proba(df3)

                #if st.button('Predict'):
                st.write("Prediction(s)")
                if prediction == 1:
                    st.warning('Yes, the link is for Phishing.')
                    st.write('Prediction Accuracy: ', prediction_proba)
                    
                else:
                    st.success('No, the link is Legitimate.')
                    st.write('Prediction Probability: ', prediction_proba)  
                    
                #calculate RMS

        else:
            st.subheader("Dataset upload")
            uploaded_file = st.file_uploader("Choose a file")
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                #Get overview of data
                st.subheader("Overview of Input Data")
                st.write(data.head())
                st.markdown("<h3></h3>", unsafe_allow_html=True)
                #Preprocess inputs
                #preprocess_df = preprocess(data, "Batch")
                if st.button('Predict'):
                    #Get batch prediction
                    prediction = best_model.predict(data)
                    prediction_df = pd.DataFrame(prediction, columns=["Prediction(s)"])
                    prediction_df = prediction_df.replace({1:'Yes, the URL is for Phishing.',
                                                    0:'No, the URL is Legitimate.'})

                    st.markdown("<h3></h3>", unsafe_allow_html=True)
                    st.subheader('Prediction')
                    st.write(prediction_df)
            
    if __name__ == '__main__':
        main()