import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import base64
import seaborn as sns
import pickle

#Title
st.title("Marketing Campaigns Analysis")


# Create a sidebar page dropdown
page = st.sidebar.radio("Choose your page", ["Home","Data Upload", "Visualizations", "Predicting Response"])
if page == "Home":
    st.image('https://www.aub.edu.lb/articles/PublishingImages/Nov-21/OSB_Eduniversal_league_ranking_story-thumb.jpg')
    st.subheader("Streamlit Final Project")
    st.subheader("Jana El Oud")
if page == "Data Upload":
    # Display details of page 1
    st.write("This application conducts Marketing Campaigns Analysis for a retailer!")
    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
         bytes_data = pd.read_csv(uploaded_file)
         st.write("filename:", uploaded_file.name)
         st.write(bytes_data)
    with st.expander("Data Explanation!"):
     st.write("""
         The dataframe above shows the demographics and purachasing behavior of customers of a retailer.
         The retail store sells 6 product categories: Wine, Fruit, Sweet, Meat, Fish, and Gold.
         The retailer sells through 3 channels: Store, Catalog, and Website .
         AcceptedCmp1, AcceptedCmp2, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5: 1 if the customer responded to each of the campaigns, 0 otherwise.
         Response: This is a derived characteristic that is calculated by subtracting the trip date by the booking date.
     """)
elif page == "Visualizations":
    # Display details of page 2
    st.subheader("Exploratory Analysis")
    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        bytes_data = pd.read_csv(uploaded_file)

    #Histograms showing number of purchases by channel
    st.subheader("Number of Purchases by Channel")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    dataframe1=pd.DataFrame(bytes_data[:2241],columns=["NumWebPurchases","NumCatalogPurchases","NumStorePurchases"])
    dataframe1.hist()
    st.pyplot()
    st.write("It is evident that the store is the most popular channel!")
    #Customer Demographics barchart
    st.subheader("Number of Purchases by Channel")
    fig = px.bar(bytes_data, x="Marital_Status", y="NumDealsPurchases", color="Education", barmode="group")
    st.plotly_chart(fig)
    #Scatterplot
    st.subheader('Scatterplot analysis')
    selected_x_var = st.selectbox('What do you want the x variable to be?', ['ID','Year_Birth', 'Income','Kidhome',	'Teenhome'	,'Dt_Customer',	'Recency',	'MntWines','MntFruits',	'MntMeatProducts'	,'MntFishProducts'	,'MntSweetProducts',	'MntGoldProds',	'NumDealsPurchases'	,'NumWebPurchases'	,'NumCatalogPurchases',	'NumStorePurchases',	'NumWebVisitsMonth'])
    selected_y_var = st.selectbox('What about the y?', ['ID','Year_Birth', 'Income','Kidhome',	'Teenhome'	,'Dt_Customer',	'Recency',	'MntWines','MntFruits',	'MntMeatProducts'	,'MntFishProducts'	,'MntSweetProducts',	'MntGoldProds',	'NumDealsPurchases'	,'NumWebPurchases'	,'NumCatalogPurchases',	'NumStorePurchases',	'NumWebVisitsMonth'])
    fig = px.scatter(bytes_data, x = bytes_data[selected_x_var], y = bytes_data[selected_y_var])
    st.plotly_chart(fig)
    st.write("Refer to the above scatter plot for different visualizations of customer demographics and purchasing behavior!")

#Third Page: Predicting Customer Response
elif page == "Predicting Response":

#file upload or user inouts
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:

        input_df = pd.read_csv(uploaded_file)

    else:

        def user_input_features():


            Education = st.selectbox('Education',('Graduation', 'PhD','Master', 'Basic','2n Cycle'))

            Marital_Status = st.selectbox('Marital Status',("Married","Together","Single","Divorced","Widow","Absurd","YOLO","Alone"))

            Income = st.slider('Income', 0,666666)

            Kidhome = st.slider('Number of Kids', 0,2,1)

            Teenhome = st.slider('Number of Teens', 0,2,1)

            Recency = st.slider('Recency', 0,99)

            data = {'Education':[Education],

                    'Marital Status':[Marital_Status],

                    'Income':[Income],

                    'Kidhome':[Kidhome],

                    'Teenhome':[Teenhome],

                    'Recency':[Recency],}
            features = pd.DataFrame(data)
            return features

    input_df = user_input_features()


#data import, cleaning, and encoding
    response_raw = pd.read_csv('https://raw.githubusercontent.com/JanaEO/Streamlit2/main/Marketing%20Campaign%20Data1.csv')

    response_raw.fillna(0, inplace=True)

    response = response_raw.drop(columns=['Response'])
    df = pd.concat([input_df,response],axis=0)
    #encoding categorical variables
    encode = ['Education','Marital_Status']
    for col in encode:

        dummy = pd.get_dummies(df[col], prefix=col)

        df = pd.concat([df, dummy], axis=1)

        del df[col]

    df = df[:1]
    df.fillna(0, inplace=True)

    features =['Income','Kidhome','Teenhome','Recency','Education_2n Cycle','Education_Basic','Education_Graduation','Education_Master','Education_PhD', 'Marital_Status_Absurd','Marital_Status_Alone','Marital_Status_Divorced','Marital_Status_Married','Marital_Status_Single','Marital_Status_Together','Marital_Status_Widow','Marital_Status_YOLO']
    df = df[features]

    st.subheader('User Input features')

    print(df.columns)

    #file upload or manual fill
    if uploaded_file is not None:

        st.write(df)

    else:

        st.write(' Currently using example parameters shown below.')

        st.write(df)


    #load pickle
    load_clf = pickle.load(open('https://raw.githubusercontent.com/JanaEO/Streamlit2/main/response_clf.pkl', 'rb'))

    #Model prediction
    prediction = load_clf.predict(df)

    prediction_proba = load_clf.predict_proba(df)

    st.write("Will the customer respond to the campaigns?")
    response_labels = np.array(['No','Yes'])

    st.write(response_labels[prediction])

    st.subheader('Prediction Probability')

    st.write(prediction_proba)
