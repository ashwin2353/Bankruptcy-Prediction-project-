import numpy as np 
import streamlit as st 
import pandas as pd 
import hydralit_components as hc
import time
import datetime
from datetime import datetime
import plotly as px
import seaborn as sns 
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score , classification_report , accuracy_score , precision_score , recall_score

#reading dataset
df = pd.read_excel("bankruptcy-prevention.xlsx")
df.rename(columns={' management_risk': 'management_risk',
                   ' financial_flexibility': 'financial_flexibility' ,
                   ' credibility' : 'credibility' , ' competitiveness' :'competitiveness' ,
                   ' operating_risk' : 'operating_risk' , ' class' : 'class'}, inplace=True)
# replace "bankruptcy" with 1 and "non-bankruptcy" with 0
df['class'] = df['class'].replace({'bankruptcy': 1, 'non-bankruptcy': 0})
#defining The X and Y Values
X = df.iloc[:,0:6]
y = df.iloc[:,-1]
#Defining Train_Test_split 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 , random_state = 42)

#defining the Layout
st.set_page_config(layout='wide',initial_sidebar_state='collapsed')
st.title('Bankruptcy Prevention App')
# specify the primary menu definition
menu_data = [
    {'icon': "📁" , 'label':"Dataset"},
    {'icon': "🗠" , 'label':"Exploratory Data Analysis"},
    {'icon': "fa fa-address-card", 'label':"Input_variables"},
    {'icon': "far fa-file-word", 'label':"Model_Evaluation"},
]

#Adding Theme
blue_theme = {
    'backgroundColor': '#0072B2',
    'textColor': '#FFFFFF',
    'font': 'sans-serif'}

#defining The MenuBar
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=blue_theme,
    home_name='Home',
    login_name=None,
    hide_streamlit_markers=False, #will show the st hamburger as well as the navbar now!
    sticky_nav=True, #at the top or not
    sticky_mode='pinned', #jumpy or not-jumpy, but sticky or pinned
)

#Adding home button
if menu_id == 'Home':
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.001)
        my_bar.progress(percent_complete + 1)

    tab1, tab2 = st.tabs(["💾 About Project",'N Ashwin Siddhartha'])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
          image = Image.open(r"C:/Users/ashwi/Desktop/project/sunrise.jpg")
          st.image(image)

    with tab1:
          st.title('About Project')
          st.text('Project Is Designed by N Ashwin Siddhartha - Group-4. For details, see in About me.')
          st.header(f"BANKRUPTCY APP")
          st.markdown('Business Objective: This is a classification project, since the variable to predict is binary bankruptcy or non-bankruptcy. The goal here is to model the probability that a business goes bankrupt from different features.')
          st.text('Variables affecting bankruptcy')
          st.text('1. industrial_risk:           0 = low risk, 0.5 = medium risk, 1 = high risk.')
          st.text('2. management_risk:          0 = low risk, 0.5 = medium risk, 1 = high risk.')
          st.text('3. financial flexibility:    0 = low flexibility, 0.5 = medium flexibility, 1 = high flexibility.')
          st.text('4. credibility:              0 = low credibility, 0.5 = medium credibility, 1 = high credibility.')
          st.text('5. competitiveness:          0 = low competitiveness, 0.5 = medium competitiveness, 1 = high competitiveness.')
          st.text('6. operating_risk:           0 = low risk, 0.5 = medium risk, 1 = high risk.')
          st.text('7. class:                    bankruptcy, non-bankruptcy (target variable).')

    with col2:
             with open(r'C:/Users/ashwi/Desktop/project/group.mp4' , 'rb') as video_file:
                  video_bytes = video_file.read()
                  st.video(video_bytes)
                  st.write('Team_Members : 1.Jagadish  , 2.Ganesh , 3.Narendra , 4.ashwin , 5.Prakash , 6.Shivani')


    with tab2:
      col1,col2 = st.columns(2)
    with col1:
        st.write('Contact Information')
        st.write('ashwinsiddhartha000@gmail.com')
        st.write('Connect with me on:')
        st.write('Click On the Website: ')
        st.write('[GitHub](https://github.com/ashwin2353)')
        st.write('[LinkedIn](https://www.linkedin.com/in/ashwin-siddhartha-8078a8166/)')

#Adding the dataset
if menu_id == 'Dataset':
    st.write('Dataset Consists of 250 Rows and 7 Columns')
    # read your dataset using pandas
    df = pd.read_excel("bankruptcy-prevention.xlsx")
    # display the dataset in a table using streamlit
    options = ['Display_table' , 'Display_Columns' , 'Display_sample','Display_NullValues' , 'Display_Count' , 'Display_Shape' , 'Display_Datatypes']
    selected_option = st.selectbox('Details of the data' , options) 
    if selected_option == 'Display_table':
         st.subheader('Total Dataset')
         st.table(df)

    elif selected_option == 'Display_Columns':
         st.subheader('Columns of the dataset')
         st.write(df.columns)

    elif selected_option == 'Display_sample':
         st.subheader('Sample Selected data ')
         st.write(df.sample(10))

    elif selected_option == 'Display_NullValues':
         st.subheader('Displaying_NullValues')
         st.write(df.isnull().sum())

    elif selected_option == 'Display_Count':
         st.subheader('Counts Of Each Variable')
         st.write(df.count())

    elif selected_option == 'Display_Shape':
         st.subheader('Shape of the dataset')
         st.write(df.shape)
    
    elif selected_option == 'Display_Datatypes':
         st.subheader('Datatypes of the each variable')
         st.write(df.dtypes)


#Adding the Exploratary Data Analysis Univarite and Multivarite Analysis
if menu_id == 'Exploratory Data Analysis':
                            my_bar = st.progress(0)

                            for percent_complete in range(100):
                             time.sleep(0.001)
                            my_bar.progress(percent_complete + 1)

                            tab1, tab2 = st.tabs(["Univariate Analysis📊" , "Multivariate Analysis📈"])
                            with tab1:
                              col1, col2 = st.columns(2)
                              with col1:
                                st.header('Exploratory Data Analysis -Univarite analysis')
                                st.subheader('Univarite analysis')
                                st.write('Univarite analysis explores each variable in a dataset separately.')
                                options = ['industrial_risk', 'management_risk', 'financial_flexibility', 'credibility', 'competitiveness', 'operating_risk']

                                #Describe The dataset
                                st.write(df.describe())

                                #Value_counts
                                selected_option = st.selectbox('Select a variable to display value counts', options)
                                st.write(df[selected_option].value_counts())

                                #Histogram
                                selected_option = st.selectbox('Select an option', options)
                                fig, ax = plt.subplots()
                                fig, ax = plt.subplots(figsize=(8, 6))
                                df.hist(column=selected_option, by='class', ax=ax, stacked=True)
                                ax.set_title("Histogram for {}".format(selected_option))
                                ax.set_xlabel(selected_option)
                                ax.set_ylabel('Frequency')
                                st.pyplot(fig)

                                #Pie Charts
                                selected_option = st.selectbox('Select a variable to create a pie chart', options)
                                fig, ax = plt.subplots(figsize=(8, 8))
                                df[selected_option].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
                                ax.set_title("Pie chart for {}".format(selected_option))
                                st.pyplot(fig)

                                #Count Plots
                                selected_option = st.selectbox('Select a variable to create a count plot', options)
                                fig, ax = plt.subplots()
                                sns.countplot(data=df, x=selected_option, ax=ax)
                                ax.set_title("Count plot for {}".format(selected_option))
                                ax.set_xlabel(selected_option)
                                ax.set_ylabel('Count')
                                st.pyplot(fig)

                                # Define tab-2 for multivariate analysis
                                with tab2:
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.header("Exploratory Data Analysis - Multivariate Analysis")
                                        st.subheader('Multivariate Analysis')
                                        st.write('Multivariate analysis is based on observation and analysis of more than one statistical outcome variable at a time.')
                                    
                                    # Define options for visualizations
                                    options = ['Correlation Heatmap', 'Pairplot', 'Boxplot']
                                    selected_option = st.selectbox('Select a visualization', options)
                                    
                                    # Display Correlation Heatmap if selected
                                    if selected_option == 'Correlation Heatmap':
                                        st.subheader('Correlation between the Variables')
                                        corr_matrix = df.corr()
                                        fig, ax = plt.subplots(figsize=(10, 8))
                                        sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True, ax=ax)
                                        ax.set_title("Correlation Matrix")
                                        st.pyplot(fig)
                                    
                                    # Display Pairplot if selected
                                    elif selected_option == 'Pairplot':
                                        st.subheader('Pairplot')
                                        fig = sns.pairplot(data=df)
                                        st.pyplot(fig)

                                        # Display Boxplot if selected
                                    elif selected_option == 'Boxplot':
                                        st.subheader('Boxplot')
                                        fig, ax = plt.subplots(figsize=(10, 8))
                                        sns.boxplot(data=df, ax=ax)
                                        ax.set_title("Boxplot")
                                        st.pyplot(fig)

if menu_id == 'Input_variables' :

    st.subheader("Enter the following details to check whether the company will go bankrupt or not.")
    # Add your input fields and logic here
    industrial_risk = st.selectbox("industrial_risk", [0, 0.5, 1], index=0)
    management_risk = st.selectbox("management_risk", [0, 0.5, 1], index=0)
    financial_flexibility = st.selectbox("financial_flexibility", [0, 0.5, 1], index=0)
    credibility = st.selectbox("credibility", [0, 0.5, 1], index=0)
    competitiveness = st.selectbox("competitiveness", [0, 0.5, 1], index=0)
    operating_risk = st.selectbox("operating_risk", [0, 0.5, 1], index=0)
    
    def predict_score(industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk):
        input_data = pd.DataFrame({'Industrial_Risk': [industrial_risk],
                                   'Management_Risk': [management_risk],
                                   'Financial_Flexibility': [financial_flexibility],
                                   'Credibility': [credibility],
                                   'Competitiveness': [competitiveness],
                                   'Operating_Risk': [operating_risk]})
        return input_data
    
    # Call the predict_bankruptcy function and display the result
    def predict_bankruptcy(input_data):
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        prediction = lr.predict(input_data)
        return prediction[0]
    
    if st.button('Predict_Class'):
        input_data = predict_score(industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk)
        prediction = predict_bankruptcy(input_data)
        if prediction == 0:
            st.write('The company is not likely to go bankrupt.')
        else:
            st.write('The company is likely to go bankrupt.')
    

if menu_id == 'Model_Evaluation':
     st.header("Logistic_Regression Model")
     st.subheader("Train_Test_Split")
     st.write(X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 , random_state = 42))
     lr = LogisticRegression()
     lr.fit(X_train,y_train)
     lr_pred_train = lr.predict(X_train)
     lr_pred_test = lr.predict(X_test)
     score = lr.score(X_train, y_train)
     score = lr.score(X_test, y_test)

     st.write("LogisticRegression test Accuracy: {:.2f}%".format(score*100))
     st.write("LogisticRegression Train Accuracy: {:.2f}%".format(score*100))

     report = classification_report(y_test, lr_pred_test, output_dict=True)
     # Display precision for each class
     for label in report:
         if label.isdigit():
             st.write(f"Precision for class {label}: {report[label]['precision']:.2f}")
     
     # Display recall for each class
     for label in report:
         if label.isdigit():
             st.write(f"Recall for class {label}: {report[label]['recall']:.2f}")
     
     # Display F1-score for each class
     for label in report:
         if label.isdigit():
             st.write(f"F1-score for class {label}: {report[label]['f1-score']:.2f}")