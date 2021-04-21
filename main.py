import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()

@st.cache
def get_data(filename):
    HR_DATA = pd.read_csv(filename)
    return HR_DATA

with header:
    st.title('Content-based Recommender System')
    st.text('In this project I build a content-based recommender system using HR_DATA.csv')

with dataset:
    st.header('HR_DATA.csv dataset')
    st.text('I got this dataset from Kaggle: https://www.kaggle.com/davidepolizzi/hr-data-set-based-on-human-resources-data-set')

    HR_DATA = get_data('data/HR_DATA.csv')
    st.write(HR_DATA.head())

    st.subheader('Position distribution on the HR DATA dataset')
    state_disribution = pd.DataFrame(HR_DATA['Position'].value_counts())
    st.bar_chart(state_disribution)

    st.subheader('State distribution on the HR DATA dataset')
    state_disribution = pd.DataFrame(HR_DATA['State'].value_counts())
    st.bar_chart(state_disribution)

    st.subheader('Department distribution on the HR DATA dataset')
    state_disribution = pd.DataFrame(HR_DATA['Department'].value_counts())
    st.bar_chart(state_disribution)

    st.subheader('MaritalDesc distribution on the HR DATA dataset')
    state_disribution = pd.DataFrame(HR_DATA['MaritalDesc'].value_counts())
    st.bar_chart(state_disribution)

with features:
    st.header('Features')

    st.markdown('* **Input:** I created this feature in order combine all the relevant parameters required for input')

with model_training:
    st.header('Training')
    st.text('Here you can get recommendation for any individual')

    sel_col, disp_col = st.beta_columns(2)

    input_name = sel_col.selectbox('Select any name from dataset', (HR_DATA['Employee_Name']))
    df = get_data('data/HR_DATA.csv')
    sel_col.subheader('Name Details')
    sel_col.write(df.loc[df['Employee_Name'] == input_name])

    df = get_data('data/HR_DATA.csv')
    df = df[df['Position'].notna()]
    df = df[df['Employee_Name'].notna()]
    df = df[df['State'].notna()]
    df = df[df['Department'].notna()]
    df = df[df['MaritalDesc'].notna()]
    df['Position'] = df['Position'].astype(str)
    df['Employee_Name'] = df['Employee_Name'].astype(str)
    df['State'] = df['State'].astype(str)
    df['Department'] = df['Department'].astype(str)
    df['MaritalDesc'] = df['MaritalDesc'].astype(str)
    df['Input'] = df['Position'].map(str) + ' ' + df['State'].map(str) + ' ' + df['Department'].map(str) + ' ' + df['MaritalDesc'].map(str)
    metadata = df.copy()
    tfidf = TfidfVectorizer(stop_words='english')
    metadata['Input'] = metadata['Input'].fillna('')
    tfidf_matrix = tfidf.fit_transform(metadata['Input'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(metadata.index, index=metadata['Employee_Name']).drop_duplicates()
    def get_recommendations(Employee_Name, cosine_sim=cosine_sim):
        idx = indices[Employee_Name]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        employee_indices = [i[0] for i in sim_scores]
        return metadata['Employee_Name'].iloc[employee_indices]

    recommender = get_recommendations(input_name)

    disp_col.subheader('Recommendations')
    disp_col.write(recommender)

    disp_col.subheader('Recommendations Details')
    for i in recommender:
        disp_col.write(df.loc[df['Employee_Name'] == i])

    


