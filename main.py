import streamlit as st
import pandas as pd
import pickle

st.write("""
# Hello!
""")

st.sidebar.header('User Input')
st.sidebar.subheader('Please enter your data:')
def get_input():
    # widget
    GPAX = st.sidebar.slider('GPAX', min_value=0.00, max_value=4.00)
    Type = st.sidebar.selectbox('Entry Type Name:', ['CHIANG RAI DEVELOPMENT SCHOLARSHIP','DIRECT ADMISSION','DIRECT ADMISSION BY SCHOOL','DIRECT ADMISSION UNDER CONDITION GPAX 2.00 FIRST SEMESTER','DISABLE STUDENT','EP-MEP PROGRAM','GOOD BEHAVE STUDENTS','INTERNATIONAL SCHOOL','QUOTA 17 NORTHERN PROVINCES','QUOTA BY COMMUNITY HOSPITAL','QUOTA BY SCHOOL','QUOTA FOR SOUTHERN BORDER','RE-ID FIRST SEMESTER GPAX 2.00','SPECIAL FOR GOOD STUDENT','SPECIAL TALENT'])
    # Type = st.sidebar.selectbox('Entry Type Name:', ['QUOTA 17 NORTHERN PROVINCES','DIRECT ADMISSION BY SCHOOL','SPECIAL FOR GOOD STUDENT','QUOTA BY SCHOOL','SPECIAL TALENT'])

    st.sidebar.subheader('Expectation for studying in MFU:')
    Q1 = st.sidebar.checkbox('beautiful scenary and atmosphere')
    Q2 = st.sidebar.checkbox('quality of life')
    Q3 = st.sidebar.checkbox('campus and facilities')
    Q4 = st.sidebar.checkbox('modern and ready-to-use learning support and facilities')
    Q5 = st.sidebar.checkbox('sources of student scholarship')
    Q6 = st.sidebar.checkbox('demand by workforce market')

    st.sidebar.subheader('Factor to apply for MFU:')
    Q23 = st.sidebar.checkbox('easy/convenient transportation')
    Q24 = st.sidebar.checkbox('suitable cost')
    Q25 = st.sidebar.checkbox('graduates with higher language/academic competency than other universities')
    Q26 = st.sidebar.checkbox('learning in English')
    Q27 = st.sidebar.checkbox('quality/reputation of university')
    Q28 = st.sidebar.checkbox('excellence in learning support and facilities')
    Q29 = st.sidebar.checkbox('provision of preferred major')
    Q30 = st.sidebar.checkbox('environment and setting motivate learning')
    Q31 = st.sidebar.checkbox('experienced and high-quality instructors')
    Q32 = st.sidebar.checkbox('suggestion by school teacher/friend/relative')
    Q33 = st.sidebar.checkbox('suggestion by family')

    st.sidebar.subheader('If your application fails, will you try again?')
    Q34 = st.sidebar.checkbox('try the same major')
    Q35 = st.sidebar.checkbox('try a different major')
    Q36 = st.sidebar.checkbox('will not try again')
    


    # change value
    if Q1 == 1 : Q1 = 1
    else: Q1 = 0

    if Q2 == 1 : Q2 = 1
    else: Q2 = 0

    if Q3 == 1 : Q3 = 1
    else: Q3 = 0

    if Q4 == 1 : Q4 = 1
    else: Q4 = 0

    if Q5 == 1 : Q5 = 1
    else: Q5 = 0

    if Q6 == 1 : Q6 = 1
    else: Q6 = 0

    if Q23 == 1 : Q23 = 1
    else: Q23 = 0

    if Q24 == 1 : Q24 = 1
    else: Q24 = 0

    if Q25 == 1 : Q25 = 1
    else: Q25 = 0

    if Q26 == 1 : Q26 = 1
    else: Q26 = 0

    if Q27 == 1 : Q27 = 1
    else: Q27 = 0

    if Q28 == 1 : Q28 = 1
    else: Q28 = 0

    if Q29 == 1 : Q29 = 1
    else: Q29 = 0

    if Q30 == 1 : Q30 = 1
    else: Q30 = 0

    if Q31 == 1 : Q31 = 1
    else: Q31 = 0

    if Q32 == 1 : Q32 = 1
    else: Q32 = 0

    if Q33 == 1 : Q33 = 1
    else: Q33 = 0

    if Q34 == 1 : Q34 = 1
    else: Q34 = 0

    if Q35 == 1 : Q35 = 1
    else: Q35 = 0

    if Q36 == 1 : Q36 = 1
    else: Q36 = 0

    

    # dictionary
    data = {'GPAX': GPAX,
            'EntryTypeName': Type,
            'Q1': Q1,
            'Q2': Q2,
            'Q3': Q3,
            'Q4': Q4,
            'Q5': Q5,
            'Q6': Q6,
            'Q23': Q23,
            'Q24': Q24,
            'Q25': Q25,
            'Q26': Q26,
            'Q27': Q27,
            'Q28': Q28,
            'Q29': Q29,
            'Q30': Q30,
            'Q31': Q31,
            'Q32': Q32,
            'Q33': Q33,
            'Q34': Q34,
            'Q35': Q35,
            'Q36': Q36}
    
# create data frame
    data_df = pd.DataFrame(data, index=[0])
    return data_df

df = get_input()

st.write(df)


data_sample = pd.read_csv('test3.csv')
df = pd.concat([df, data_sample],axis=0)
# st.write(df)

cat_data = pd.get_dummies(df[['EntryTypeName']])

# st.write(cat_data)

#Combine all transformed features together
X_new = pd.concat([cat_data, df], axis=1)
X_new = X_new[:1] # Select only the first row (the user input data)
#Drop un-used feature
X_new = X_new.drop(columns=['EntryTypeName'])
# st.write(X_new)


# -- Reads the saved normalization model
# load_nor = pickle.load(open('normalization.pkl', 'rb'))
#Apply the normalization model to new data
# X_new = load_nor.transform(X_new)
st.write(X_new)

# -- Reads the saved classification model
load_knn = pickle.load(open('best_knn.pkl', 'rb'))
# Apply model for prediction
prediction = load_knn.predict(X_new)
st.write(prediction)