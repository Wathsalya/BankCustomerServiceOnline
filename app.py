import streamlit as st
import pandas as pd 
#from matplotlib import pyplot as plt
#from plotly import graph_objs as go
#from sklearn.linear_model import LinearRegression
import numpy as np



#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#from sklearn.metrics import mean_squared_error, r2_score
import base64
#from io import BytesIO

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis  # LDA,QDA
#from sklearn.naive_bayes import GaussianNB  # Naive Bayes
#from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



#st.set_page_config(page_title='The Machine Learning App-3181/3239',
                 #  layout='wide')

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

st.image("data//P2.png", width=500)
st.image("data//building.jpg", width=1020)


df = pd.read_csv("data//train.csv")
df3 = pd.read_csv("data//Book3.csv")
def build_model(df, df2):
    df['LoanAmount_log'] = np.log(df['LoanAmount'])
    # df['LoanAmount_log'].hist(bins=20)
    # plt.show()
    # plt.hist(df['ApplicantIncome'])
    df.isnull().sum()
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())
    df.LoanAmount_log = df.LoanAmount_log.fillna(df.LoanAmount_log.mean())
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['TotalIncome_log'] = np.log(df['TotalIncome'])

    X = df.iloc[:, np.r_[1:5, 9:11, 13:15]].values  # Using all column except for the last column as X
    Y = df.iloc[:, 12].values  # Selecting the last column as Y

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100 - split_size) / 100)

    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    from sklearn.preprocessing import LabelEncoder  # here categorical variale also become numeric

    labelEncoder_X = LabelEncoder()
    for i in range(0, 5):
        X_train[:, i] = labelEncoder_X.fit_transform(X_train[:, i])
    X_train[:, 7] = labelEncoder_X.fit_transform(X_train[:, 7])

    labelEncoder_Y = LabelEncoder()
    Y_train = labelEncoder_Y.fit_transform(Y_train)

    st.markdown('**1.4. new details**:')
    st.write('X_TRAIN')
    st.info(X_train)
    st.write('Y_TRAIN')
    st.info(Y_train)

    for i in range(0, 5):
        X_test[:, i] = labelEncoder_X.fit_transform(X_test[:, i])
    X_test[:, 7] = labelEncoder_X.fit_transform(X_test[:, 7])

    Y_test = labelEncoder_Y.fit_transform(Y_test)
    st.write('X_TEST')
    st.info(X_test)
    st.write('Y_TEST')
    st.info(Y_test)

    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.fit_transform(X_test)

    from sklearn.tree import DecisionTreeClassifier
    DTClassifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    DTClassifier.fit(X_train, Y_train)  # giving train set to the classifier

    y_pred = DTClassifier.predict(X_test)  # predict the y from x test by classifier
    st.write('Y_PRED')
    st.info(y_pred)  # predict y tet from x test

    from sklearn import metrics
    st.write('the acuracy of decision tree is')
    st.info(metrics.accuracy_score(y_pred, Y_test))

    from sklearn.naive_bayes import GaussianNB
    NBClassifier = GaussianNB()
    NBClassifier.fit(X_train, Y_train)
    y_pred2 = NBClassifier.predict(X_test)
    st.write('naive_bayes y_pred')
    st.info(y_pred2)
    st.write('the acuracy of naive_ayes  is')
    st.info(metrics.accuracy_score(y_pred2, Y_test))

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, Y_train)
    y_pred3 = lda.predict(X_test)
    st.write('the acuracy of LDA  is')
    st.info(metrics.accuracy_score(y_pred3, Y_test))

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, Y_train)
    y_pred4 = qda.predict(X_test)
    st.write('the acuracy of QDA  is')
    st.info(metrics.accuracy_score(y_pred4, Y_test))

    # lets begin the predit the model

    df2['Gender'].fillna(df2['Gender'].mode()[0], inplace=True)
    df2['Dependents'].fillna(df2['Dependents'].mode()[0], inplace=True)
    df2['Self_Employed'].fillna(df2['Self_Employed'].mode()[0], inplace=True)

    df2['Loan_Amount_Term'].fillna(df2['Loan_Amount_Term'].mode()[0], inplace=True)
    df2['Credit_History'].fillna(df2['Credit_History'].mode()[0], inplace=True)
    # print(df2.isnull().sum())

    df2.boxplot(column="LoanAmount")
    # plt.show()
    df2.boxplot(column="ApplicantIncome")
    # plt.show()

    df2.LoanAmount = df2.LoanAmount.fillna(df2.LoanAmount.mean())
    df2['LoanAmount_log'] = np.log(df2['LoanAmount'])
    # print(df2.isnull().sum())

    df2['TotalIncome'] = df2['ApplicantIncome'] + df2['CoapplicantIncome']
    df2['TotalIncome_log'] = np.log(df2['TotalIncome'])
    # print(df2.head())

    test = df2.iloc[:, np.r_[1:5, 9:11, 13:15]].values
    for i in range(0, 5):
        test[:, i] = labelEncoder_X.fit_transform(test[:, i])
    test[:, 7] = labelEncoder_X.fit_transform(test[:, 7])

    st.write('test details99999')
    st.info(test)

    test = ss.fit_transform(test)
    pred = NBClassifier.predict(test)
    p = df2['Loan_ID']
    st.write('PREDICTION OF TEST CSV')
    st.info(pred)
    ID = []  # initialize list Dc = []# initialize list
    STATUS = []  # initialize list Dc = []# initialize list

    for i in p:
        ID.append(i)
    for n in pred:
        STATUS.append(n)

    st.info(type(ID))
    st.info(type(STATUS))
    d = {'LOAN ID': ID, 'LOAN STATUS': STATUS}
    data = pd.DataFrame(d, columns=['LOAN ID', 'LOAN STATUS'])
    data['LOAN STATUS'].replace((1, 0), ('Yes', 'No'), inplace=True)
    st.write(data)
    data2 = data.loc[(data['LOAN STATUS'] == 'No')]
    data1 = data.loc[(data['LOAN STATUS'] == 'Yes')]

    st.write(data1)
    st.write(data2)
    csv = data1.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    st.markdown(href, unsafe_allow_html=True)
def build_model2(df3,df4):
    # print(type(df3))
    # print(df3)

    # Print summary statistics
    description = df3.describe()
    print(description)

    st.info("\n")

    # st.info DataFrame information
    st.info(df3.isnull().sum())

    st.info("\n")
    df3['Gender'].fillna(df3['Gender'].mode()[0], inplace=True)
    df3.Age = df3.Age.fillna(df3.Age.mean())
    df3['Married'].fillna(df3['Married'].mode()[0], inplace=True)
    df3['BankCustomer'].fillna(df3['BankCustomer'].mode()[0], inplace=True)
    df3['EducationLevel'].fillna(df3['EducationLevel'].mode()[0], inplace=True)
    df3['Ethnicity'].fillna(df3['Ethnicity'].mode()[0], inplace=True)
    df3['ZipCode'].fillna(df3['ZipCode'].mode()[0], inplace=True)

    st.info(df3.isnull().sum())

    # Import LabelEncoder
    from sklearn.preprocessing import LabelEncoder

    # Instantiate LabelEncoder
    le = LabelEncoder()

    # Iterate over all the values of each column and extract their dtypes
    for col in df3:
        # Compare if the dtype is object
        if df3[col].dtype == 'object':
            # Use LabelEncoder to do the numeric transformation
            df3[col] = le.fit_transform(df3[col])
    st.info(df3)

    # convert to categorical data to dummy data
    # credit_var = pd.get_dummies(df3[col],
    # columns=["Married", "EducationLevel", "Citizen", "DriversLicense", "Ethnicity"])
    # As we can see all features are of numeric type now

    x = df3.iloc[:, np.r_[0:15]].values
    y = df3.iloc[:, 16].values

    st.info([x])
    st.info([y])

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=123)

    # Instantiate MinMaxScaler and use it to rescale X_train and X_test
    from sklearn.preprocessing import MinMaxScaler

    scale = MinMaxScaler(feature_range=(0, 1))
    rescaledX_train = scale.fit_transform(X_train)
    rescaledX_test = scale.fit_transform(X_test)

    # Import LogisticRegression
    from sklearn.linear_model import LogisticRegression

    # Instantiate a LogisticRegression classifier with default parameter values
    logreg = LogisticRegression()

    # Fit logreg to the train set
    logreg.fit(rescaledX_train, Y_train)

    # Import confusion_matrix
    from sklearn.metrics import confusion_matrix

    # Use logreg to predict instances from the test set and store it
    y_pred = logreg.predict(rescaledX_test)

    # Get the accuracy score of logreg model and st.info it
    st.info("Accuracy of logistic regression classifier: ")
    st.info(logreg.score(rescaledX_test, Y_test))

    # st.info the confusion matrix of the logreg model
    st.info(confusion_matrix(Y_test, y_pred))

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(rescaledX_train, Y_train)
    y_pred2 = rf.predict(rescaledX_test)
    st.info(rf.score(rescaledX_test, Y_test))
    confusionmatrix = (confusion_matrix(Y_test, y_pred2))
    st.info(confusionmatrix)

    df4['Gender'].fillna(df4['Gender'].mode()[0], inplace=True)
    df4.Age = df4.Age.fillna(df4.Age.mean())
    df4['Married'].fillna(df4['Married'].mode()[0], inplace=True)
    df4['BankCustomer'].fillna(df4['BankCustomer'].mode()[0], inplace=True)
    df4['EducationLevel'].fillna(df4['EducationLevel'].mode()[0], inplace=True)
    df4['Ethnicity'].fillna(df4['Ethnicity'].mode()[0], inplace=True)
    df4['ZipCode'].fillna(df4['ZipCode'].mode()[0], inplace=True)

    for col in df4:
        # Compare if the dtype is object
        if df4[col].dtype == 'object':
            # Use LabelEncoder to do the numeric transformation
            df4[col] = le.fit_transform(df4[col])

    credit_var3 = pd.get_dummies(df4[col],
                                 columns=["Married", "EducationLevel", "Citizen", "DriversLicense", "Ethnicity"])
    xtest = df4.iloc[:, np.r_[0:15]].values
    xtest1 = scale.fit_transform(xtest)
    yprednew = logreg.predict(xtest1)

    st.info(yprednew)


st.sidebar.header('1. Upload your loan CSV data')
uploaded_file2 = st.sidebar.file_uploader("Upload Test CSV file", type=["csv"])
split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
st.sidebar.header('2. Upload your credit card CSV data')
uploaded_file3 = st.sidebar.file_uploader("Upload Test credit CSV file", type=["csv"])

nav = st.sidebar.radio("Navigation",["HOME","lOAN APPROVEMENT","CREDIT CARD ISSUENCE"])
if nav == "HOME":
    st.header("Welcome")
   # if st.checkbox("Show Table"):
     #   st.table(df2)
if nav == "lOAN APPROVEMENT":
    if uploaded_file2 is not None:
        df2 = pd.read_csv(uploaded_file2)
        build_model(df,df2)
if nav == "CREDIT CARD ISSUENCE":
    if uploaded_file3 is not None:
        df4 = pd.read_csv(uploaded_file3)
        build_model2(df3, df4)




