import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from scipy import stats
import pickle
import category_encoders as ce
import numpy as np
import math
# algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
# model evaluation
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
# hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
# model calibration
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import backend


def splitting(dataFrame, target, isCategorical):
    X = dataFrame.drop(target, axis=1)
    arr = []
    for i in range(len(dataFrame[target].unique())):
        arr.append(i)
    y = dataFrame[target].replace(dataFrame[target].unique(), arr)

    X_cat = backend.encoding(X)

    X_train, X_test, y_train, y_test = train_test_split(X_cat, y, test_size=0.2, random_state=48)

    return X, X_train, X_test, y_train, y_test, arr


def save_mlm(ml_encoder, model_name):
    output1 = open(r'pickleFiles\tempMLMPickleFile_{}.pkl'
                   .format(model_name), 'wb')
    pickle.dump(ml_encoder, output1)
    output1.close()


def model(clf, X_train, X_test, y_train, y_test, selected_model):
    clf.fit(X_train, y_train)
    save_mlm(clf, selected_model)
    y_pred_proba = clf.predict_proba(np.array(X_test))[:, 1]
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, y_pred_proba, accuracy


def printModelResults(y_test, y_pred, y_pred_proba, isCat, df, target_col, arr, selected_model, accuracy):
    # creating a translated y_test

    if isCat:
        y_test_translated = y_test.replace(arr, df[target_col].unique())
    else:
        y_test_translated = y_test

    st.info("The Accuracy of the {} generated is {}%".format(selected_model, (accuracy * 100)), icon='✅')

    # confusion matrix
    con_mtx = confusion_matrix(y_test, y_pred)
    val = np.mat(con_mtx)
    classnames = list(set(y_test_translated))
    df_cm = pd.DataFrame(val, index=classnames, columns=classnames)
    st.info("The confusion matrix is as follows: ", icon='✅')

    # plot confusion matrix
    fig1 = px.imshow(df_cm, labels=dict(x='Predicted Label', y='True Label', color='Event Counts'),
                     title='Churn {} Results'.format(selected_model), text_auto=True, aspect='auto',
                     color_continuous_scale='cividis')
    fig1.update()
    st.plotly_chart(fig1)


def load_mlm(model_name):
    pklFile = open(r'pickleFiles\tempMLMPickleFile_{}.pkl'
                   .format(model_name), 'rb')
    retrieved_cat_model = pickle.load(pklFile)
    pklFile.close()
    return retrieved_cat_model


def calcResults(inputDict, X, selected_model, arr, target_unique, target_col):
    inputData = pd.DataFrame(inputDict)
    df1_category = X.select_dtypes(exclude=['int64', 'float64'])
    df2_category = inputData.select_dtypes(exclude=['int64', 'float64'])
    df3_category = X.select_dtypes(include=['int64', 'float64'])
    df4_category = inputData.select_dtypes(include=['int64', 'float64'])
    df_category = pd.concat([df1_category, df2_category], ignore_index=True)
    df_numeric = pd.concat([df3_category, df4_category], ignore_index=True)
    retrieved_encoder = backend.load_encoder()
    encoded_input = retrieved_encoder.fit_transform(df_category)
    final_input = pd.concat([encoded_input, df_numeric], axis=1)
    final_input_arr = final_input.iloc[-1].to_numpy()
    final_input_arr_reshaped = final_input_arr.reshape((1, -1))

    retrieved_cat_model = load_mlm(selected_model)
    cat_pred = retrieved_cat_model.predict(final_input_arr_reshaped)
    cat_pred_df = pd.DataFrame(cat_pred, index=[0], columns=['Drug'])
    cat_pred_translated = cat_pred_df.replace(arr, target_unique)
    b = cat_pred_translated['Drug'][0]
    st.info("The predicted {} for the input value is '{}'".format(target_col, b), icon="✅")


def Calc():
    # df = pd.read_csv("drug200.csv")
    uploadedFile = st.file_uploader("Choose a file: ")
    if uploadedFile is not None:
        df = pd.read_csv(uploadedFile)
        st.info("The dataframe is as follows:", icon='ℹ')
        st.write(df)
        col_name = df.columns

        target_col = st.selectbox("Select your target column: ", col_name)
        isCat = backend.catCheck(df, target_col)
        if isCat:
            st.info("Feature selected is Categorical Data", icon='ℹ')
        else:
            st.info("Feature selected is Numerical Data, Select a feature with categorical data", icon='ℹ')

        plot_color_col = st.selectbox("Select your ploting column: ", col_name)
        isCat2 = backend.catCheck(df, plot_color_col)
        if isCat2:
            st.info("Feature selected is Categorical Data", icon='ℹ')
        else:
            st.info("Feature selected is Numerical Data, Select a feature with categorical data", icon='ℹ')

        # plotting the double graph
        # if st.button("Plot graph", use_container_width=True):
        st.info("Plot of '{}' against '{}' is as follows:".format(target_col, plot_color_col), icon='✅')
        fig = px.histogram(df, x=target_col, color=plot_color_col, barmode='group', height=400)
        st.plotly_chart(fig)

        if isCat:
            X, X_train, X_test, y_train, y_test, arr = splitting(df, target_col, isCat)

            classificationModels = ['Logistic Regression Model', 'Random Forest Model', 'XGBoost Model',
                                    'CatBoost Model']
            selected_model = st.radio("Select the type of model", classificationModels)
            if selected_model == 'Logistic Regression Model':
                clf = LogisticRegression(max_iter=2500, random_state=48)
            if selected_model == 'Random Forest Model':
                clf = RandomForestClassifier(random_state=48)
            if selected_model == 'XGBoost Model':
                clf = xgb.XGBClassifier(random_state=48)
            if selected_model == 'CatBoost Model':
                clf = CatBoostClassifier(random_state=48, verbose=False)

            b1 = b2 = False
            if st.button('Create Model', use_container_width=True):
                # b1 = True
                y_pred, y_pred_prob, accuracy = model(clf, X_train, X_test, y_train, y_test, selected_model)
                printModelResults(y_test, y_pred, y_pred_prob, isCat, df, target_col, arr, selected_model,
                                  accuracy)

            st.write("\n\nEnter your input values: ")
            inputDict = {}
            for i in range(len(X.columns)):
                if backend.catCheck(X, X.columns[i]):
                    a = st.selectbox("Select value for {}".format(X.columns[i]), X[X.columns[i]].unique())
                    inputDict.update({X.columns[i]: [a]})
                else:
                    a = st.number_input("Enter value for {}. (Value of input file ranges from {} to {})".
                                        format(X.columns[i],
                                               X[X.columns[i]].min(),
                                               X[X.columns[i]].max()))
                    # a = 17
                    inputDict.update({X.columns[i]: [a]})
            if st.button('Predict Results with input values', use_container_width=True):
                # b2 = True
                model(clf, X_train, X_test, y_train, y_test, selected_model)  # Necessary for model creation
                calcResults(inputDict, X, selected_model, arr, df[target_col].unique(), target_col)


def app():
    st.title("Classification")
    Calc()
