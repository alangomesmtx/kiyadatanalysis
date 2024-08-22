import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy import stats


def Calc():
    def printResults(correlation, rmse, col2, y_pred_output, fig, results, graph):
        if results:
            st.info("The correlation is = {} ".format(round(correlation, 3)), icon='✅')
            st.info("The root mean squared error is = {} ".format(round(rmse, 3)), icon='✅')
            st.info("Predicted {} is = {}".format(col2, round(y_pred_output[0], 4)), icon='✅')
        if graph:
            st.pyplot(fig)

    def splitting(df, col1, col2):
        X = df[col1]
        y = df[col2]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train = X_train.to_numpy().reshape((-1, 1))
        X_test = X_test.to_numpy().reshape((-1, 1))
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        return X_train, X_test, y_train, y_test

    def lrm(col1, col2, X_train, y_train, X_test, y_test, inputForPred):

        def linReg(X_train, y_train, X_test, y_test, inputForPred):
            model_linReg = linear_model.LinearRegression()
            model_linReg = model_linReg.fit(X_train, y_train)
            y_pred_test = model_linReg.predict(X_test)
            y_pred_output = model_linReg.predict([[inputForPred]])

            correlation = stats.pearsonr(y_test, y_pred_test)[0]
            rmse = math.sqrt(mean_squared_error(y_test, y_pred_test))

            return correlation, rmse, y_pred_output, y_pred_test

        def plotGraphLinReg(col1, col2, X_train, y_train, X_test, y_pred_test, inputForPred, y_pred_output):
            fig = plt.figure(figsize=(10, 6))
            plt.scatter(X_train, y_train, color='b')
            plt.plot(X_test, y_pred_test, 'r')
            plt.plot(inputForPred, y_pred_output, marker='o', markersize=10, markeredgecolor="green",
                     markerfacecolor="yellow")
            plt.title("Linear Regression Chart")
            plt.xlabel(col1)
            plt.ylabel(col2)

            # fig = px.scatter(pd.concat([X_train, y_train]))
            return fig

        correlation, rmse, y_pred_output, y_pred_test = linReg(X_train, y_train, X_test, y_test, inputForPred)
        fig = plotGraphLinReg(col1, col2, X_train, y_train, X_test, y_pred_test, inputForPred, y_pred_output)
        return correlation, rmse, y_pred_output, fig

    def svrm(X_train, y_train, X_test, y_test, inputForPred, a, b, hypetune, button):

        def svr(X_train, y_train, X_test, y_test, inputForPred, a, b, hyptune):
            if hyptune:
                C_arr = [0.1, 1, 10, 100, 1000]
                eps_arr = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

                hyper_arr = []
                hyper_cols = ['c', 'epsilon', 'correlation']

                for C in C_arr:
                    for epsilon in eps_arr:
                        model_svr = SVR(kernel='rbf', C=C, epsilon=epsilon)
                        model_svr.fit(X_train, y_train)
                        y_pred_svr = model_svr.predict(X_test)
                        corrSVR = stats.pearsonr(y_test, y_pred_svr)[0]
                        hyper_arr.append([C, epsilon, corrSVR])

                hyper_df = pd.DataFrame(hyper_arr, columns=hyper_cols)
                hyper_df = hyper_df.sort_values(by=['correlation'], ascending=False)

                a = hyper_df['c'].iloc[0]
                b = hyper_df['epsilon'].iloc[0]

            model_svr = SVR(kernel='rbf', C=a, epsilon=b)
            model_svr = model_svr.fit(X_train, y_train)

            y_pred_test = model_svr.predict(X_test)
            y_pred_output = model_svr.predict([[inputForPred]])

            correlation = stats.pearsonr(y_test, y_pred_test)[0]
            mse_svr = mean_squared_error(y_test, y_pred_test)
            rmse_svr = math.sqrt(mse_svr)

            return y_test, y_pred_test, correlation, rmse_svr, y_pred_output, a, b

        def plotGraphSVR1(col1, col2, X_train, y_train, X_test, y_test, y_pred_test, inputForPred, y_pred_output, a, b):
            regressor = SVR(kernel='rbf', C=a, epsilon=b)
            regressor.fit(X_train, y_train)

            X_grid = np.arange(min(X_test), max(X_test), 0.01)  # this step required because data is feature scaled.
            X_grid = X_grid.reshape((len(X_grid), 1))

            fig = plt.figure(figsize=(12, 8))

            plt.scatter(X_train, y_train, color='blue')
            plt.plot(X_grid, regressor.predict(X_grid), color='red')
            plt.plot(inputForPred, y_pred_output, marker='o', markersize=10, markeredgecolor="green",
                     markerfacecolor="yellow")
            plt.title('SVR')
            plt.xlabel(col1)
            plt.ylabel(col2)

            return fig

        def plotGraphSVR2(X_train, y_train, inputForPred, y_pred_output_1,
                          y_pred_output_2, a, b):
            fig = plt.figure(figsize=(30, 15))
            fig, axs = plt.subplots(2, sharex=True, sharey=True)

            regressor1 = SVR(kernel='rbf')
            regressor1.fit(X_train, y_train)

            axs[0].scatter(X_train, y_train, color='blue')

            X_grid = np.arange(min(X_train), max(X_train), 0.01)  # this step required because data is feature scaled.
            X_grid = X_grid.reshape((len(X_grid), 1))

            axs[0].plot(X_grid, regressor1.predict(X_grid), color='red')
            axs[0].plot(inputForPred, y_pred_output_1, marker='o', markersize=10, markeredgecolor="green",
                        markerfacecolor="yellow")

            axs[0].set_title("Without Hyper Tuning")

            regressor2 = SVR(kernel='rbf', C=a, epsilon=b)
            regressor2.fit(X_train, y_train)

            axs[1].scatter(X_train, y_train, color='blue')

            axs[1].plot(X_grid, regressor2.predict(X_grid), color='red')
            axs[1].plot(inputForPred, y_pred_output_2, marker='o', markersize=10, markeredgecolor="green",
                        markerfacecolor="yellow")

            axs[1].set_title("With Hyper Tuning")

            axs[1].scatter(X_train, y_train, color="blue")

            return fig

        if button == 1:
            y_test, y_pred_test, correlation, rmse_svr, y_pred_output, a, b = svr(X_train, y_train, X_test, y_test,
                                                                                  inputForPred, a, b, hypetune)
            fig = plotGraphSVR1(col1, col2, X_train, y_train, X_test, y_test, y_pred_test, inputForPred, y_pred_output,
                                a, b)
            return correlation, rmse_svr, y_pred_output, fig
        if button == 2:
            y_test_1, y_pred_test_1, correlation_1, rmse_svr_1, y_pred_output_1, a_1, b_1 = svr(X_train, y_train,
                                                                                                X_test, y_test,
                                                                                                inputForPred, a, b,
                                                                                                False)
            y_test_2, y_pred_test_2, correlation_2, rmse_svr_2, y_pred_output_2, a_2, b_2 = svr(X_train, y_train,
                                                                                                X_test, y_test,
                                                                                                inputForPred, a, b,
                                                                                                True)
            fig = plotGraphSVR2(X_train, y_train, inputForPred, y_pred_output_1,
                                y_pred_output_2, a_2, b_2)
            return correlation_1, rmse_svr_1, y_pred_output_1, correlation_2, rmse_svr_2, y_pred_output_2, fig

    # df = pd.read_csv('df_normalized.csv')
    uploadedFile = st.file_uploader("Choose a file: ")
    if uploadedFile is not None:
        df = pd.read_csv(uploadedFile)
        st.info("The dataframe is as follows:", icon='ℹ️')
        st.write(df)
        col_name = df.columns

        modelTypes = ["Linear Regression Model", "Support Vector Regression Model"]
        selectedModel = st.radio("Select a Regression Model:", modelTypes)

        col1 = st.selectbox("Select column/feature for X axis:", col_name)
        col2 = st.selectbox("Select column/feature for y axis:", col_name)

        X_train, X_test, y_train, y_test = splitting(df, col1, col2)
        inputForPred = st.number_input("Enter input value for '{}'. (Value ranges from '{}' to '{}'.)"
                                       .format(col1, df[col1].min(), df[col1].max()))

        if selectedModel == "Linear Regression Model":
            if st.button('Give Results'):
                correlation, rmse, y_pred_output, fig = lrm(col1, col2, X_train, y_train, X_test, y_test, inputForPred)
                printResults(correlation, rmse, col2, y_pred_output, fig, True, True)
        elif selectedModel == "Support Vector Regression Model":
            c1, c2, c3 = st.columns(3)
            b1 = b2 = b3 = False
            with c1:
                if st.button('Give Results without Hyper Tuning', use_container_width=True):
                    b1 = True

            with c2:
                if st.button('Give Results with Hyper Tuning', use_container_width=True):
                    b2 = True
            with c3:
                if st.button('See both results with Comparative Chart', use_container_width=True):
                    b3 = True
            if b1:
                correlation, rmse_svr, y_pred_output, fig = svrm(X_train, y_train, X_test, y_test,
                                                                 inputForPred, 1.0, 0.1, False, 1)
                printResults(correlation, rmse_svr, col2, y_pred_output, fig, True, True)
            if b2:
                correlation, rmse_svr, y_pred_output, fig = svrm(X_train, y_train, X_test, y_test,
                                                                 inputForPred, 1.0, 0.1, True, 1)
                printResults(correlation, rmse_svr, col2, y_pred_output, fig, True, True)
            if b3:
                correlation_1, rmse_svr_1, y_pred_output_1, correlation_2, rmse_svr_2, y_pred_output_2, fig = svrm(X_train,
                                                                                                                   y_train,
                                                                                                                   X_test,
                                                                                                                   y_test,
                                                                                                                   inputForPred,
                                                                                                                   1.0, 0.1,
                                                                                                                   True, 2)
                c11, c12 = st.columns(2)
                with c11:
                    printResults(correlation_1, rmse_svr_1, col2, y_pred_output_1, 0, True, False)
                with c12:
                    printResults(correlation_2, rmse_svr_2, col2, y_pred_output_2, 0, True, False)
                printResults(correlation_2, rmse_svr_2, col2, y_pred_output_2, fig, False, True)


def app():
    st.title("Regression Model")
    Calc()
