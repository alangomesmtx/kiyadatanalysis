import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
import backend
import classification as cl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier


def Calc():
    def silhouette_method(x):
        silhouette_avg = []
        range_n_clusters = range(2, 11)
        for num_clusters in range_n_clusters:
            # initialise kmeans
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(x)
            cluster_labels = list(kmeans.labels_)
            # silhouette score
            silhouette_avg.append(silhouette_score(x, cluster_labels))
        ideal_Clusters = range_n_clusters[silhouette_avg.index(max(silhouette_avg))]
        return range_n_clusters, silhouette_avg, ideal_Clusters

    def pnpModelResults(range_n_clusters, silhouette_avg, ideal_Clusters):
        df_plot = pd.DataFrame({'Values of K': range_n_clusters, 'Silhouette Score': silhouette_avg})
        fig = px.line(df_plot, x='Values of K', y='Silhouette Score', markers=True, line_shape='linear',
                      title='Silhouette Score Plot for Different values of K')
        fig.update_traces(marker=dict(size=10))
        fig.update_xaxes(tickvals=df_plot['Values of K'], ticktext=df_plot['Values of K'])
        silhouetteScore = max(silhouette_avg)
        st.info("The ideal number of clusters is: '{}', with Silhouette Score: '{}'.".format(ideal_Clusters,
                                                                                             silhouetteScore),
                icon='✅')
        st.plotly_chart(fig)

    def plotGraph(plot_dimensions, x, selectedColumns, model):
        if plot_dimensions == '2D Plot':
            # Get all labels
            labels = model.labels_
            # Get unique labels
            unique_labels = set(labels)

            # Create a list to store scatter traces
            scatter_data = []

            # Create scatter traces for each unique label
            for label in unique_labels:
                x_label = x[labels == label, 0]
                y_label = x[labels == label, 1]
                trace = go.Scatter(
                    x=x_label,
                    y=y_label,
                    mode='markers',
                    marker=dict(color=label),
                    name=f'Cluster {label}'  # Assign a name based on the label
                )
                scatter_data.append(trace)
            trace = go.Scatter(x=model.cluster_centers_[:, 0],
                               y=model.cluster_centers_[:, 1],
                               mode='markers',
                               marker=dict(color='yellow',
                                           size=10,
                                           line=dict(color='black', width=1)
                                           ),
                               # text=["Centroid " + str(i) for i in range(len(model.cluster_centers_))],
                               # hoverinfo='text',
                               name='Centroids'
                               )
            scatter_data.append(trace)

            # Create a layout for the plot
            layout = go.Layout(
                title="Cluster Plot",
                xaxis=dict(title=selectedColumns[0]),
                yaxis=dict(title=selectedColumns[1]),
                showlegend=True
            )

            # Create a Figure object and plot the data
            fig = go.Figure(data=scatter_data, layout=layout)
            # Show the Plotly plot
            st.plotly_chart(fig)

        if plot_dimensions == '3D Plot':
            # 3d scatterplot using plotly
            Scene = dict(xaxis=dict(title=selectedColumns[0]), yaxis=dict(title=selectedColumns[1]),
                         zaxis=dict(title=selectedColumns[2]))

            # Get all labels
            labels = model.labels_
            # Get unique labels
            unique_labels = set(labels)

            # Create a list to store scatter traces
            scatter_data = []

            for label in unique_labels:
                x_label = x[labels == label, 0]
                y_label = x[labels == label, 1]
                z_label = x[labels == label, 2]
                trace = go.Scatter3d(
                    x=x_label,
                    y=y_label,
                    z=z_label,
                    mode='markers',
                    marker=dict(color=label),
                    name=f'Cluster {label}'  # Assign a name based on the label
                )
                scatter_data.append(trace)

            # trace = go.Scatter3d(x=x[:, 0], y=x[:, 1], z=x[:, 2], mode='markers',marker=dict(color = labels,
            # size= 10, line=dict(color= 'black',width = 10)))
            trace = go.Scatter3d(x=model.cluster_centers_[:, 0],
                                 y=model.cluster_centers_[:, 1],
                                 z=model.cluster_centers_[:, 2],
                                 mode='markers',
                                 marker=dict(color='yellow',
                                             size=10,
                                             line=dict(color='black', width=1)
                                             ),
                                 # text=["Centroid " + str(i) for i in range(len(model.cluster_centers_))],
                                 # hoverinfo='text',
                                 name='Centroids'
                                 )
            scatter_data.append(trace)
            layout_2 = go.Layout(margin=dict(l=0, r=0),
                                 scene=Scene,
                                 height=800,
                                 width=1000,
                                 title="Cluster Plot",
                                 showlegend=True,
                                 plot_bgcolor="red"
                                 # paper_bgcolor='darkgrey'
                                 )
            # Create a Figure object and plot the data
            fig = go.Figure(data=scatter_data, layout=layout_2)
            st.plotly_chart(fig)

    uploaded_file = st.file_uploader("Upload your file:")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        col_names = list(df.columns)

        for i in col_names:
            backend.catCheck(df, i)

        columns_to_drop = st.multiselect("Select columns to drop", df.columns)

        if columns_to_drop:
            # Remove selected columns from the DataFrame
            df = df.drop(columns=columns_to_drop)
            # Display the cleaned data
            st.write("Data after dropping selected columns:")
            st.write(df)
        else:
            # If no columns are selected, display the original data
            st.write("Dataframe without dropping columns")
            st.write(df)

        col_names = df.columns
        X_cat = backend.encoding(df)
        x = X_cat

        c1, c2 = st.columns(2)
        with c1:
            b1 = st.button("Give Silhouette Analysis", use_container_width=True)
        if b1:
            range_n_clusters, silhouette_avg, ideal_Clusters = silhouette_method(x)
            pnpModelResults(range_n_clusters, silhouette_avg, ideal_Clusters)

        with c2:
            b2 = st.button("Create Model", use_container_width=True)
            if 'b2' not in st.session_state:
                st.session_state.b2 = False

        if b2 or st.session_state.b2:
            st.session_state.b2 = True
            range_n_clusters, silhouette_avg, ideal_Clusters = silhouette_method(x)
            model = KMeans(n_clusters=ideal_Clusters, init="k-means++", max_iter=300, n_init=10, random_state=0)
            y_clusters = model.fit_predict(x)
            df_2 = df
            df_2["Cluster"] = y_clusters
            x["Cluster"] = y_clusters
            print(model.cluster_centers_)
            st.write(df_2)

            plot_dimensions = st.radio("Select a plot type:", ['2D Plot', '3D Plot'])
            not_cat_columns = []
            for i in df.columns:
                if not backend.catCheck(df, i):
                    not_cat_columns.append(i)

            if plot_dimensions == '2D Plot':
                selectedColumn1 = st.selectbox("Select column for X axis:", not_cat_columns)
                selectedColumn2 = st.selectbox("Select column for Y axis:", not_cat_columns)
                selectedColumns = [selectedColumn1, selectedColumn2]
                x_plot = df.iloc[:, [df.columns.get_loc(selectedColumn1),
                                     df.columns.get_loc(selectedColumn2)]].values
                range_n_clusters, silhouette_avg, ideal_Clusters = silhouette_method(x_plot)
            if plot_dimensions == '3D Plot':
                selectedColumn1 = st.selectbox("Select column for X axis:", not_cat_columns)
                selectedColumn2 = st.selectbox("Select column for Y axis:", not_cat_columns)
                selectedColumn3 = st.selectbox("Select column for Z axis:", not_cat_columns)
                selectedColumns = [selectedColumn1, selectedColumn2, selectedColumn3]
                x_plot = df.iloc[:, [df.columns.get_loc(selectedColumn1),
                                     df.columns.get_loc(selectedColumn2),
                                     df.columns.get_loc(selectedColumn3)]].values
                range_n_clusters, silhouette_avg, ideal_Clusters = silhouette_method(x_plot)
            y_clusters_plot = model.fit(x_plot)
            plotGraph(plot_dimensions, x_plot, selectedColumns, model)

            target_col = 'Cluster'
            isCat = backend.catCheck(df, target_col)

            plot_color_col = st.selectbox("Select your ploting column: ", col_names)
            isCat2 = backend.catCheck(df, plot_color_col)
            if isCat2:
                st.info("Feature selected is Categorical Data", icon='ℹ️')
            else:
                st.info("Feature selected is Numerical Data, Select a feature with categorical data", icon='ℹ️')

            # plotting the double graph
            st.info("Plot of '{}' against '{}' is as follows:".format(target_col, plot_color_col), icon='✅')
            fig = px.histogram(df, x=target_col, color=plot_color_col, barmode='group', height=400)
            st.plotly_chart(fig)

            X, X_train, X_test, y_train, y_test, arr = cl.splitting(df, target_col, isCat)

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
            if st.button('Create Classification Model', use_container_width=True):
                # b1 = True
                y_pred, y_pred_prob, accuracy = cl.model(clf, X_train, X_test, y_train, y_test, selected_model)
                cl.printModelResults(y_test, y_pred, y_pred_prob, isCat, df, target_col, arr, selected_model,
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
                    inputDict.update({X.columns[i]: [a]})
            if st.button('Predict Results with input values', use_container_width=True):
                # b2 = True
                cl.model(clf, X_train, X_test, y_train, y_test, selected_model)  # Necessary for model creation
                cl.calcResults(inputDict, X, selected_model, arr, df[target_col].unique(), target_col)


def app():
    st.title("Clustering")
    Calc()
