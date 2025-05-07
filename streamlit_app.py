import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, accuracy_score, r2_score, classification_report, confusion_matrix
from sklearn import tree
import random
from shapash import SmartExplainer
import graphviz
from sklearn.tree import export_graphviz
import dagshub
import mlflow
import mlflow.sklearn
import os
import shap
from urllib.parse import urlparse
from streamlit_shap import st_shap
from dagshub import init as dagshub_init
from sklearn.model_selection import RandomizedSearchCV

# Initialize DagsHub and MLflow
dagshub_init(
    repo_owner="deema-hazim",
    repo_name="insurance-final-project",
    mlflow=True,
)

st.set_page_config(page_title="", layout="wide")

st.markdown(
    """
<style>
    .main {
        background-color: #f7f9fc;
    }

    .stTabs [data-baseweb="tab-list"] button {
        background-color: #d1e7ff;
        color: black;
        border-radius: 8px;
        margin: 5px;
    }

    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #0d6efd;
        color: white;
    }

    .stMarkdown h3 {
        color: #0d6efd;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Navigation menu
selected = option_menu(
    menu_title=None,
    options=[
        "📖 Business Case Presentation",
        "📊 Data Vizualization",
        "🤖 Prediction Models",
        "⚙️ Hyperparameter Tuning",
    ],
    default_index=0,
    orientation="horizontal",
)

#Load the dataset
df = pd.read_csv("insurance.csv")


if selected == "📖 Business Case Presentation":
    st.title("🏥 Insurance Charges Analysis and Prediction")

    st.markdown(
        """
    ### 🔍 **Introduction**
    The insurance market is significantly influenced by various demographic and lifestyle factors. This dashboard explores how elements like **age**, **BMI**, **smoking status**, and **region** affect insurance charges, offering valuable insights for insurance companies to optimize pricing strategies and for individuals to understand potential premiums.
    """
    )

    st.markdown(
        """
    ### 📂 **Dataset Overview**
    - **Source:** Sample insurance dataset with demographic and health-related attributes.
    - **Size:** 1,338 records
    - **Features:** Age, Sex, BMI, Number of Children, Smoking Status, Region, and Insurance Charges.
    """
    )

    st.markdown(
        """
    ### 🎯 **Project Goals**
    1. **Exploratory Data Analysis (EDA):** Understand how different factors influence insurance charges.
    2. **Data Visualization:** Provide intuitive visual insights.
    3. **Predictive Modeling:** Estimate insurance charges based on user inputs.
    """
    )

    st.success(
        "This project leverages regression modeling and interactive dashboards to uncover insurance charge trends and deliver on-the-fly charge estimations."
    )

    st.header("📊 Data Preview")
    st.dataframe(df.head())

    # Visualizing average charges by smoker status
    st.subheader("💸 Average Charges by Smoker Status")
    avg_charges_smoker = df.groupby("smoker")["charges"].mean().reset_index()
    fig1, ax1 = plt.subplots()
    sns.barplot(data=avg_charges_smoker, x="smoker", y="charges", palette="viridis", ax=ax1)
    ax1.set_title("Average Insurance Charges by Smoker Status")
    st.pyplot(fig1)

    # Scatterplot of BMI vs Charges colored by smoking status
    st.subheader("📈 BMI vs. Insurance Charges by Smoking Status")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker", palette="Set1", ax=ax3)
    ax3.set_title("BMI vs. Charges")
    st.pyplot(fig3)

elif selected == "📊 Data Vizualization":
    # Visualize the relationship between different features
    st.subheader("Distribution of Age")
    fig_age, ax_age = plt.subplots()  # Create a figure and an axes.
    sns.histplot(df['age'], kde=True, ax=ax_age) # Pass the axes to the plot
    st.pyplot(fig_age)  # Pass the figure to st.pyplot()

    st.subheader("Insurance Charges by Age and Sex")
    fig_age_sex, ax_age_sex = plt.subplots()
    sns.boxplot(x='sex', y='charges', data=df, ax=ax_age_sex)
    st.pyplot(fig_age_sex)

    st.subheader("Insurance Charges by Smoking Status")
    fig_smoker, ax_smoker = plt.subplots()
    sns.boxplot(x='smoker', y='charges', data=df, ax=ax_smoker)
    st.pyplot(fig_smoker)

    st.subheader("Insurance Charges by Region")
    fig_region, ax_region = plt.subplots()
    sns.boxplot(x='region', y='charges', data=df, ax=ax_region)
    st.pyplot(fig_region)

    st.subheader("Correlation Heatmap")
    # Select only numeric columns for correlation calculation
    numeric_df = df.select_dtypes(include=np.number)  
    correlation_matrix = numeric_df.corr()
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax_corr)
    st.pyplot(fig_corr)

elif selected == "🤖 Prediction Models":
    # Subpage selection for models
    model_selection = st.radio(
        "Select Model", ["Decision Tree", "Linear Regression", "Logistic Regression"]
    )

    # Common data preparation
    model_df = df.copy()
    model_df["smoker"] = model_df["smoker"].map({"yes": 1, "no": 0})
    model_df["sex"] = model_df["sex"].map({"male": 0, "female": 1})  # Numerical encoding for gender
    X = model_df[["age", "bmi", "children", "smoker", "sex"]]  # Include 'sex' in features
    y = model_df["charges"]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_selection == "Decision Tree":
        st.subheader("🚬 Predict Smoking Status with Decision Tree")
        X_smoke = model_df[['age', 'bmi', 'children', 'charges']]
        y_smoke = model_df['smoker']

        X_train_smoke, X_test_smoke, y_train_smoke, y_test_smoke = train_test_split(X_smoke, y_smoke, test_size=0.2, random_state=42)

        max_depth_smoke = st.slider("Select Max Depth for Smoking Prediction Tree", 1, 20, 2)
        model_smoke = DecisionTreeClassifier(max_depth=max_depth_smoke, random_state=1)
        model_smoke.fit(X_train_smoke, y_train_smoke)


        y_pred_smoke = model_smoke.predict(X_test_smoke)

        accuracy_smoke = accuracy_score(y_test_smoke, y_pred_smoke)
        precision_smoke = precision_score(y_test_smoke, y_pred_smoke)

        st.success(f"Accuracy: {accuracy_smoke:.4f}")
        st.success(f"Precision: {precision_smoke:.4f}")

        st.subheader("Feature Importance")
        if hasattr(model_smoke, 'feature_importances_'):
            importances = model_smoke.feature_importances_
            feature_df = pd.DataFrame({
                "Feature": X_smoke.columns,
                "Importance": importances
            }).sort_values("Importance", ascending=False)

            fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
            sns.barplot(x="Importance", y="Feature", data=feature_df, color="#82cafc", ax=ax_importance)
            ax_importance.set_title("Feature Importance", fontsize=14, fontweight='bold')
            ax_importance.set_xlabel("Mean absolute Contribution", fontsize=12)
            ax_importance.set_ylabel("Feature", fontsize=12)
            ax_importance.tick_params(axis='x', labelsize=10)
            ax_importance.tick_params(axis='y', labelsize=10)
            ax_importance.grid(axis='x', linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig_importance)
            plt.clf()

        st.subheader("🌳 Decision Tree Visualization")
        tree_smoke = export_graphviz(
            model_smoke,
            out_file=None,
            feature_names=X_smoke.columns,
            class_names=['No', 'Yes'],
            filled=True,
            rounded=True,
            special_characters=True
        )
        graph_smoke = graphviz.Source(tree_smoke)
        st.graphviz_chart(graph_smoke)

        

    elif model_selection == "Linear Regression":
        st.subheader("🧮 Estimate Insurance Charges with Linear Regression")

        # Prepare data including gender
        model_df = df.copy()
        model_df["smoker"] = model_df["smoker"].map({"yes": 1, "no": 0})
        model_df["sex"] = model_df["sex"].map(
            {"male": 0, "female": 1}
        )  # Numerical encoding for gender
        X = model_df[
            ["age", "bmi", "children", "smoker", "sex"]
        ]  # Include 'sex' in features
        y = model_df["charges"]

        # Split the data into training and test sets (including 'sex')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Linear Regression Model
        model = LinearRegression()
        model.fit(X_train, y_train)  # Train on training data

        st.subheader("⚙️ Input Features for Prediction")
        age = st.number_input("Age:", min_value=18, max_value=100, value=30)
        bmi = st.number_input("BMI:", min_value=10.0, max_value=60.0, value=25.0)
        children = st.number_input("Number of Children:", min_value=0, max_value=5, value=0)
        smoker_input = st.selectbox("Smoker:", ["No", "Yes"])
        smoker = 1 if smoker_input == "Yes" else 0
        sex_input = st.selectbox("Gender:", ["Male", "Female"])
        sex = 0 if sex_input == "Male" else 1  # Encode user input for gender

        # Create a DataFrame from the user inputs (including 'sex')
        user_data = pd.DataFrame(
            {
                "age": [age],
                "bmi": [bmi],
                "children": [children],
                "smoker": [smoker],
                "sex": [sex],
            }
        )

        if st.button("Estimate Charge"):
            # Make prediction using the user inputs
            predicted_charge = model.predict(user_data)[0]
            st.success(f"**Estimated Insurance Charge: ${predicted_charge:,.2f}**")

        # Predictions on the test set for evaluation metrics
        y_pred = model.predict(X_test)

        # Mean Absolute Error
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("📊 Model Evaluation Metrics (on Test Data)")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**R-squared (R²):** {r2:.2f}")

        st.subheader("Predicted vs. Actual Insurance Charges")
        fig_reg, ax_reg = plt.subplots()
        ax_reg.scatter(y_test, y_pred, alpha=0.7)  # Scatter plot of actual vs. predicted
        ax_reg.plot(y_test, y_test, color='red')  # Add a diagonal line for perfect predictions
        ax_reg.set_xlabel("Actual Charges")
        ax_reg.set_ylabel("Predicted Charges")
        ax_reg.set_title("Linear Regression Performance")
        st.pyplot(fig_reg)

    elif model_selection == "Logistic Regression": # New Logistic Regression Tab
        st.subheader("🎯 Predict Smoking Status with Logistic Regression")

        # Prepare data for Logistic Regression (predicting smoker or not)
        X_logistic = model_df[['age', 'bmi', 'children', 'charges', 'sex']]  # Features
        y_logistic = model_df['smoker']  # Target: 1 for smoker, 0 for non-smoker

        # Split data into training and testing sets
        X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(
            X_logistic, y_logistic, test_size=0.2, random_state=42
        )

        # Create and train the Logistic Regression model
        model_logistic = LogisticRegression(random_state=42)  # Initialize the model
        model_logistic.fit(X_train_logistic, y_train_logistic)  # Train the model

        # Make predictions on the test set
        y_pred_logistic = model_logistic.predict(X_test_logistic)

        # Evaluate the model
        st.subheader("Model Evaluation")
        report = classification_report(y_test_logistic, y_pred_logistic, output_dict=True) #get the classification report
        df_report = pd.DataFrame(report).transpose() #make it to dataframe
        st.dataframe(df_report) #show dataframe  # Display a classification report

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test_logistic, y_pred_logistic)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        st.pyplot(plt.gcf())

        # Get user input for prediction
        st.subheader("Make a Prediction")
        age_logistic = st.number_input("Age:", min_value=18, max_value=100, value=30, key="age_logistic")
        bmi_logistic = st.number_input("BMI:", min_value=10.0, max_value=60.0, value=25.0, key="bmi_logistic")
        children_logistic = st.number_input("Number of Children:", min_value=0, max_value=5, value=0, key="children_logistic")
        charges_logistic = st.number_input("Insurance Charges:", min_value=0.0, max_value=100000.0, value=10000.0, key="charges_logistic")
        sex_logistic_input = st.selectbox("Gender:", ["Male", "Female"], key="sex_logistic")
        sex_logistic = 0 if sex_logistic_input == "Male" else 1

        # Create a DataFrame from user inputs
        user_data_logistic = pd.DataFrame({
            'age': [age_logistic],
            'bmi': [bmi_logistic],
            'children': [children_logistic],
            'charges': [charges_logistic],
            'sex': [sex_logistic],
        })

        # Make a prediction
        if st.button("Predict Smoking Status", key="predict_logistic"):
            prediction_logistic = model_logistic.predict(user_data_logistic)[0]
            if prediction_logistic == 1:
                st.error("The model predicts this person is a smoker.")
            else:
                st.success("The model predicts this person is not a smoker.")


# — Page 5: Hyperparameter Tuning —
elif selected == "⚙️ Hyperparameter Tuning":

    # 1) Data prep (you'll have loaded df_insurance elsewhere and cached it)
    @st.cache_data
    def load_data():
        df = pd.read_csv("insurance.csv")
        df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
        X = df.drop(["sex", "region", "charges"], axis=1)
        y = df["charges"]
        return train_test_split(X, y, random_state=42)

    X_train, X_test, y_train, y_test = load_data()

    # 2) Helper for metrics
    def eval_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, r2

    st.title("🔧 Hyperparameter Tuning & MLflow Dashboard")

    tab1, tab2, tab3 = st.tabs(
        [
            "1️⃣ Manual Grid Search",
            "2️⃣ Browse MLflow Runs",
            "3️⃣ Randomized Search",  # Changed tab name
        ]
    )

    # -----------------------------------------------------------------------------
    with tab1:
        st.header("Manual Grid Search of Decision Tree")
        max_depth = st.selectbox("Choose max_depth", [3, 5, 10, None])
        if st.button("▶️ Train & Log to MLflow"):
            with mlflow.start_run(run_name=f"DT-max_depth-{max_depth}"):
                model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                rmse, mae, r2 = eval_metrics(y_test, preds)
                st.write(f"*RMSE:* {rmse:.1f},  *MAE:* {mae:.1f},  *R²:* {r2:.3f}")

                # log into MLflow
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                # register the model
                tracking_url_store = mlflow.tracking.get_tracking_uri()
                is_file_store = urlparse(tracking_url_store).scheme == "file"
                if is_file_store:
                    mlflow.sklearn.log_model(model, "model")
                else:
                    mlflow.sklearn.log_model(
                        model,
                        "model",
                        registered_model_name="DecisionTreeRegressorModel",
                    )
                st.success("✅ Logged to MLflow")

    # -----------------------------------------------------------------------------
    with tab2:
        st.header("Browse Logged Experiments")
        client = mlflow.tracking.MlflowClient()
        #  get the experiment,  errors if it does not exist.
        experiment = client.get_experiment_by_name("Default")
        if experiment is None:
            st.warning(
                "No experiments found. Please run some models in the 'Manual Grid Search' tab."
            )
        else:
            runs = client.search_runs(experiment_ids=[experiment.experiment_id])
            if not runs:
                st.info("No runs found for this experiment. Please train a model.")
            else:
                runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                display_df = runs_df[
                    [
                        "run_id",
                        "params.max_depth",
                        "metrics.rmse",
                        "metrics.mae",
                        "metrics.r2",
                    ]
                ].rename(
                    columns={
                        "params.max_depth": "max_depth",
                        "metrics.rmse": "rmse",
                        "metrics.mae": "mae",
                        "metrics.r2": "r2",
                    }
                )
                st.dataframe(display_df, use_container_width=True)

    # ---------------------------------------------------------------------------------
    with tab3:
        st.header("Randomized Search for Hyperparameter Tuning")
        st.write(
            "Perform Randomized Search to find the best Decision Tree Regressor model."
        )

        # Define the parameter grid for RandomizedSearchCV
        param_grid = {
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["auto", "sqrt", "log2", None],  # Corrected
        }

        # Number of random samples to try
        n_iter = 10  # You can adjust this

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(
            DecisionTreeRegressor(random_state=42),
            param_grid,
            n_iter=n_iter,
            cv=5,  # 5-fold cross-validation
            random_state=42,
            n_jobs=-1,  # Use all available cores
            scoring="neg_mean_squared_error",  # Use a scoring metric
        )

        if st.button("▶️ Run Randomized Search"):
            with mlflow.start_run(run_name="RandomizedSearchCV_DT"):
                random_search.fit(X_train, y_train)  # Fit to the training data

                # Log parameters from the best model
                best_params = random_search.best_params_
                st.write("Best Hyperparameters:", best_params)
                for param, value in best_params.items():
                    mlflow.log_param(param, value)

                # Get the best model
                best_model = random_search.best_estimator_

                # Make predictions
                y_pred = best_model.predict(X_test)
                rmse, mae, r2 = eval_metrics(y_test, y_pred)
                st.write(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

                # Log metrics
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                # Log the best model
                tracking_url_store = mlflow.tracking.get_tracking_uri()
                is_file_store = urlparse(tracking_url_store).scheme == "file"
                if is_file_store:
                    mlflow.sklearn.log_model(best_model, "model")
                else:
                    mlflow.sklearn.log_model(
                        best_model,
                        "model",
                        registered_model_name="DecisionTreeRegressorModel_RS",
                    )
                st.success("✅ Randomized Search Completed and logged to MLflow")
