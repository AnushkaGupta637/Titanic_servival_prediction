import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

st.title("🚢 Titanic Model Performance Comparison")

uploaded_file = st.file_uploader("Upload Titanic Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.write(df.head())

    st.subheader("Dataset Info")
    st.write(df.shape)
    st.write(df.info())
    st.write(df.isnull().sum())

    # Feature selection & preprocessing
    x = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
    y = df["Survived"]
    x["Age"] = x["Age"].fillna(x["Age"].mean())

    encoder = LabelEncoder()
    x["Sex"] = encoder.fit_transform(x["Sex"])

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # Model dictionary
    models = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": SVC(),
        "Naive Bayes": GaussianNB(),
        "K-Nearest Neighbour": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    results = []

    for name, model in models.items():
        st.subheader(f"🔎 {name}")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        st.write(pd.DataFrame(report).transpose())

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="magma", ax=ax)
        ax.set_title(f"{name} - Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

    # Results Summary
    results_df = pd.DataFrame(results)
    st.subheader("📊 Model Performance Summary")
    st.write(results_df)

    st.subheader("🔬 Performance Comparison")
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(kind="bar", cmap="viridis", ax=ax)
    plt.title("Model Performance Comparison")
    st.pyplot(fig)
else:
    st.info("👆 Please upload the Titanic dataset CSV file to begin.")
