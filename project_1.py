# Bank Customer Churn Prediction — Streamlit App
# Converted from a Google Colab notebook to a Streamlit-compatible application.

import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Bank Customer Churn Prediction")
st.write(
    "Upload the European bank customer CSV dataset to explore the data, "
    "compare churn prediction models, and generate customer churn-risk scores."
)

st.info(
    "Upload the original `European_Bank.csv` file. The app performs cleaning, "
    "feature engineering, model training, evaluation, and risk scoring automatically."
)

uploaded_file = st.file_uploader(
    "Upload your customer churn CSV file",
    type=["csv"],
    help="Expected target column: Exited (0 = stayed, 1 = exited).",
)

if uploaded_file is None:
    st.warning("Please upload the CSV file to start the analysis.")
    st.stop()

try:
    raw_data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the uploaded CSV file: {exc}")
    st.stop()

st.subheader("1. Dataset Overview")
st.caption("Preview of the uploaded data")
st.dataframe(raw_data.head(10), use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.metric("Rows", f"{raw_data.shape[0]:,}")
with c2:
    st.metric("Columns", f"{raw_data.shape[1]:,}")

required_columns = [
    "CustomerId", "Surname", "CreditScore", "Geography", "Gender", "Age",
    "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "Exited",
]
missing_columns = [col for col in required_columns if col not in raw_data.columns]
if missing_columns:
    st.error(
        "The uploaded file is missing required column(s): "
        + ", ".join(missing_columns)
        + ". Please upload the expected European bank customer dataset."
    )
    st.stop()

data = raw_data.copy()
initial_rows = len(data)
data = data.drop_duplicates().copy()
data = data.dropna(subset=["Exited"]).copy()

try:
    data["Exited"] = pd.to_numeric(data["Exited"], errors="raise").astype(int)
except Exception:
    st.error("The `Exited` column must contain numeric values 0 and 1.")
    st.stop()

if not set(data["Exited"].unique()).issubset({0, 1}):
    st.error("The `Exited` target column must contain only 0 (stayed) and 1 (exited).")
    st.stop()

# Remove identifiers and columns not used as predictors, if present.
data = data.drop(columns=[c for c in ["Year", "CustomerId", "Surname"] if c in data.columns])

# Handle missing values before feature engineering.
numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
categorical_columns = data.select_dtypes(include=["object", "category", "string"]).columns.tolist()

for col in numeric_columns:
    if col != "Exited" and data[col].isna().any():
        data[col] = data[col].fillna(data[col].median())

for col in categorical_columns:
    if data[col].isna().any():
        modes = data[col].mode()
        if not modes.empty:
            data[col] = data[col].fillna(modes.iloc[0])

# Feature engineering, following the notebook's feature definitions.
salary_denominator = data["EstimatedSalary"].replace(0, np.nan)
data["Balance_Salary_Ratio"] = (
    data["Balance"] / salary_denominator
).replace([np.inf, -np.inf], np.nan).fillna(0)

data["Product_Density"] = data["NumOfProducts"] / (data["Tenure"] + 1)
data["Engagement_Product_Interaction"] = data["IsActiveMember"] * data["NumOfProducts"]
data["Age_Tenure_Interaction"] = data["Age"] * data["Tenure"]

# Remove rows with invalid/missing target and confirm predictors are usable.
data = data.dropna(subset=["Exited"]).copy()
X = data.drop(columns=["Exited"]).copy()
y = data["Exited"].astype(int).copy()

if y.nunique() < 2:
    st.error("The dataset must include both target classes: Exited = 0 and Exited = 1.")
    st.stop()

categorical_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
numeric_features = X.select_dtypes(include=np.number).columns.tolist()

st.write("**Target distribution**")
target_counts = y.value_counts().sort_index()
target_table = pd.DataFrame({
    "Customer Status": ["Stayed (0)", "Exited (1)"],
    "Customers": [
        int(target_counts.get(0, 0)),
        int(target_counts.get(1, 0)),
    ],
})
st.dataframe(target_table, hide_index=True, use_container_width=True)

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(data=target_table, x="Customer Status", y="Customers", ax=ax)
ax.set_title("Customer Churn Distribution")
ax.set_xlabel("Customer Status")
ax.set_ylabel("Number of Customers")
st.pyplot(fig)
plt.close(fig)

with st.expander("View data quality checks"):
    st.write(f"Duplicate rows removed: {initial_rows - len(raw_data.drop_duplicates())}")
    st.write("Missing values remaining in predictors before model preprocessing:")
    st.dataframe(data.isna().sum().rename("Missing Values").to_frame(), use_container_width=True)
    st.write("Feature-engineered data preview:")
    st.dataframe(data.head(10), use_container_width=True)

st.subheader("2. Feature Engineering")
engineered_features = [
    "Balance_Salary_Ratio",
    "Product_Density",
    "Engagement_Product_Interaction",
    "Age_Tenure_Interaction",
]
st.write("The app creates four additional features based on the original notebook.")
st.dataframe(data[engineered_features].describe().T.round(4), use_container_width=True)

corr = data[engineered_features + ["Exited"]].corr(numeric_only=True)["Exited"].drop("Exited").sort_values()
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=corr.values, y=corr.index, ax=ax)
ax.axvline(0, color="black", linewidth=1)
ax.set_title("Correlation of Engineered Features with Churn")
ax.set_xlabel("Correlation with Exited")
ax.set_ylabel("Engineered Feature")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("3. Train and Compare Machine Learning Models")
test_size = st.slider("Test dataset size (%)", min_value=10, max_value=35, value=20, step=5)
if y.value_counts().min() < 2:
    st.error("Each target class needs at least two records to create a stratified train/test split.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100, random_state=42, stratify=y
)

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler()),
])
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ],
    remainder="drop",
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    ),
}

if st.button("Run model training and evaluation", type="primary"):
    with st.spinner("Training models and calculating evaluation metrics..."):
        trained_models = {}
        predictions = {}
        probabilities = {}
        results = []

        for name, model in models.items():
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ])
            pipeline.fit(X_train, y_train)
            trained_models[name] = pipeline
            predictions[name] = pipeline.predict(X_test)
            probabilities[name] = pipeline.predict_proba(X_test)[:, 1]

            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions[name]),
                "Precision": precision_score(y_test, predictions[name], zero_division=0),
                "Recall": recall_score(y_test, predictions[name], zero_division=0),
                "F1-Score": f1_score(y_test, predictions[name], zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, probabilities[name]),
            })

        results_df = pd.DataFrame(results).sort_values(
            by="ROC-AUC", ascending=False
        ).reset_index(drop=True)
        best_model_name = results_df.iloc[0]["Model"]
        best_model = trained_models[best_model_name]

        # Save results in Streamlit session state so download/display remain available.
        st.session_state["results_df"] = results_df
        st.session_state["trained_models"] = trained_models
        st.session_state["predictions"] = predictions
        st.session_state["probabilities"] = probabilities
        st.session_state["best_model_name"] = best_model_name
        st.session_state["best_model"] = best_model
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["risk_data"] = X_test.copy()
        st.session_state["source_data"] = data.copy()
    st.success("All four models have been trained and evaluated.")

if "results_df" in st.session_state:
    results_df = st.session_state["results_df"]
    trained_models = st.session_state["trained_models"]
    predictions = st.session_state["predictions"]
    probabilities = st.session_state["probabilities"]
    best_model_name = st.session_state["best_model_name"]
    best_model = st.session_state["best_model"]
    X_test_saved = st.session_state["X_test"]
    y_test_saved = st.session_state["y_test"]

    st.markdown("### Model comparison")
    st.dataframe(
        results_df.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}",
            "ROC-AUC": "{:.4f}",
        }),
        use_container_width=True,
    )
    st.success(f"Best model by ROC-AUC: {best_model_name} (ROC-AUC = {results_df.iloc[0]['ROC-AUC']:.4f})")

    fig, ax = plt.subplots(figsize=(11, 5))
    results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].plot(
        kind="bar", ax=ax
    )
    ax.set_title("Comparison of Bank Customer Churn Prediction Models")
    ax.set_ylabel("Score")
    ax.set_xlabel("Machine Learning Model")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="lower right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Confusion matrices and classification reports")
    selected_model = st.selectbox("Choose a model to inspect", list(trained_models.keys()))
    cm = confusion_matrix(y_test_saved, predictions[selected_model], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Stayed (0)", "Exited (1)"],
        yticklabels=["Stayed (0)", "Exited (1)"],
    )
    ax.set_title(f"Confusion Matrix — {selected_model}")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    st.pyplot(fig)
    plt.close(fig)

    report = classification_report(
        y_test_saved, predictions[selected_model],
        labels=[0, 1], target_names=["Stayed", "Exited"],
        zero_division=0, output_dict=True
    )
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in trained_models:
        RocCurveDisplay.from_predictions(
            y_test_saved, probabilities[name], name=name, ax=ax
        )
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random classification")
    ax.set_title("ROC Curve Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("4. Customer Churn Risk Scoring")
    risk_data = X_test_saved.copy()
    risk_data["Actual_Exited"] = y_test_saved.to_numpy()
    risk_data["Predicted_Churn_Probability"] = best_model.predict_proba(X_test_saved)[:, 1]
    risk_data["Predicted_Exited"] = best_model.predict(X_test_saved)

    def assign_risk(probability):
        if probability < 0.30:
            return "Low Risk"
        if probability < 0.60:
            return "Medium Risk"
        return "High Risk"

    risk_data["Churn_Risk_Category"] = risk_data["Predicted_Churn_Probability"].apply(assign_risk)
    risk_data = risk_data.sort_values("Predicted_Churn_Probability", ascending=False)

    risk_summary = (
        risk_data["Churn_Risk_Category"]
        .value_counts()
        .reindex(["Low Risk", "Medium Risk", "High Risk"], fill_value=0)
        .rename_axis("Risk Category")
        .reset_index(name="Number of Customers")
    )
    risk_summary["Percentage"] = (
        risk_summary["Number of Customers"] / len(risk_data) * 100
    ).round(2)

    a, b, c = st.columns(3)
    for col, label in zip([a, b, c], ["Low Risk", "Medium Risk", "High Risk"]):
        count = int(risk_summary.loc[risk_summary["Risk Category"] == label, "Number of Customers"].iloc[0])
        with col:
            st.metric(label, f"{count:,}")

    st.write("Risk bands are illustrative thresholds based on predicted model scores, not guaranteed probabilities.")
    st.dataframe(risk_summary, hide_index=True, use_container_width=True)
    st.dataframe(risk_data.head(20), use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=risk_summary, x="Risk Category", y="Number of Customers", ax=ax)
    ax.set_title("Customer Distribution by Predicted Churn Risk")
    ax.set_xlabel("Risk Category")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("5. Download Results")
    csv_outputs = {
        "European_Bank_Feature_Engineered.csv": st.session_state["source_data"],
        "Chapter_7_Model_Comparison.csv": results_df,
        "Chapter_7_Customer_Churn_Risk_Scores.csv": risk_data,
        "Chapter_7_Risk_Category_Summary.csv": risk_summary,
    }
    for filename, frame in csv_outputs.items():
        st.download_button(
            label=f"Download {filename}",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
            key=f"download_{filename}",
        )
else:
    st.caption("After uploading the CSV, click **Run model training and evaluation** to display the model metrics and risk scores.")
