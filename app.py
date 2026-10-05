
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="Pathstone Privacy Intrusion Detection",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 Pathstone Family Office")
st.subheader("Privacy & Intrusion Detection System")

np.random.seed(42)

n = 500

data = pd.DataFrame({
    "User_ID": np.random.randint(1001, 1051, n),
    "Login_Hour": np.random.randint(8, 19, n),
    "Failed_Logins": np.random.randint(0, 3, n),
    "Files_Accessed": np.random.randint(1, 10, n),
    "Data_Transferred_MB": np.random.randint(10, 500, n),
    "Location_Change": np.random.randint(0, 2, n),
    "Sensitive_Data_Access": np.random.randint(0, 2, n)
})

data.loc[490:499, "Login_Hour"] = np.random.randint(0, 5, 10)
data.loc[490:499, "Failed_Logins"] = np.random.randint(5, 10, 10)
data.loc[490:499, "Files_Accessed"] = np.random.randint(20, 50, 10)
data.loc[490:499, "Data_Transferred_MB"] = np.random.randint(2000, 5000, 10)
data.loc[490:499, "Location_Change"] = 1
data.loc[490:499, "Sensitive_Data_Access"] = 1

features = [
    "Login_Hour",
    "Failed_Logins",
    "Files_Accessed",
    "Data_Transferred_MB",
    "Location_Change",
    "Sensitive_Data_Access"
]

model = IsolationForest(
    contamination=0.02,
    random_state=42
)

data["Prediction"] = model.fit_predict(data[features])

data["Status"] = data["Prediction"].map({
    1: "Normal",
    -1: "Intrusion"
})

privacy_risk = data[
    (data["Sensitive_Data_Access"] == 1) &
    (data["Status"] == "Intrusion")
]

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Dataset",
        "EDA",
        "Intrusion Detection",
        "Privacy Risk",
        "Final Results"
    ]
)

if page == "Dashboard":

    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", len(data))
    col2.metric("Normal Activities", len(data[data["Status"] == "Normal"]))
    col3.metric("Detected Intrusions", len(data[data["Status"] == "Intrusion"]))
    col4.metric("Privacy Risks", len(privacy_risk))

    st.divider()

    st.subheader("Case Study")

    st.write("""
    This project simulates the privacy and security characteristics
    of the Pathstone Family Office breach using synthetic data.
    """)

    st.info(
        "The dataset is synthetic and does not contain actual Pathstone customer information."
    )

elif page == "Dataset":

    st.header("📊 User Activity Dataset")

    st.dataframe(data, use_container_width=True)

elif page == "EDA":

    st.header("📈 Exploratory Data Analysis")

    st.dataframe(data.describe(), use_container_width=True)

    st.subheader("Data Transfer Analysis")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data["Data_Transferred_MB"], bins=30)
    ax.set_xlabel("Data Transferred (MB)")
    ax.set_ylabel("Number of Activities")
    ax.set_title("Data Transfer Distribution")
    st.pyplot(fig)

    st.subheader("Failed Login Analysis")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data["Failed_Logins"], bins=10)
    ax.set_xlabel("Failed Login Attempts")
    ax.set_ylabel("Number of Activities")
    ax.set_title("Failed Login Distribution")
    st.pyplot(fig)

elif page == "Intrusion Detection":

    st.header("🚨 Intrusion Detection")

    intrusions = data[data["Status"] == "Intrusion"]

    st.metric("Detected Intrusions", len(intrusions))

    st.dataframe(
        intrusions,
        use_container_width=True
    )

    counts = data["Status"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values)
    ax.set_xlabel("Activity Type")
    ax.set_ylabel("Number of Activities")
    ax.set_title("Normal vs Intrusion Activities")
    st.pyplot(fig)

elif page == "Privacy Risk":

    st.header("🔒 Privacy Risk Detection")

    st.metric(
        "Privacy Risk Records",
        len(privacy_risk)
    )

    st.dataframe(
        privacy_risk,
        use_container_width=True
    )

    st.warning(
        "These activities represent potential unauthorized access to sensitive information."
    )

elif page == "Final Results":

    st.header("📋 Final Results")

    normal = len(data[data["Status"] == "Normal"])
    intrusion = len(data[data["Status"] == "Intrusion"])
    privacy = len(privacy_risk)

    results = pd.DataFrame({
        "Category": [
            "Total Records",
            "Normal Activities",
            "Detected Intrusions",
            "Privacy Risk Records"
        ],
        "Count": [
            len(data),
            normal,
            intrusion,
            privacy
        ]
    })

    st.dataframe(results, use_container_width=True)

    st.success("""
    The system identifies abnormal user activity using Isolation Forest
    and detects potential privacy risks involving sensitive information.
    """)
    
