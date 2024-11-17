import streamlit as st
import sqlite3
import hashlib
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlite3 import Error
import pyttsx3
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential, load_model
from sklearn.svm import SVC
import plotly.express as px
import tensorflow as tf
from gtts import gTTS
import io
import pickle
from PIL import Image
from keras.models import load_model
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import tempfile
import base64
import speech_recognition as sr
from io import BytesIO
from datetime import datetime, timedelta
import calendar
import plotly.express as px
import plotly.graph_objects as go
import os

# --------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------
@st.cache_data
def load_data():
    encodings = ['utf-8', 'latin-1', 'iso-8859-1']
    for encoding in encodings:
        try:
            data = pd.read_csv('survey_lung_cancer.csv', encoding=encoding)
            return data
        except UnicodeDecodeError:
            continue
    st.error("Could not decode file. Please check the file encoding.")
    return None

data = load_data()

# If data loaded successfully, preprocess and split data
if data is not None:
    # Initialize Encoder and Scaler
    label_encoder = LabelEncoder()
    scaler = StandardScaler()

    # Label Encoding for categorical columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])

    # Fill missing values
    data.fillna(method='ffill', inplace=True)

    # Split data into features and target variable
    X = data.drop('LUNG_CANCER', axis=1)
    y = data['LUNG_CANCER']

    # Scale the features
    X_scaled = scaler.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Function to preprocess the data
def preprocess_data(df):
    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Convert categorical columns to numeric
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        df[column] = LabelEncoder().fit_transform(df[column])

    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric)
    return df


# ---------------------------------------------------------------------------------------------------------------

# CSS Styling for the login page to match the uploaded design
def load_css():
    st.markdown(
        """
        <style>
        /* Background styling */
        body {
            background-image: url('lux111.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        /* Centered and styled login box */
        .login-box {
            background-color: rgba(0, 0, 0, 0.8);
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);
            width: 300px;
            margin: auto;
            text-align: center;
        }

        /* Input fields styling */
        .stTextInput input {
            background-color: #1e2b48;
            color: white;
            border: none;
            padding: 0.8rem;
            border-radius: 8px;
            font-size: 1rem;
        }

        /* Login button styling */
        .stButton button {
            background-color: green;
            color: white;
            border: none;
            padding: 0.8rem;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            transition: background 0.3s;
        }

        .stButton button:hover {
            background-color: #284b63;
        }

        /* Link styling */
        a {
            color: #4ea5d9;
            text-decoration: none;
        }

        a:hover {
            color: #7ac7ff;
        }

        /* Title styling */
        h1 {
            font-size: 2.5rem;
            color: #fff;
            font-weight: 600;
        }

        /* Subtitle text for registration */
        .register-text {
            font-size: 1rem;
            color: #a3b8cf;
            margin-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Call CSS loader
load_css()


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def set_background(image_path):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/avif;base64,{get_base64_image(image_path)}");
            background-size: 75% 100%;         /* Ensures the background covers the entire screen */
            background-position: right;    /* Centers the background image */
            background-repeat: no-repeat;   /* Prevents the background from repeating */
            background-attachment: fixed;   /* Keeps the image fixed during scrolling */
            filter: brightness(90%);        /* Adjusts brightness for better text readability */
            width: 100vw;
            height: 100vh;

        }}

        </style>
        """,
        unsafe_allow_html=True
    )

def get_base64_path(path_image):
    """Encodes the image at path_image as base64."""
    with open(path_image, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def set_page(path_image):
    """Sets a full-page background image with proper scaling and centering."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/avif;base64,{get_base64_path(path_image)}");
            background-size: cover;         /* Ensures the background covers the entire screen */
            background-position: center center;    /* Centers the background image */
            background-repeat: no-repeat;   /* Prevents the background from repeating */
            background-attachment: fixed;   /* Keeps the image fixed during scrolling */
            filter: brightness(90%);        /* Adjusts brightness for better text readability */
            width: 100vw;
            height: 100vh;
        }}
        .right-image {{
            background-image: url("data:image/avif;base64,{get_base64_path(path_image)}");
            position: absolute;
            top: 50%;
            left: 10%; /* Adjusts the position from the right side */
            transform: translateY(-50%);
            width: 300px; /* Set the width of the image */
            height: 500px; /* Maintains the aspect ratio */
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2); /* Adds a shadow for better visibility */
            border-radius: 8px; /* Optional: Rounds the corners */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------------------------------------------------------

# Hash password function for security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        st.error(f"Error connecting to database: {e}")
    return conn

def initialize_database():
    conn = sqlite3.connect("admin.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Admin (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

initialize_database()

def add_admin(username, password):
    conn = create_connection("admin.db")
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO Admin (username, password) VALUES (?, ?)", (username, hash_password(password)))
            conn.commit()
        except sqlite3.IntegrityError:
            st.error("Admin username already exists.")
        finally:
            conn.close()

def authenticate_admin(username, password):
    conn = create_connection("admin.db")
    admin = None
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM Admin WHERE username = ? AND password = ?", (username, hash_password(password)))
            admin = cursor.fetchone()
        finally:
            conn.close()
    return admin

def registration_page():
    st.title("Admin Registration")
    username = st.text_input("Admin Username")
    password = st.text_input("Password", type="password")

    if st.button("Register Admin"):
        add_admin(username, password)
        st.success("Admin registered successfully!")

def login_page():
    st.title("Admin Login")
    username = st.text_input("Admin Username")
    password = st.text_input("Password", type="password")

    set_background("lux111.jpg")
    if st.button("Login"):
        admin = authenticate_admin(username, password)
        if admin:
            st.session_state["username"] = username
            st.session_state["is_admin"] = True
            st.success("Admin login successful!")
           # st.experimental_rerun()
        else:
            st.error("Invalid admin username or password.")

# Function to handle logout
def logout():
    st.session_state["username"] = None
    st.session_state["is_admin"] = False
    st.session_state.page = "Dashboard"
    st.success("Successfully logged out!")
        #st.experimental_rerun()


# -------------------------------------------------------------------------------------------------------------------

# Function to show the user dashboard (Home, About, Prevention)

def show_user_dashboard():
    # Initialize the page state in session_state if not present
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    st.markdown(
        """
        <div style="text-align:center;">
            <h1 style='color: #ffffff;'>Welcome to User Dashboard</h1>
        </div>
        """, unsafe_allow_html=True
    )

    # Add CSS for styling
    st.markdown("""
        <style>
            .button-container button {
                background-color: #007bff; /* Blue color */
                color: white;
                padding: 8px 16px;
                font-size: 16px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            .button-container button:hover {
                background-color: #0056b3; /* Darker blue on hover */
            }
            .button-container {
                display: flex;
                justify-content: space-around;
            }
        </style>
    """, unsafe_allow_html=True)

    # Container to hold buttons horizontally
    st.markdown('<div class="button-container">', unsafe_allow_html=True)

    # Buttons to navigate between pages
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Home"):
            st.session_state.page = "Home"
    with col2:
        if st.button("About"):
            st.session_state.page = "About"
    with col3:
        if st.button("Chatbot"):
            st.session_state.page = "Chatbot"
    with col4:
        if st.button("Prevention"):
            st.session_state.page = "Prevention"
    st.markdown('</div>', unsafe_allow_html=True)

    # Display content based on the selected page in session_state
    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "About":
        about_page()
    elif st.session_state.page == "Chatbot":
        chatbot_page()
    elif st.session_state.page == "Prevention":
        tips_page()


# Function to show the admin dashboard (Dataset, Prediction, Image Prediction, Logout)
def show_admin_dashboard():
    st.markdown(
        """
        <div style=text-align: center;">
            <h1 style='color: #ffffff;'>Welcome to Admin Dashboard </h1>
            
        </div>
        """, unsafe_allow_html=True
    )
    # st.title("Welcome to the Admin Dashboard")


    # Add CSS for styling
    st.markdown("""
        <style>
            .title {
                font-size: 36px;
                color: #2c3e50;
                font-weight: bold;
                text-align: center;
            }
            .card {
                background-color: #fff;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                margin: 20px 0;
            }
            .card-header {
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 15px;
            }
            .card-body {
                font-size: 16px;
                line-height: 1.6;
            }
        </style>
    """, unsafe_allow_html=True)



    # CSS for custom button styling
    st.markdown("""
        <style>
        .button-container button {
            background-color: #007bff; /* Blue color */
            color: white;
            padding: 8px 16px;
            font-size: 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s;
            width: 100%; /* Make button take full column width */
        }
        .button-container button:hover {
            background-color: #0056b3; /* Darker blue on hover */
        }
        .button-container {
            display: flex;
            justify-content: space-around;
        }
        </style>
    """, unsafe_allow_html=True)

    # Container to hold buttons horizontally with custom styling
    st.markdown('<div class="button-container">', unsafe_allow_html=True)

    if 'page' not in st.session_state:
        st.session_state.page = "Dashboard"



        # Create five equally spaced columns for the buttons
    col1, col2, col3, col4, col5 = st.columns(5)

    # Show buttons with HTML for blue color and update session state on click
    with col1:
        if st.button("Dashboard", key="admin_dashboard"):
            st.session_state.page = "Dashboard"
    with col2:
        if st.button("Dataset", key="dataset"):
            st.session_state.page = "Dataset"
    with col3:
        if st.button("Reports", key="reports"):
            st.session_state.page = "Reports"
    with col4:
        if st.button("Image", key="image_reports"):
            st.session_state.page = "Image"
    with col5:
        if st.button("Logout", key="logout"):
            st.session_state.page = "Logout"

    # Display content based on the current value of page in session state
    if st.session_state.page == "Dashboard":
        Admin_Dashboard()
    elif st.session_state.page == "Dataset":
        dataset_page()
    elif st.session_state.page == "Reports":
        lung_cancer_prediction_page()
    elif st.session_state.page == "Image":
        segmentation_page()
    elif st.session_state.page == "Logout":
        logout()  # Call the logout function


# Function to show the dashboard
def show_dashboard(user_type="User"):
    if user_type == "User":
        show_user_dashboard()
    elif user_type == "Admin":
        show_admin_dashboard()



# --------------------------------------------------------------------------------------------------


def show_dashboard():
    st.title("Dashboard")
    st.write("--------------------------------------")

    # Sample data
    total_patients = 61938
    admitted_patients = 31786
    operational_cost = 15826
    cost_per_patient = 5.1
    doctors = 290
    avg_patients_per_dr = 26.79

    # Data for charts
    departments = ["Gynaecology", "Surgery", "Neurology", "Cardiology", "Oncology", "Orthopaedics", "Dermatology"]
    patients = [10000, 15000, 9000, 8000, 5000, 6000, 4000]
    costs = [3.2, 4.0, 3.5, 4.2, 2.5, 3.8, 2.9]
    wait_times = [38, 36, 45, 39, 41, 46, 37]
    satisfaction = [23, 54, 23]  # Percentages for satisfaction (Excellent, Okay, Poor)

    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", f"{total_patients:,}")
    col1.write(f"Patients Admitted: {admitted_patients:,}")

    col2.metric("Operational Cost", f"${operational_cost:,}K")
    col2.write(f"Avg Cost per Patient: ${cost_per_patient}K")

    col3.metric("Doctors", f"{doctors}")
    col3.write(f"Avg Patients per Dr per Month: {avg_patients_per_dr}")

    st.write("-----------------------------------")
    # Use columns for the next set of graphs to make them appear side by side
    col1, col2,col3,col4= st.columns(4)

    with col1:
        # Treemap for Department Costs
        st.subheader("Department Overview")
        fig = px.treemap(names=departments, parents=[""] * len(departments), values=patients,
                         title="Patients by Department",
                         color=costs, color_continuous_scale="teal")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Radar Chart for Available Staff per Division
        st.subheader("Available Staff per Division")
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[10, 15, 8, 12, 7, 9, 6],
            theta=departments,
            fill='toself',
            name='Number of Doctors'
        ))
        fig.add_trace(go.Scatterpolar(
            r=[8, 12, 6, 10, 5, 7, 5],
            theta=departments,
            fill='toself',
            name='Patients per Doctor'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    st.write("-------------------------------")





    # col1, col2 = st.columns(2)
    # Line + Bar Chart for Cost, Admitted, Outpatient
    with col3:
        # Average Wait Times by Division
        st.subheader("Average Wait Times by Division")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=departments, x=wait_times, orientation='h', marker_color="teal"))
        fig.update_layout(title="Average Wait Times by Division", xaxis_title="Minutes", yaxis_title="Department")
        st.plotly_chart(fig, use_container_width=True)



    with col4:
        # Pie Chart for Patient Satisfaction
        st.subheader("Patient Satisfaction")
        fig = go.Figure(data=[go.Pie(labels=["Excellent", "Okay", "Poor"], values=satisfaction, hole=0.4)])
        fig.update_traces(marker=dict(colors=["#2ca02c", "#1f77b4", "#d62728"]))
        st.plotly_chart(fig, use_container_width=True)

    col1,col2=st.columns(2)


    with col1:
        st.subheader("Monthly Statistics")
        months = pd.date_range(start="2023-01-01", periods=12, freq='M')
        admitted_patients_monthly = np.random.randint(2000, 5000, size=12)
        outpatient_patients_monthly = np.random.randint(1000, 4000, size=12)
        cost_monthly = np.random.uniform(2.5, 5.0, size=12) * 1000

        fig = go.Figure()
        fig.add_trace(go.Bar(x=months, y=cost_monthly, name="Cost", marker_color="teal"))
        fig.add_trace(
            go.Scatter(x=months, y=admitted_patients_monthly, name="Admitted Patients", line=dict(color="darkcyan")))
        fig.add_trace(
            go.Scatter(x=months, y=outpatient_patients_monthly, name="Outpatient", line=dict(color="lightseagreen")))
        fig.update_layout(barmode='group', title="Cost and Patients Over Time")
        st.plotly_chart(fig, use_container_width=True)

    st.write("------------------------------")


    with col2:
        # Doctor's Treatment Plan & Confidence in Treatment
        st.subheader("Doctor's Treatment Plan")
        plan_confidence_data = {
            "Agreement Level": ["Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"],
            "Treatment Plan": [11, 6, 49, 12, 6],
            "Confidence in Treatment": [16, 20, 15, 13, 15]
        }

        df_plan_confidence = pd.DataFrame(plan_confidence_data)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_plan_confidence["Treatment Plan"], y=df_plan_confidence["Agreement Level"],
                             orientation='h', name="Doctor's Treatment Plan", marker_color="darkcyan"))
        fig.add_trace(go.Bar(x=df_plan_confidence["Confidence in Treatment"], y=df_plan_confidence["Agreement Level"],
                             orientation='h', name="Confidence in Treatment", marker_color="lightseagreen"))
        fig.update_layout(barmode='stack', title="Doctor's Treatment Plan & Confidence Levels",
                          xaxis_title="Percentage", yaxis_title="Agreement Level")
        st.plotly_chart(fig, use_container_width=True)


# Main function to handle page routing and session management
# Main function to handle page routing and session management
def main():
    if "username" not in st.session_state:
        st.session_state["username"] = None
        st.session_state["is_admin"] = False

    # Sidebar to select User/Admin
    user_type_selection = st.sidebar.radio("Choose Access", ["Dashboard", "User", "Admin"])

    # Default page for Dashboard
    page = "dashboard"

    if user_type_selection == "User":
        # Show User Dashboard
        show_user_dashboard()
    elif user_type_selection == "Admin":
        # Admin access requires login
        if not st.session_state["is_admin"]:
            # Check if admin is registered
            conn = create_connection("admin.db")
            cursor = conn.cursor()
            cursor.execute("SELECT username FROM Admin")
            registered_admins = cursor.fetchall()
            conn.close()

            if registered_admins:
                login_page()
            else:
                registration_page()
        if st.session_state["is_admin"]:
            show_admin_dashboard()

    else:
        # Show Dashboard for any other selection
        show_dashboard()

# -------------------------------------------------------------------------------------------------------------------


# Set page title







# ------------------------------------------------------------------------------------------------------------------

def Admin_Dashboard():
    st.write("----------------------------------------------------")
    set_page("lux124.jpg")
    # Title of the dashboard
    st.title("Medical Dashboard")

    # First row - Display X-ray images and the skeleton diagram
    col1,col2,col3 = st.columns(3)

    with col1:
        # Display the first X-ray image
        st.image("path_to_xray_image1.jpg", caption="Chest X-ray", use_column_width=True)
        st.image("path_to_xray_image2.jpg", caption="Elbow X-ray", use_column_width=True)


    with col2:
        # Display the second X-ray image
        st.image("path_to_skeleton_image.jpg", caption="Full Skeleton", use_column_width=True)
        st.image("lung_cancer.jpg",caption="lung Skeleton", use_column_width=True)

    with col3:
        # Display the second X-ray image
        st.image("lung_image.jpeg", caption="details", use_column_width=True)
        st.image("lung_image21.webp", caption="lung cancer", use_column_width=True)

    st.write("----------------------------------------------------")

    # Second row - Display graphs and data visuals
    col4, col5 = st.columns(2)

    with col4:
        # Example line graph for some health metric
        st.subheader("Health Metric Over Time")
        fig, ax = plt.subplots()
        x = np.arange(1, 11)
        y = np.random.randint(50, 100, size=10)
        ax.plot(x, y, marker="o")
        ax.set_title("Bone Density")
        ax.set_xlabel("Time")
        ax.set_ylabel("Density")
        st.pyplot(fig)

    with col5:
        # Example area chart for another health metric
        st.subheader("Other Health Metric")
        fig, ax = plt.subplots()
        x = np.arange(1, 11)
        y = np.random.randint(50, 100, size=10)
        ax.fill_between(x, y, color="skyblue", alpha=0.5)
        ax.set_title("Bone Strength")
        ax.set_xlabel("Time")
        ax.set_ylabel("Strength")
        st.pyplot(fig)

    st.write("---------------------------------")
    # Bottom row - Display more metrics and controls
    col6, col7, col8 = st.columns([1, 1, 1])

    with col6:
        st.metric(label="Bone Health Score", value="85%", delta="+5%")

    with col7:
        st.metric(label="Calcium Level", value="2605 mg", delta="2.5%")

    with col8:
        st.metric(label="Vitamin D Level", value="60%", delta="+1%")

    # Additional space for customized settings or features
    st.title("Dashboard Controls")
    st.slider("Set Bone Density Threshold", 0, 100, 75)
    st.slider("Set Calcium Level Threshold", 1000, 3000, 2500)


# home page
def home_page():
    st.write("----------------------------------------")
    set_page("lux11.jpg")

    # Page title section
    st.markdown("<h2 style='text-align: center; color: white;'>Lung Cancer Prediction Application</h2>",
                unsafe_allow_html=True)
    col1,col2=st.columns(2)
    with col1:
        # Introduction section in a box style with a rounded edge
        st.markdown(
            """
            <div style="background-color: silver; padding: 20px; border-radius: 10px; text-align: center; max-width: 700px; margin: 0 auto;">
                <h4 style='color: #333333;'>Welcome to the Lung Cancer Prediction Tool</h4>
                <p style='font-size: 16px; color: #666666;'>
                    Lung cancer is one of the most common and serious types of cancer. This application predicts the lung cancer status of a patient based on various features using Machine Learning models. 
                    The app uses multiple algorithms to analyze symptoms and health data, providing a reliable prediction of lung cancer risk.
                </p>
            </div>
            """, unsafe_allow_html=True
        )


    with col2:
        st.image("dashboard.jpg")
        st.image("dashboard1.jpg")
    st.write("<hr style='border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)

    # Path to the video
    video_path = "lung_cancer_video.mp4"

    # Check if the video file exists in the current directory
    if os.path.exists(video_path):
        # Custom CSS Styling for the video container
        st.markdown(
            """
            <style>
                .video-container {
                    text-align: center;
                    max-width: 800px;
                    margin: 20px auto;
                    border-radius: 15px;
                    box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.1);
                }
                .video-container h3 {
                    font-size: 24px;
                    color: #2c3e50;
                    margin-bottom: 10px;
                }
            </style>
            """, unsafe_allow_html=True
        )

        # Title for the video
        st.markdown("<h3 class='video-container'>Lung Cancer Awareness Video</h3>", unsafe_allow_html=True)

        # Display the video using Streamlit's built-in video player
        st.video(video_path, start_time=0, muted=True, autoplay=True)
    else:
        st.write("Video file not found. Please check the file path.")

    st.write("<hr style='border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)

    # Link to American Cancer Society website
    st.markdown(
        """
        <div style="text-align: center; margin-top: 20px;">
            <a href="https://en.wikipedia.org/wiki/American_Cancer_Society" target="_blank" style="color: #4a90e2; font-size: 18px; font-weight: bold; text-decoration: none;">
                Lung Cancer: Official Website - American Cancer Society
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------------------------------------------------------------------------------------------

def about_page():
    # Page Title and Divider
    st.write("--------------------------------")
    st.title("About")
    set_page("lux101.webp")
    st.write("")
    st.subheader("Different Lung Cancer Types")
    st.write("--------------------------------")
    st.markdown(
        """
        <div style="background-color: silver; padding: 20px; border-radius: 10px; text-align: center; max-width: 700px; margin: 0 auto;">
            <h3 style='color: #333333;'>Lung Cancer</h3>
            <p style='font-size: 16px; color: #000000;'>
                Lung cancer is the leading cause of cancer deaths worldwide. Smoking is the biggest risk factor, accounting for 85% of cases. Other factors include exposure to radon, asbestos, and other carcinogens, as well as genetic predispositions. Early detection is crucial as lung cancer is more treatable when diagnosed in its initial stages. Symptoms may include a persistent cough, shortness of breath, and chest pain, but many cases are detected at advanced stages when symptoms become more pronounced.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

    st.write("")
    st.write("")
    st.write("--------------------------------")
    # Column layout for different lung cancer types
    col1, col2, col3 = st.columns(3)

    # Define a fixed width and height for images
    image_width, image_height = 200,150


    # Adenocarcinoma Card
    with col1:
        st.image("adeno.jpeg", width=image_width, caption="Adenocarcinoma")
        st.subheader("Adenocarcinoma ")

        st.markdown(
                """
                <div style="text-align:justify;">
                "Adenocarcinoma is a type of cancer that starts in the glands that line your organs. "
                "These glands secrete mucus and digestive juices. If your glandular epithelial cells begin to change or grow out of control, tumors can form. "
                "Adenocarcinoma is the most common type of cancer involving your organs."
                </div>
                """, unsafe_allow_html=True
        )

        st.write("⭐⭐⭐⭐⭐")
        if st.button("Learn More - Adenocarcinoma"):
            st.write("Learn more: [Adenocarcinoma](https://my.clevelandclinic.org)")

    # Large Cell Carcinoma Card
    with col2:
        st.image("large.png", width=image_width, caption="Large Cell Carcinoma")
        st.subheader("Large Cell Carcinoma")
        st.markdown("""
            <div style="text-align:justify;">
            "Large cell lung cancer is a type of non-small cell lung cancer (NSCLC) that is being diagnosed less often because of improved testing."
            "Large cell lung cancer is categorized as such by how the cancer cells look under a microscope. "
            "The cells do not clearly look like adenocarcinoma or squamous cell lung cancer"
            </div>
            """, unsafe_allow_html=True
        )
        st.write("⭐⭐⭐⭐")
        if st.button("Learn More - Large Cell Carcinoma"):
            st.write("Learn more: [Large Cell Carcinoma](https://www.lungevity.org)")

    # Squamous Cell Carcinoma Card
    with col3:
        st.image("squamous.jpg", width=image_width, caption="Squamous Cell Carcinoma")
        st.subheader("Squamous Cell Carcinoma")
        st.markdown(
            """
            <div style=text-align:justify;>
            "Squamous cell lung cancer, also called squamous cell carcinoma of the lung, accounts for about 30% of all lung cancers. "
            "It's a type of non-small cell lung cancer (NSCLC) that typically is treated using one or more types of therapy—surgery, "
            "radiation, chemotherapy, angiogenesis inhibitors, or immunotherapy.")
            </div>
            """,unsafe_allow_html=True
        )
        st.write("⭐⭐⭐⭐⭐")
        if st.button("Learn More - Squamous Cell Carcinoma"):
            st.write("Learn more: [Squamous Cell Carcinoma](https://www.beaumont.org)")

    # Additional Lung Cancer Information
    st.write("--------------------------------")

    st.write("For more information, visit [American Cancer Society](https://www.cancer.org/cancer/lung-cancer.html)")


# --------------------------------------------------------------------------------------------------------------------------------------

def dataset_page():
    st.write("--------------------------------")
    st.title("Dataset")
    set_page("lux6.jpg")
    st.write("---------------------------------")
    st.title("Numerical Dataset")
    data_view = st.selectbox("Choose a view", ["View Data", "Lung Cancer Patients", "Non-Cancer Patients"])

    if data_view == "View Data":
        st.dataframe(data)

    elif data_view == "Lung Cancer Patients":
        cancer_data = data[data['LUNG_CANCER'] == 1]
        st.dataframe(cancer_data)
        st.write(f"Total Lung Cancer Patients: {len(cancer_data)}")

        # Pie chart for Lung Cancer Patients
        cancer_count = len(cancer_data)
        non_cancer_count = len(data) - cancer_count
        labels = ['Lung Cancer', 'No Lung Cancer']
        values = [cancer_count, non_cancer_count]
        fig = px.pie(names=labels, values=values, title='Lung Cancer Patients')
        st.plotly_chart(fig)

    elif data_view == "Non-Cancer Patients":
        non_cancer_data = data[data['LUNG_CANCER'] == 0]
        st.dataframe(non_cancer_data)
        st.write(f"Total Non-Cancer Patients: {len(non_cancer_data)}")

        # Pie chart for Non-Cancer Patients
        non_cancer_count = len(non_cancer_data)
        cancer_count = len(data) - non_cancer_count
        labels = ['No Lung Cancer', 'Lung Cancer']
        values = [non_cancer_count, cancer_count]
        fig = px.pie(names=labels, values=values, title='Non-Cancer Patients')
        st.plotly_chart(fig)

    st.write("-----------------")
    # Dictionary of features and their descriptions
    st.write("### Dataset Features")
    st.write("-----------------")

    # Define the dataset features as a dictionary
    features = {
        "1. GENDER": "Gender of the patient [M/F]",
        "2. AGE": "Age of the patient",
        "3. SMOKING": "Whether the patient is a smoker [Y/N]",
        "4. YELLOW_FINGERS": "Yellow fingers due to smoking [Y/N]",
        "5. ANXIETY": "Anxiety levels [Y/N]",
        "6. PEER_PRESSURE": "Peer pressure experienced [Y/N]",
        "7. CHRONIC DISEASE": "Presence of chronic diseases [Y/N]",
        "8. FATIGUE": "Fatigue levels [Y/N]",
        "9. ALLERGY": "Presence of allergies [Y/N]",
        "10. WHEEZING": "Wheezing sounds [Y/N]",
        "11. ALCOHOL CONSUMING": "Alcohol consumption [Y/N]",
        "12. COUGHING": "Presence of coughing [Y/N]",
        "13. SHORTNESS OF BREATH": "Shortness of breath [Y/N]",
        "14. SWALLOWING DIFFICULTY": "Difficulty swallowing [Y/N]",
        "15. CHEST PAIN": "Chest pain [Y/N]",
        "16. LUNG_CANCER": "Lung cancer diagnosis [Y/N]"
    }

    # Create four columns
    col1, col2, col3, col4 = st.columns(4)

    # Loop through the features and divide them among the columns
    for i, (feature, description) in enumerate(features.items()):
        if i % 4 == 0:
            col1.write(f"**{feature}**: {description}")
        elif i % 4 == 1:
            col2.write(f"**{feature}**: {description}")
        elif i % 4 == 2:
            col3.write(f"**{feature}**: {description}")
        else:
            col4.write(f"**{feature}**: {description}")

    # --------------------------------------------------------------------------------------------------------------

    st.write("--------------------------------------------------------")

    # Set the image directory
    image_directory = 'Data/test'

    # Image dataset view
    st.title("Lung Cancer Image Data")
    image_type = st.selectbox("Choose Image Type",
                              ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"])

    # Dictionary to match selection to folder name
    image_type_to_folder = {
        "Adenocarcinoma": "adenocarcinoma",
        "Large Cell Carcinoma": "large.cell.carcinoma",
        "Normal": "normal",
        "Squamous Cell Carcinoma": "squamous.cell.carcinoma"
    }

    # Get folder path for the selected image type
    selected_folder = image_type_to_folder[image_type]
    folder_path = os.path.join(image_directory, selected_folder)

    # List all images in the selected folder
    images = os.listdir(folder_path)

    if images:
        # Add a slider to navigate through images
        image_index = st.slider("Select Image", 0, len(images) - 1, 0)

        # Display the selected image
        image_path = os.path.join(folder_path, images[image_index])
        img_color = Image.open(image_path)  # Open the image

        # Ensure the original image is in RGB format
        img_color = img_color.convert("RGB")

        # Convert the image to grayscale
        img_gray = img_color.convert("L")

        # Resize both images to make them smaller
        img_color = img_color.resize((200, 200))
        img_gray = img_gray.resize((200, 200))

        # Convert grayscale to a colormap for visual similarity with the uploaded image
        img_gray_np = np.array(img_gray)  # Convert grayscale PIL image to numpy array
        img_colormap = cv2.applyColorMap(img_gray_np,
                                         cv2.COLORMAP_VIRIDIS)  # Apply colormap similar to your uploaded image

        # Convert back to PIL image for display in Streamlit
        img_colormap_pil = Image.fromarray(img_colormap)

        # Display images side by side using Streamlit columns
        col1, col2 = st.columns(2)

        with col1:
            st.image(img_colormap_pil, caption=f"{image_type}  (Colormap)", use_column_width=True)

        with col2:
            st.image(img_color, caption=f"{image_type}  (graycolor)", use_column_width=True)

    else:
        st.write(f"No images found for {image_type}.")

    st.write("-----------------")
    st.write(
        "Dataset Source: [Lung Cancer numerical Dataset](https://www.kaggle.com/datasets/akashnath29/lung-cancer-dataset)")
    st.write(
        "Dataset Source: [Lung Cancer Image Dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)")

    st.write("-----------------")




# ----------------------------------------------------------------------------------------------------------------------

# Function to create the Datas table in numeric.db
def create_datas_table(conn):
    try:
        sql_create_datas_table = """
        CREATE TABLE IF NOT EXISTS Datas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            GENDER INTEGER,
            AGE INTEGER,
            SMOKING INTEGER,
            YELLOW_FINGERS INTEGER,
            ANXIETY INTEGER,
            PEER_PRESSURE INTEGER,
            CHRONIC_DISEASE INTEGER,
            FATIGUE INTEGER,
            ALLERGY INTEGER,
            WHEEZING INTEGER,
            ALCOHOL_CONSUMING INTEGER,
            COUGHING INTEGER,
            SHORTNESS_OF_BREATH INTEGER,
            SWALLOWING_DIFFICULTY INTEGER,
            CHEST_PAIN INTEGER,
            prediction TEXT
        );
        """
        c = conn.cursor()
        c.execute(sql_create_datas_table)
        conn.commit()
    except Error as e:
        st.error(f"Error: {e}")

# Initialize the SQLite database and table for numeric.db
numeric_database = "numeric.db"
conn_numeric = create_connection(numeric_database)
if conn_numeric is not None:
    create_datas_table(conn_numeric)

@st.cache_resource
def load_best_model():
    best_model = pickle.load(open('numerical_best_model.pkl', 'rb'))
    return best_model



def lung_cancer_prediction_page():
    st.write("--------------------------------")
    st.title("Lung Cancer Prediction")
    st.write("Enter patient details for prediction.")
    set_page("lux33.avif")


    col1,col2=st.columns(2)
    with col1:
        id = st.text_input("Patient_ID")
    with col2:
        patient_name = st.text_input("Patient Name")


    # Group the features into three columns
    col1, col2, col3 = st.columns(3)

    # Divide into four columns
    col1, col2, col3, col4 = st.columns(4)

    # Input for GENDER, AGE, SMOKING, YELLOW FINGERS
    with col1:
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True, key="gender")
        GENDER = 1 if gender == "Male" else 0
    with col2:
        AGE = st.number_input("Age", 1, 100, key="age")
    with col3:
        SMOKING = 1 if st.radio("Smoking", ["Yes", "No"], horizontal=True, key="smoking") == "Yes" else 0
    with col4:
        YELLOW_FINGERS = 1 if st.radio("Yellow Fingers", ["Yes", "No"], horizontal=True,
                                       key="yellow_fingers") == "Yes" else 0

    # Input for ANXIETY, PEER PRESSURE, CHRONIC DISEASE, FATIGUE
    with col1:
        ANXIETY = 1 if st.radio("Anxiety", ["Yes", "No"], horizontal=True, key="anxiety") == "Yes" else 0
    with col2:
        PEER_PRESSURE = 1 if st.radio("Peer Pressure", ["Yes", "No"], horizontal=True,
                                      key="peer_pressure") == "Yes" else 0
    with col3:
        CHRONIC_DISEASE = 1 if st.radio("Chronic Disease", ["Yes", "No"], horizontal=True,
                                        key="chronic_disease") == "Yes" else 0
    with col4:
        FATIGUE = 1 if st.radio("Fatigue", ["Yes", "No"], horizontal=True, key="fatigue") == "Yes" else 0

    # Input for ALLERGY, WHEEZING, ALCOHOL CONSUMING, COUGHING
    with col1:
        ALLERGY = 1 if st.radio("Allergy", ["Yes", "No"], horizontal=True, key="allergy") == "Yes" else 0
    with col2:
        WHEEZING = 1 if st.radio("Wheezing", ["Yes", "No"], horizontal=True, key="wheezing") == "Yes" else 0
    with col3:
        ALCOHOL_CONSUMING = 1 if st.radio("Alcohol Consuming", ["Yes", "No"], horizontal=True,
                                          key="alcohol_consuming") == "Yes" else 0
    with col4:
        COUGHING = 1 if st.radio("Coughing", ["Yes", "No"], horizontal=True, key="coughing") == "Yes" else 0

    # Input for SHORTNESS OF BREATH, SWALLOWING DIFFICULTY, CHEST PAIN
    with col1:
        SHORTNESS_OF_BREATH = 1 if st.radio("Shortness of Breath", ["Yes", "No"], horizontal=True,
                                            key="shortness_of_breath") == "Yes" else 0
    with col2:
        SWALLOWING_DIFFICULTY = 1 if st.radio("Swallowing Difficulty", ["Yes", "No"], horizontal=True,
                                              key="swallowing_difficulty") == "Yes" else 0
    with col3:
        CHEST_PAIN = 1 if st.radio("Chest Pain", ["Yes", "No"], horizontal=True, key="chest_pain") == "Yes" else 0

    # Create input features DataFrame
    input_features = {
        "GENDER": GENDER,
        "AGE": AGE,
        "SMOKING": SMOKING,
        "YELLOW_FINGERS": YELLOW_FINGERS,
        "ANXIETY": ANXIETY,
        "PEER_PRESSURE": PEER_PRESSURE,
        "CHRONIC_DISEASE": CHRONIC_DISEASE,
        "FATIGUE": FATIGUE,
        "ALLERGY": ALLERGY,
        "WHEEZING": WHEEZING,
        "ALCOHOL_CONSUMING": ALCOHOL_CONSUMING,
        "COUGHING": COUGHING,
        "SHORTNESS_OF_BREATH": SHORTNESS_OF_BREATH,
        "SWALLOWING_DIFFICULTY": SWALLOWING_DIFFICULTY,
        "CHEST_PAIN": CHEST_PAIN,
    }

    input_df = pd.DataFrame([input_features])

    # Assuming you have a dataset called 'data' loaded elsewhere in your app
    X = data.drop("LUNG_CANCER", axis=1)
    y = data["LUNG_CANCER"]

    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    # Split and preprocess data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    input_scaled = scaler.transform(input_df)

    # Train the SVC model
    load_best_model = SVC(probability=True)

    load_best_model.fit(X_train, y_train)

    # Prediction
    prediction = load_best_model.predict(input_scaled)[0]
    prediction_proba = load_best_model.predict_proba(input_scaled)[0]

    st.write("-----------------------------------------------------------")
    st.write("**Prediction/Save Patient Details**")

    # # Create two columns for the buttons
    # col1, col2, col3, col4 = st.columns(4)

    # "Predict" button in col1
    # with col1:
    if st.button("Predict"):
        result = "likely to have lung cancer" if prediction == 0 else "not likely to have lung cancer"
        if prediction == 0:
            st.success("The patient is likely to have lung cancer.")
            st.image("broken-heart.gif")
        else:
            st.error("The patient is not likely to have lung cancer.")
            st.balloons()

        # Voice Output
        engine = pyttsx3.init()
        engine.say(f"The patient is predicted to be {result}")
        engine.runAndWait()

        # Display probabilities
        st.write(
            f"Prediction Probability: Lung Cancer = {prediction_proba[0]:.2f}, No Lung Cancer = {prediction_proba[1]:.2f}")

        # Model performance on test data
        y_pred_test = load_best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_test)
        st.write(f"Model Accuracy on Test Data: {accuracy:.2%}")

        # Save prediction result to database
        conn_numeric.execute("""
            INSERT INTO Datas(patient_name, GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE,
                                     FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH,
                                     SWALLOWING_DIFFICULTY, CHEST_PAIN, prediction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_name, GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE,
            FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH,
            SWALLOWING_DIFFICULTY, CHEST_PAIN, result
        ))
        conn_numeric.commit()
        st.success("Prediction and details saved to database.")

    # "Save" button in col2
    # with col2:
    #     if st.button("Update"):
    #
    #
    #         # Input field to retrieve patient by ID
    #         id = st.number_input("Enter Patient ID", min_value=1, step=1, key="patient_id")
    #         if st.button("Search by ID"):
    #             cursor = conn_numeric.cursor()
    #             cursor.execute("SELECT * FROM Datas WHERE id=?", (id,))
    #             patient_data = cursor.fetchone()
    #             if patient_data:
    #                 st.session_state.patient_record = patient_data  # Save record to session state
    #                 st.write("Patient details loaded.")
    #             else:
    #                 st.error("Patient ID not found.")
    #                 st.session_state.patient_record = None  # Clear session state if ID not found
    #
    #         # Check if patient_record exists in session state, else use default values
    #         patient_data = st.session_state.get("patient_record", [None] * 17)
    #
    # with col3:
    #     st.button("Delete")

    st.write("----------------------------------------------------")

    # col1,col2=st.columns(2)
    # with col1:
    #     st.button("Patient Details")
    #     # Check the state to decide whether to show data
    #     if st.session_state.get("show_data", False):  # Defaults to False if not set
    #         # Create a cursor object from the connection
    #         cursor = conn_numeric.cursor()
    #
    #         # Execute the query to fetch all records
    #         cursor.execute("SELECT * FROM Datas")
    #
    #         # Fetch all records
    #         records = cursor.fetchall()
    #
    #         # Get column names from the cursor description
    #         columns = [desc[0] for desc in cursor.description]
    #
    #         # Create a DataFrame to display the records
    #         df = pd.DataFrame(records, columns=columns)
    #
    #         # Display the DataFrame
    #         st.write(df)





    st.write("**Show Patient Details**")

    #Show all entries from DB

    # Define the "Show" and "Reset" buttons in two columns
    col1, col2= st.columns(2)

    # Show button
    with col1:
        if st.button("Show"):
            # Toggle the visibility state to show the data
            st.session_state.show_data = True
        # Reset button
    with col2:
        if st.button("close"):
            # Toggle the visibility state to hide the data
            st.session_state.show_data = False

    # with col3:
    #     st.write("**Patient Details**")
    #
    #     #Layout for input and button to be placed horizontally
    #     col1, col2 = st.columns([2, 1])  # The columns will take relative space, 2 for input, 1 for button
    #
    #     #Input for specific Patient ID (in the first column)
    #
    #     patient_id = st.number_input("Patient ID ", min_value=1, step=1)
    #     show_button = st.button("Patient ID")
    #
    #     # When the button is clicked, execute the query and show details
    #     if show_button:
    #         cursor = conn_numeric.cursor()
    #
    #         # Execute the query to fetch the specific patient record by ID
    #         cursor.execute("SELECT * FROM Datas WHERE id = ?", (patient_id,))
    #         patient_record = cursor.fetchone()  # Fetch the record
    #
    #         if patient_record:
    #             # Get column names from the cursor description
    #             columns = [desc[0] for desc in cursor.description]
    #
    #             # Create a DataFrame to display the specific patient record
    #             patient_df = pd.DataFrame([patient_record], columns=columns)
    #
    #             # Display the DataFrame with patient details
    #             st.write("Patient Details:")
    #             st.write(patient_df)
    #         else:
    #             st.error(f"No patient found with ID: {patient_id}. Please try again.")
    #
    #         # Close cursor after the query
    #         cursor.close()

    # Check the state to decide whether to show data
    if st.session_state.get("show_data", False):  # Defaults to False if not set
        # Create a cursor object from the connection
        cursor = conn_numeric.cursor()

        # Execute the query to fetch all records
        cursor.execute("SELECT * FROM Datas")

        # Fetch all records
        records = cursor.fetchall()

        # Get column names from the cursor description
        columns = [desc[0] for desc in cursor.description]

        # Create a DataFrame to display the records
        df = pd.DataFrame(records, columns=columns)

        # Display the DataFrame
        st.write(df)

    st.write("----------------------------------------------------")

#     cursor = conn_numeric.cursor()



    st.write("**Update/Delete Patient Details**")

    # Input field to retrieve patient by ID
    patient_id = st.number_input("Enter Patient ID", min_value=1, step=1, key="patient_id")
    if st.button("Search by ID"):
        cursor = conn_numeric.cursor()
        cursor.execute("SELECT * FROM Datas WHERE id=?", (patient_id,))
        patient_data = cursor.fetchone()
        if patient_data:
            st.session_state.patient_record = patient_data  # Save record to session state
            st.write("Patient details loaded.")

            # Display patient details in a table
            st.table({
                "Patient Details": [
                    "ID", "Name", "Gender", "Age", "Smoking", "Yellow Fingers", "Anxiety", "Peer Pressure",
                    "Chronic Disease", "Fatigue", "Allergy", "Wheezing", "Alcohol Consuming", "Coughing",
                    "Shortness of Breath", "Swallowing Difficulty", "Chest Pain"
                ],
                "Values": list(patient_data)
            })

        else:
            st.error("Patient ID not found.")
            st.session_state.patient_record = None  # Clear session state if ID not found

# Check if patient_record exists in session state, else use default values
        patient_data = st.session_state.get("patient_record", [None] * 17)

        # Collect user input with pre-filled values if available
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Patient Name", patient_data[1] if len(patient_data) > 1 else "",
                                         key="patient_name_input")
        with col2:
            AGE = st.text_input("Age", patient_data[3] if len(patient_data) > 3 else "", key="age_input")

        # Create columns for select boxes
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            GENDER = st.selectbox(
                "Gender", ["M", "F"],
                index=["M", "F"].index(patient_data[2]) if len(patient_data) > 2 and patient_data[2] in ["M", "F"] else 0,
                key="gender_select"
            )
            SMOKING = st.selectbox(
                "Smoking", ["Yes", "No"],
                index=["Yes", "No"].index(patient_data[4]) if len(patient_data) > 4 and patient_data[4] in ["Yes",
                                                                                                            "No"] else 0,
                key="smoking_select"
            )
            YELLOW_FINGERS = st.selectbox(
                "Yellow Fingers", ["Yes", "No"],
                index=["Yes", "No"].index(patient_data[5]) if len(patient_data) > 5 and patient_data[5] in ["Yes",
                                                                                                            "No"] else 0,
                key="yellow_fingers_select"
            )

        with col2:
            ANXIETY = st.selectbox(
                "Anxiety", ["Yes", "No"],
                index=["Yes", "No"].index(patient_data[6]) if len(patient_data) > 6 and patient_data[6] in ["Yes",
                                                                                                            "No"] else 0,
                key="anxiety_select"
            )
            PEER_PRESSURE = st.selectbox(
                "Peer Pressure", ["Yes", "No"],
                index=["Yes", "No"].index(patient_data[7]) if len(patient_data) > 7 and patient_data[7] in ["Yes",
                                                                                                            "No"] else 0,
                key="peer_pressure_select"
            )
            CHRONIC_DISEASE = st.selectbox(
                "Chronic Disease", ["Yes", "No"],
                index=["Yes", "No"].index(patient_data[8]) if len(patient_data) > 8 and patient_data[8] in ["Yes",
                                                                                                            "No"] else 0,
                key="chronic_disease_select"
            )

        with col3:
            FATIGUE = st.selectbox(
                "Fatigue", ["Yes", "No"],
                index=["Yes", "No"].index(patient_data[9]) if len(patient_data) > 9 and patient_data[9] in ["Yes",
                                                                                                            "No"] else 0,
                key="fatigue_select"
            )
            ALLERGY = st.selectbox(
                "Allergy", ["Yes", "No"],
                index=["Yes", "No"].index(patient_data[10]) if len(patient_data) > 10 and patient_data[10] in ["Yes",
                                                                                                               "No"] else 0,
                key="allergy_select"
            )
            WHEEZING = st.selectbox(
                "Wheezing", ["Yes", "No"],
                index=["Yes", "No"].index(patient_data[11]) if len(patient_data) > 11 and patient_data[11] in ["Yes",
                                                                                                               "No"] else 0,
                key="wheezing_select"
            )

        with col4:
            ALCOHOL_CONSUMING = st.selectbox(
                "Alcohol Consuming", ["Yes", "No"],
                index=["Yes", "No"].index(patient_data[12]) if len(patient_data) > 12 and patient_data[12] in ["Yes",
                                                                                                               "No"] else 0,
                key="alcohol_select"
            )
            COUGHING = st.selectbox(
                "Coughing", ["Yes", "No"],
                index=["Yes", "No"].index(patient_data[13]) if len(patient_data) > 13 and patient_data[13] in ["Yes",
                                                                                                               "No"] else 0,
                key="coughing_select"
            )
            SHORTNESS_OF_BREATH = st.selectbox(
                "Shortness of Breath", ["Yes", "No"],
                index=["Yes", "No"].index(patient_data[14]) if len(patient_data) > 14 and patient_data[14] in ["Yes",
                                                                                                               "No"] else 0,
                key="shortness_breath_select"
            )

        with col5:
            SWALLOWING_DIFFICULTY = st.selectbox(
                "Swallowing Difficulty", ["Yes", "No"],
                index=["Yes", "No"].index(patient_data[15]) if len(patient_data) > 15 and patient_data[15] in ["Yes",
                                                                                                               "No"] else 0,
                key="swallowing_select"
            )
            CHEST_PAIN = st.selectbox(
                "Chest Pain", ["Yes", "No"],
                index=["Yes", "No"].index(patient_data[16]) if len(patient_data) > 16 and patient_data[16] in ["Yes",
                                                                                                               "No"] else 0,
                key="chest_pain_select"
            )

    # "Update" and "Delete" buttons
    col_update, col_delete = st.columns(2)

    with col_update:
        if st.button("Update"):
            if st.session_state.get("patient_record"):
                input_data = [
                    1 if GENDER == "F" else 0,  # Gender
                    AGE,
                    1 if SMOKING == "Yes" else 0,
                    1 if YELLOW_FINGERS == "Yes" else 0,
                    1 if ANXIETY == "Yes" else 0,
                    1 if PEER_PRESSURE == "Yes" else 0,
                    1 if CHRONIC_DISEASE == "Yes" else 0,
                    1 if FATIGUE == "Yes" else 0,
                    1 if ALLERGY == "Yes" else 0,
                    1 if WHEEZING == "Yes" else 0,
                    1 if ALCOHOL_CONSUMING == "Yes" else 0,
                    1 if COUGHING == "Yes" else 0,
                    1 if SHORTNESS_OF_BREATH == "Yes" else 0,
                    1 if SWALLOWING_DIFFICULTY == "Yes" else 0,
                    1 if CHEST_PAIN == "Yes" else 0
                ]

                if len(input_data) == 15:
                    prediction_result = load_best_model.predict([input_data])
                    result = "likely to have lung cancer" if prediction_result[
                                                                 0] == 1 else "not likely to have lung cancer"

                    cursor = conn_numeric.cursor()
                    cursor.execute("""
                        UPDATE Datas SET
                        patient_name = ?, GENDER = ?, AGE = ?, SMOKING = ?, YELLOW_FINGERS = ?, ANXIETY = ?,
                        PEER_PRESSURE = ?, CHRONIC_DISEASE = ?, FATIGUE = ?, ALLERGY = ?, WHEEZING = ?,
                        ALCOHOL_CONSUMING = ?, COUGHING = ?, SHORTNESS_OF_BREATH = ?, SWALLOWING_DIFFICULTY = ?,
                        CHEST_PAIN = ?, prediction = ? WHERE id = ?
                    """, (
                        patient_name,
                        1 if GENDER == "M" else 0,
                        AGE,
                        1 if SMOKING == "Yes" else 0,
                        1 if YELLOW_FINGERS == "Yes" else 0,
                        1 if ANXIETY == "Yes" else 0,
                        1 if PEER_PRESSURE == "Yes" else 0,
                        1 if CHRONIC_DISEASE == "Yes" else 0,
                        1 if FATIGUE == "Yes" else 0,
                        1 if ALLERGY == "Yes" else 0,
                        1 if WHEEZING == "Yes" else 0,
                        1 if ALCOHOL_CONSUMING == "Yes" else 0,
                        1 if COUGHING == "Yes" else 0,
                        1 if SHORTNESS_OF_BREATH == "Yes" else 0,
                        1 if SWALLOWING_DIFFICULTY == "Yes" else 0,
                        1 if CHEST_PAIN == "Yes" else 0,
                        result,
                        patient_id
                    ))
                    conn_numeric.commit()
                    st.success("Patient details and prediction result updated successfully!")
                else:
                    st.error("Ensure all 15 features are properly entered.")

    with col_delete:
        if st.button("Delete"):
            if st.session_state.get("patient_record"):
                cursor = conn_numeric.cursor()
                cursor.execute("DELETE FROM Datas WHERE id = ?", (patient_id,))
                conn_numeric.commit()
                st.success("Patient details deleted successfully.")

    # Update and Delete Patient Details

    # Divider line
    st.write("--------------------------------------------------------------------")

    if st.button("Reset"):
        st.session_state.show_data = False
        st.session_state.save_data = False
        st.session_state.update_data = False
        st.session_state.delete_data = False


# -----------------------------------------------------------------------------------------------------------------------------------


# Database setup
def create_data_table(conn):
    try:
        sql_create_data_table = """
        CREATE TABLE IF NOT EXISTS Data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            gender TEXT,
            age INTEGER,
            segmented_image_path TEXT,
            detection TEXT
        );
        """
        c = conn.cursor()
        c.execute(sql_create_data_table)
        conn.commit()
    except Error as e:
        st.error(f"Error: {e}")


# Initialize SQLite database and table
image_database = "image.db"
conn_image = sqlite3.connect(image_database)
if conn_image is not None:
    create_data_table(conn_image)


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_model_effinetb0.h5')
    return model


def preprocess_image(image):
    image = np.array(image)
    resized_image = cv2.resize(image, (350, 350))
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    final_image = rgb_image / 255.0
    return np.expand_dims(final_image, axis=0)


def save_image(image, filename):
    cv2.imwrite(filename, image)


def process_image(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    filtered_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(filtered_image, 100, 200)
    _, segmented_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return gray_image, filtered_image, edges, segmented_image


# Streamlit app
def segmentation_page():
    st.write("---------------------------------")
    st.title("Lung Cancer Image Segmentation and Detection")

    set_page("lux1.avif")

    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    uploaded_file = st.file_uploader("Upload a lung CT scan image...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file

    if (st.button("Reset")):
        st.session_state.uploaded_file = None
        st.session_state.processed_images = {}
        st.success("All results cleared!")

    # Function to convert an image to a base64 string for display
    def image_to_base64(image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    if st.session_state.uploaded_file:
        input_image = Image.open(st.session_state.uploaded_file)
        img_base64 = image_to_base64(input_image)

        st.markdown(
            f'<img src="data:image/jpeg;base64,{img_base64}" alt="Input Image" style="display: block; margin-left: auto; margin-right: auto; width: 600px;">',
            unsafe_allow_html=True,
        )

        # Preprocessing, model prediction, and segmentation
        preprocessed_image = preprocess_image(input_image)
        model = load_model()
        prediction = model.predict(preprocessed_image)
        class_idx = np.argmax(prediction, axis=1)
        cancer_types = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']
        predicted_cancer = cancer_types[class_idx[0]]
        st.subheader(f"Detection Result: {predicted_cancer}")

        patient_name = st.text_input("Patient Name:")
        gender = st.selectbox("Gender:", ("Male", "Female", "Other"))
        age = st.number_input("Age:", min_value=0, step=1)

        gray_image, filtered_image, edges, segmented_image = process_image(input_image)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(gray_image, caption="Grayscale Image", use_column_width=True)
        with col2:
            st.image(filtered_image, caption="Filtered Image", use_column_width=True)
        with col3:
            st.image(edges, caption="Edge Detected Image", use_column_width=True)
        with col4:
            st.image(segmented_image, caption="Segmented Image", use_column_width=True)

        if st.button("Save Results"):
            if patient_name and uploaded_file:
                segmented_image_path = "segmented_image.png"
                save_image(np.array(input_image), segmented_image_path)
                cursor = conn_image.cursor()
                cursor.execute("""
                INSERT INTO Data (patient_name, gender, age, segmented_image_path, detection) VALUES (?, ?, ?,?, ?)
                """, (patient_name, gender, age, segmented_image_path, predicted_cancer))
                conn_image.commit()
                st.success("Results saved to the database!")
            else:
                st.error("Please enter patient details and upload an image before saving.")

        st.write("-----------------------------------------------------------")

        st.subheader("Manage Patient Records")

        st.write("--------------------------------------------------------")

        # Input field for Patient ID to retrieve, update, or delete records
        patient_id = st.number_input("Enter Patient ID to Retrieve Details:", min_value=1, step=1)

        # Variables to store the retrieved data
        patient_name, gender, age, segmented_image_path = None, None, None, None

        # Retrieve and display patient details
        if st.button("Retrieve"):
            if patient_id:
                cursor = conn_image.cursor()
                cursor.execute("SELECT patient_name, gender, age, segmented_image_path FROM Data WHERE id=?",
                               (patient_id,))
                result = cursor.fetchone()

                if result:
                    patient_name, gender, age, segmented_image_path = result
                    st.write(f"**Patient Name:** {patient_name}")
                    st.write(f"**Gender:** {gender}")
                    st.write(f"**Age:** {age}")
                    st.write(f"**Detection:** {predicted_cancer}")

                    # Display the segmented image if available
                    if segmented_image_path:
                        image = Image.open(segmented_image_path)
                        st.image(image, caption="Segmented Image", use_column_width=True, width=300)
                else:
                    st.error("Patient ID not found.")
            else:
                st.error("Please enter a valid Patient ID.")

        # Set initial values for the update fields based on retrieved data
        updated_name = st.text_input("New Patient Name:", value=patient_name if patient_name else "")
        updated_gender = st.selectbox("New Gender:", ("Male", "Female", "Other"),
                                      index=["Male", "Female", "Other"].index(gender) if gender else 0)
        updated_age = st.number_input("New Age:", min_value=0, step=1, value=age if age is not None else 0)
        new_image_file = st.file_uploader("Upload New Image for Update", type=["jpg", "png", "jpeg"])

        # Columns for Update and Delete buttons
        col1, col2 = st.columns(2)

        with col1:
            # Update functionality
            if st.button("Update"):
                if patient_id:
                    if updated_name or updated_gender or updated_age or new_image_file:
                        update_fields = []
                        update_values = []

                        if updated_name:
                            update_fields.append("patient_name=?")
                            update_values.append(updated_name)
                        if updated_gender:
                            update_fields.append("gender=?")
                            update_values.append(updated_gender)
                        if updated_age:
                            update_fields.append("age=?")
                            update_values.append(updated_age)
                        if new_image_file:
                            new_image = Image.open(new_image_file)
                            new_image_path = "updated_segmented_image.png"
                            save_image(np.array(new_image), new_image_path)
                            update_fields.append("segmented_image_path=?")
                            update_values.append(new_image_path)

                        # Append patient ID to the end of the values list
                        update_values.append(patient_id)

                        # Execute the update query
                        cursor = conn_image.cursor()
                        cursor.execute(f"UPDATE Data SET {', '.join(update_fields)} WHERE id=?", tuple(update_values))
                        conn_image.commit()
                        st.success("Patient record updated!")
                    else:
                        st.error("Please provide at least one new value to update.")
                else:
                    st.error("Please enter a valid Patient ID.")

        with col2:
            # Delete functionality
            if st.button("Delete"):
                if patient_id:
                    cursor = conn_image.cursor()
                    cursor.execute("DELETE FROM Data WHERE id=?", (patient_id,))
                    conn_image.commit()
                    st.success("Patient record deleted!")
                else:
                    st.error("Please enter a valid Patient ID.")
        st.write("---------------------------------------------------------------")


# -------------------------------------------------------------------------------------------------------------------------------

def tips_page():
    st.write("--------------------------------")
    st.write("""
        ## Preventing Lung Cancer
        ----------------------------------------------
        1. Avoid smoking and exposure to secondhand smoke.
        2. Test your home for radon and reduce exposure.
        3. Avoid exposure to asbestos and other carcinogens.
        4. Maintain a healthy diet and regular exercise.
        5. Get regular check-ups and report any symptoms to your doctor.
    """)
    # set_page("lux25.jpeg")
    set_page("lux123.avif")
    st.write("---------------------")
    # st.image("lung_cancer.jpg", width=500)
    # st.write("---------------------")

    # Column layout for different lung cancer types
    col1, col2, col3 = st.columns(3)

    # Define a fixed width and height for images
    image_width, image_height = 200, 150

    # Adenocarcinoma Card
    with col1:
        st.image("adeno.jpeg", width=image_width, caption="Adenocarcinoma")
        st.subheader("Adenocarcinoma cancer")
        st.markdown(
            """
            <div style="text-align:justify;">
            "There are some factors that might increase the risk for these cancers"
            )
            </div>
            """, unsafe_allow_html=True
        )
        st.write("⭐⭐⭐⭐⭐")
        if st.button("Learn More - Adenocarcinoma"):
            st.write("Learn more: [Prevention](https://www.cancer.org)")

    # Large Cell Carcinoma Card
    with col2:
        st.image("large.png", width=image_width, caption="Large Cell Carcinoma")
        st.subheader("Large Cell Carcinoma")
        st.markdown(
            """
            <div style="text-align:justify;">
            "Depending on the stage of large cell carcinoma and how far it has spread."
            </div>
            """,unsafe_allow_html=True
        )
        st.write("⭐⭐⭐⭐")
        if st.button("Learn More - Large Cell Carcinoma"):
            st.write("Learn more: [Prevention](https://www.medicalnewstoday.com)")

    # Squamous Cell Carcinoma Card
    with col3:
        st.image("squamous.jpg", width=image_width, caption="Squamous Cell Carcinoma")
        st.subheader("Squamous Cell Carcinoma")
        st.markdown(
            """
            <div style="text-align:justify;">
            "Lung cancer is largely a preventable disease assist patients smoking cessation.")
            </div>
            """, unsafe_allow_html=True
        )
        st.write("⭐⭐⭐⭐⭐")
        if st.button("Learn More - Squamous Cell Carcinoma"):
            st.write("Learn more: [Prevention](https://www.va.gov)")

    st.write("-------------------------------------------")
    st.write("@Lung cancer : https://www.cdc.gov/lung-cancer/prevention/index.html")



# -------------------------------------------------------------------------------------------------------------------------------


def chatbot_response(user_message, doctor_specialization):
    # General chatbot responses
    general_responses = {
        "hello doctor": "Hello! How can I assist you today?",
        "how are you?": "I'm here to help you as much as I can!",
        "bye": "Goodbye! Take care of your health!",
        "appointment please": "To schedule an appointment, please share your preferred date and time.",
        "prevention tips": "Avoid smoking, eat a balanced diet, exercise, and get regular check-ups.",
    }

    # Specialized responses for different doctors
    specialized_responses = {
        "Cardiologist": {
            "chest pain": "Chest pain can be serious. I recommend a check-up as soon as possible.",
            "heart health": "Maintaining a healthy weight and reducing stress helps your heart!",
        },
        "Dermatologist": {
            "skin issues": "For skin concerns, moisturize regularly and avoid irritants.",
            "acne": "Using a gentle cleanser and avoiding oily products can help manage acne.",
        },
        "Pediatrician": {
            "child health": "Regular check-ups and a balanced diet are key for a child’s growth.",
            "fever": "Monitor the fever, stay hydrated, and seek care if it persists.",
        },
    }

    # Merge general and specialized responses
    responses = {**general_responses, **specialized_responses.get(doctor_specialization, {})}

    # Return response or fallback message
    return responses.get(user_message.lower(), "I'm sorry, I didn’t understand that. Could you rephrase?")


def listen_to_user():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.write("Listening... Speak now.")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            user_message = recognizer.recognize_google(audio)
            return user_message
    except Exception as e:
        st.error(f"Voice input error: {e}")
        return ""


def speak_response(response):
    engine = pyttsx3.init()
    engine.say(response)
    engine.runAndWait()


def chatbot_page():
    st.write("----------------------------------------")
    st.title("Doctor Chatbot")
    set_page("chatbot.webp")

    # Doctors' data
    doctors = {
        "Dr. Sarah Johnson": {"image": "doctor.jpg", "specialization": "Cardiologist"},
        "Dr. John Smith": {"image": "doctor2.jpg", "specialization": "Dermatologist"},
        "Dr. Emily Davis": {"image": "doctor3.jpg", "specialization": "Pediatrician"},
    }

    # Select doctor
    selected_doctor = st.selectbox("Choose a doctor:", list(doctors.keys()))

    # Check if the doctor has changed
    if "selected_doctor" not in st.session_state or st.session_state.selected_doctor != selected_doctor:
        # Reset chat history and update the selected doctor
        st.session_state.selected_doctor = selected_doctor
        st.session_state.chat_history = []

    doctor_info = doctors[selected_doctor]

    # Display doctor info
    col1, col2 = st.columns([1, 2])
    with col1:
        doctor_image = Image.open(doctor_info["image"])
        st.image(doctor_image, caption=selected_doctor, use_column_width=True)
    with col2:
        st.subheader(selected_doctor)
        st.write(f"Specialization: {doctor_info['specialization']}")

    # Chat history display
    st.subheader(f"Chat with {selected_doctor}")
    for chat in st.session_state.chat_history:
        if chat["type"] == "user":
            st.write(f"**You:** {chat['message']}")
        else:
            st.write(f"**{selected_doctor}:** {chat['message']}")

    # Text input for user message
    user_message = st.text_input("Enter your message:")

    # Voice input button
    if st.button("Speak Now"):
        user_message = listen_to_user()
        st.write(f"**You (via voice):** {user_message}")

    # Process message
    if user_message:
        st.session_state.chat_history.append({"type": "user", "message": user_message})

        # Generate doctor response
        response = chatbot_response(user_message, doctor_info["specialization"])
        st.session_state.chat_history.append({"type": "doctor", "message": response})

        # Display and speak response
        st.write(f"**{selected_doctor}:** {response}")
        speak_response(response)


if __name__ == "__main__":
    main()