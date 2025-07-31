import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import base64

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

fields = {"username": "Username", "password": "Password"}
name, auth_status, username = authenticator.login(fields=fields, location='main')

if auth_status is False:
    st.error('Username or password is incorrect')
elif auth_status is None:
    st.warning('Please enter your username and password')
elif auth_status:
    st.sidebar.success(f"Welcome, {name}!")

    def get_base64_image(image_path):
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        return encoded

    def set_background(image_path):
        base64_str = get_base64_image(image_path)
        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{base64_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)

    set_background("background.jpg")

    st.sidebar.title("Navigator")
    page = st.sidebar.radio("Select a Page", ["Telecom Customer Churn Predictor", "Telecom Customer Churn Dashboard", "AI Assistant"])
    authenticator.logout('Logout', 'sidebar')

    if page == "Telecom Customer Churn Predictor":
        st.title('Telecom Customer Churn Predictor')

    elif page == "Telecom Customer Churn Dashboard":
        st.title('Telecom Customer Churn Dashboard')

    elif page == "AI Assistant":
        st.title('AI Assistant')