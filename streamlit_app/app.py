import streamlit as st
from model_helper import predict

st.set_page_config(
    page_title="Vehicle Damage Detection",
    page_icon="🚗",
    layout="centered"
)

st.markdown("""
    <style>
        .title-text {
            font-size: 38px;
            font-weight: 700;
            color: #ffffff;
            text-align: center;
        }
        .subtitle-text {
            font-size: 18px;
            color: #f1f1f1;
            text-align: center;
        }
        .header-box {
            background: linear-gradient(135deg, #1f4037, #99f2c8);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
        }
        .prediction-box {
            background-color: #f0f9ff;
            padding: 20px;
            border-radius: 12px;
            border-left: 6px solid #1f77b4;
            font-size: 20px;
            font-weight: 600;
            color: #0f172a;
        }
        .footer-text {
            text-align: center;
            color: gray;
            font-size: 14px;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="header-box">
        <div class="title-text">🚗 Vehicle Damage Detection</div>
        <div class="subtitle-text">
            AI-powered car damage classification using Deep Learning
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("### 📤 Upload Vehicle Image")
upload_image = st.file_uploader(
    "Supported formats: JPG, PNG",
    type=["jpg", "png"]
)

if upload_image:
    img_path = "temp_img.jpg"

    with open(img_path, "wb") as f:
        f.write(upload_image.getbuffer())

    st.markdown("### 🖼 Uploaded Image")
    st.image(upload_image, caption="Uploaded Vehicle Image")

    with st.spinner("🔍 Analyzing damage..."):
        prediction = predict(img_path)

    st.markdown("### 📊 Prediction Result")
    st.markdown(
        f"""
        <div class="prediction-box">
            ✅ <strong>Detected Damage Type:</strong><br>
            {prediction}
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("""
    <div class="footer-text">
        Built with ❤️ using PyTorch & Streamlit<br>
        Deep Learning Project – Vehicle Damage Classification
    </div>
""", unsafe_allow_html=True)
