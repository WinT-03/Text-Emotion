import streamlit as st
from predict import predict_emotion

st.set_page_config(page_title="Text Emotion App", layout="centered")

st.title("🎭 Nhận diện cảm xúc từ văn bản")
st.markdown("Nhập một đoạn văn tiếng Anh bất kỳ và xem cảm xúc tương ứng là gì!")

user_input = st.text_area("Nhập văn bản tại đây", "")

if st.button("Dự đoán cảm xúc"):
    if user_input.strip() == "":
        st.warning("Vui lòng nhập nội dung trước khi dự đoán.")
    else:
        with st.spinner("Đang dự đoán..."):
            emotion = predict_emotion(user_input)
        st.success(f"✨ Cảm xúc được dự đoán: **{emotion.upper()}**")
