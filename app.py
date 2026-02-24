import streamlit as st
import numpy as np
from PIL import Image
import joblib
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

@st.cache_resource
def load_models():
    resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    pca = joblib.load('data/feature/pca_transformer.pkl')
    knn = joblib.load('data/results/knn_model.pkl')
    return resnet, pca, knn

def process_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_resized = image.resize((224, 224))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def main():
    st.set_page_config(page_title="Lung Disease Diagnosis", layout="wide")
    
    with st.sidebar:
        st.header("About")
        st.write("This application uses a ResNet50 feature extractor combined with PCA and a KNN classifier to detect Pneumonia from Chest X-ray images.")
        st.write("Please upload a clear, front-facing Chest X-ray.")

    st.title("Lung Disease Classification")
    st.markdown("---")

    try:
        resnet, pca, knn = load_models()
    except Exception as e:
        st.error(f"Model loading failed. Error: {e}")
        return

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Input Image")
        uploaded_file = st.file_uploader("Upload X-ray image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

    with col2:
        st.subheader("Analysis Result")
        if uploaded_file is not None:
            if st.button("Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Processing image..."):
                    try:
                        img_tensor = process_image(image)
                        features_resnet = resnet.predict(img_tensor)
                        features_pca = pca.transform(features_resnet)
                        prediction = knn.predict(features_pca)
                        
                        st.markdown("### Diagnosis:")
                        if prediction[0] == 1:
                            st.error("PNEUMONIA DETECTED")
                            st.info("Recommendation: Please consult a healthcare professional immediately.")
                        else:
                            st.success("NORMAL (NO PNEUMONIA)")
                            st.info("Recommendation: No abnormalities detected based on this scan.")
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
        else:
            st.info("Please upload an image to begin analysis.")

if __name__ == '__main__':
    main()