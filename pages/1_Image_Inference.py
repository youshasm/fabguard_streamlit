import streamlit as st
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
import io

# Set page configuration
st.set_page_config(
    page_title="Image Inference",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model loading to improve performance
@st.cache_resource
def load_model():
    """Load the trained YOLO model"""
    try:
        model = YOLO('model.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image, model, confidence_threshold=0.5):
    """
    Process the uploaded image with YOLO model
    
    Args:
        image: PIL Image object
        model: Loaded YOLO model
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        annotated_image: Image with bounding boxes
        results: Detection results
    """
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Run inference
        results = model(img_array, conf=confidence_threshold)
        
        # Get the annotated image
        annotated_image = results[0].plot()
        
        # Convert back to PIL Image for display
        annotated_pil = Image.fromarray(annotated_image)
        
        return annotated_pil, results[0]
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def main():
    st.title("🖼️ Image Defect Detection")
    st.markdown("Upload an image to detect defects using your trained YOLOv11 model")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Sidebar for parameters
    st.sidebar.header("Detection Parameters")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    # Display model info
    st.sidebar.header("Model Information")
    st.sidebar.info(f"Model: YOLOv11n (Custom Trained)")
    st.sidebar.info(f"Classes: Defect, No Defect")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image file for defect detection"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Display image info
            st.write(f"**Image Size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Image Mode:** {image.mode}")
        
        # Process image when button is clicked
        if st.button("🔍 Detect Defects", type="primary"):
            with st.spinner("Processing image..."):
                # Process the image
                annotated_image, results = process_image(image, model, confidence_threshold)
                
                if annotated_image is not None and results is not None:
                    with col2:
                        st.subheader("Detection Results")
                        st.image(annotated_image, caption="Detected Objects", use_container_width=True)
                    
                    # Display detection statistics
                    st.subheader("📊 Detection Summary")
                    
                    if len(results.boxes) > 0:
                        # Create detection summary
                        detections = results.boxes
                        
                        # Count detections by class
                        class_counts = {}
                        total_detections = len(detections)
                        
                        for detection in detections:
                            class_id = int(detection.cls)
                            class_name = model.names[class_id]
                            confidence = float(detection.conf)
                            
                            if class_name not in class_counts:
                                class_counts[class_name] = []
                            class_counts[class_name].append(confidence)
                        
                        # Display summary metrics
                        col3, col4, col5 = st.columns(3)
                        
                        with col3:
                            st.metric("Total Detections", total_detections)
                        
                        with col4:
                            defect_count = len(class_counts.get('defect', []))
                            st.metric("Defects Found", defect_count)
                        
                        with col5:
                            avg_confidence = np.mean([conf for confs in class_counts.values() for conf in confs])
                            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                        
                        # Detailed detection table
                        st.subheader("🔍 Detailed Detections")
                        
                        detection_data = []
                        for i, detection in enumerate(detections):
                            class_id = int(detection.cls)
                            class_name = model.names[class_id]
                            confidence = float(detection.conf)
                            bbox = detection.xyxy[0].cpu().numpy()
                            
                            detection_data.append({
                                "Detection #": i + 1,
                                "Class": class_name.title(),
                                "Confidence": f"{confidence:.3f}",
                                "Bbox (x1,y1,x2,y2)": f"({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})"
                            })
                        
                        st.dataframe(detection_data, use_container_width=True)
                        
                        # Add download button for annotated image
                        st.subheader("💾 Download Results")
                        
                        # Convert annotated image to bytes for download (safer approach)
                        img_buffer = io.BytesIO()
                        annotated_image.save(img_buffer, format='JPEG')
                        img_buffer.seek(0)
                        
                        st.download_button(
                            label="📥 Download Annotated Image",
                            data=img_buffer.getvalue(),
                            file_name=f"defect_detection_{uploaded_file.name}",
                            mime="image/jpeg"
                        )
                        
                    else:
                        st.info("No detections found with the current confidence threshold. Try lowering the threshold in the sidebar.")
                        
                        # Still show the processed image
                        with col2:
                            st.subheader("Detection Results")
                            st.image(image, caption="No Detections Found", use_container_width=True)
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload an image file to begin defect detection")
        
        # Add some example information
        st.subheader("ℹ️ How to Use")
        st.markdown("""
        1. **Upload an Image**: Use the file uploader above to select an image
        2. **Adjust Settings**: Use the sidebar to modify detection parameters
        3. **Run Detection**: Click the "Detect Defects" button to analyze the image
        4. **View Results**: See the annotated image and detection statistics
        5. **Download**: Save the annotated image with detections
        """)
        
        st.subheader("📋 Supported Formats")
        st.markdown("PNG, JPG, JPEG, BMP, TIFF")

if __name__ == "__main__":
    main()