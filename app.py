import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Defect Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🔍 Defect Detection System")
    st.markdown("### AI-Powered Defect Detection using YOLOv11")
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Welcome to the **Defect Detection System**! This application uses a custom-trained YOLOv11 model 
        to detect defects in images and videos. Choose from the options in the sidebar to get started.
        
        ### 🚀 Features
        - **Real-time Image Analysis**: Upload images for instant defect detection
        - **Video Processing**: Analyze videos frame by frame for comprehensive defect detection
        - **Adjustable Parameters**: Fine-tune detection sensitivity with confidence thresholds
        - **Detailed Statistics**: Get comprehensive reports on detection results
        - **Download Results**: Save annotated images and videos
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Model Information
        - **Architecture**: YOLOv11n
        - **Training**: Custom Dataset
        - **Classes**: Defect, No Defect
        - **Framework**: Ultralytics
        """)
    
    # Navigation guide
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        #### 🖼️ Image Inference
        - Upload single images (PNG, JPG, JPEG, BMP, TIFF)
        - Get instant results with bounding boxes
        - View detailed detection statistics
        - Download annotated images
        
        👈 **Click "Image Inference" in the sidebar to start**
        """)
    
    with col4:
        st.markdown("""
        #### 🎥 Video Inference
        - Upload video files (MP4, AVI, MOV, MKV, WMV)
        - Frame-by-frame defect detection
        - Comprehensive video analysis
        - Download processed videos
        
        👈 **Click "Video Inference" in the sidebar to start**
        """)
    
    # Quick stats/info
    st.markdown("---")
    st.markdown("### 📈 System Capabilities")
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("Supported Image Formats", "5", help="PNG, JPG, JPEG, BMP, TIFF")
    
    with col6:
        st.metric("Supported Video Formats", "5", help="MP4, AVI, MOV, MKV, WMV")
    
    with col7:
        st.metric("Model Classes", "2", help="Defect, No Defect")
    
    with col8:
        st.metric("Framework", "YOLOv11", help="Ultralytics YOLOv11n")
    
    # Usage tips
    st.markdown("---")
    st.markdown("### 💡 Tips for Best Results")
    
    st.markdown("""
    1. **Image Quality**: Use high-quality, well-lit images for better detection accuracy
    2. **Video Processing**: Be patient with video processing - it can take several minutes
    3. **Confidence Threshold**: Adjust the confidence threshold to balance precision and recall
    4. **File Size**: Larger files will take longer to process
    5. **Format Support**: Stick to supported formats for optimal compatibility
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Defect Detection System | Powered by YOLOv11 & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()