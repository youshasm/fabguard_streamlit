import streamlit as st
import torch
import numpy as np
from PIL import Image
import tempfile
import os
import io
import time

# Set page configuration
st.set_page_config(
    page_title="FabGuard - Defect Detection System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model loading to improve performance
@st.cache_resource
def load_model(model_path):
    """Load the trained YOLO model"""
    try:
        # Import YOLO lazily to catch system-level import errors
        try:
            from ultralytics import YOLO
        except Exception as e:
            st.error(
                "⚠️ System libraries required by OpenCV/Ultralytics are missing. "
                "On Linux, ensure `libgl1-mesa-glx` and `libglib2.0-0` are installed."
            )
            st.exception(e)
            return None

        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            return None
            
        model = YOLO(model_path)
        st.success(f"✅ Model loaded successfully: {model_path}")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def process_image(image, model, confidence_threshold=0.5, model_type="binary"):
    """Process uploaded image with YOLO model"""
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
        st.error(f"❌ Error processing image: {str(e)}")
        return None, None

def process_video(video_path, model, confidence_threshold=0.5, progress_bar=None, status_text=None, model_type="binary"):
    """Process uploaded video with YOLO model"""
    try:
        # Lazy import cv2
        try:
            import cv2
        except Exception as e:
            st.error(
                "⚠️ OpenCV (cv2) failed to import. System libraries may be missing. "
                "On Streamlit Cloud, ensure packages.txt includes required libraries."
            )
            st.exception(e)
            return None, None

        # Open video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video writer
        output_path = tempfile.mktemp(suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Statistics tracking
        total_detections = 0
        defect_frames = 0
        frame_stats = []
        class_counts = {}
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            if progress_bar:
                progress_bar.progress(frame_count / total_frames)
            if status_text:
                status_text.text(f"Processing frame {frame_count + 1}/{total_frames}")
            
            # Run inference on frame
            results = model(frame, conf=confidence_threshold)
            
            # Get annotated frame
            annotated_frame = results[0].plot()
            
            # Write frame to output video
            out.write(annotated_frame)
            
            # Collect statistics
            frame_detections = len(results[0].boxes)
            total_detections += frame_detections
            
            if frame_detections > 0:
                defect_frames += 1
                
                # Count detections by class for this frame
                frame_defects = 0
                for detection in results[0].boxes:
                    class_id = int(detection.cls)
                    class_name = model.names[class_id]
                    
                    # Count by class
                    if class_name not in class_counts:
                        class_counts[class_name] = 0
                    class_counts[class_name] += 1
                    
                    # For binary models, count defects
                    if model_type == "binary" and class_name.lower() in ['defect', 'defective']:
                        frame_defects += 1
                
                frame_stats.append({
                    'frame': frame_count,
                    'detections': frame_detections,
                    'defects': frame_defects,
                    'timestamp': frame_count / fps
                })
            
            frame_count += 1
        
        # Clean up
        cap.release()
        out.release()
        
        stats = {
            'total_frames': total_frames,
            'frames_with_defects': defect_frames,
            'total_detections': total_detections,
            'frame_stats': frame_stats,
            'class_counts': class_counts,
            'fps': fps,
            'duration': total_frames / fps
        }
        
        return output_path, stats
    
    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
        return None, None

def display_detection_results(results, model, annotated_image, uploaded_file, model_type="binary"):
    """Display detection results and statistics"""
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
            if model_type == "binary":
                defect_count = len(class_counts.get('defect', []))
                st.metric("Defects Found", defect_count)
            else:
                # For multiclass, show most common class
                if class_counts:
                    most_common_class = max(class_counts.keys(), key=lambda x: len(class_counts[x]))
                    st.metric("Most Common Class", most_common_class)
        
        with col5:
            avg_confidence = np.mean([conf for confs in class_counts.values() for conf in confs])
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        # Class distribution chart for multiclass
        if model_type == "multiclass" and len(class_counts) > 1:
            st.subheader("📊 Class Distribution")
            class_data = []
            for class_name, confidences in class_counts.items():
                class_data.append({
                    "Class": class_name.title(),
                    "Count": len(confidences),
                    "Avg Confidence": f"{np.mean(confidences):.3f}"
                })
            st.dataframe(class_data, use_container_width=True)
        
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
        
        # Download button for annotated image
        st.subheader("💾 Download Results")
        
        # Convert annotated image to bytes for download
        img_buffer = io.BytesIO()
        annotated_image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        st.download_button(
            label="📥 Download Annotated Image",
            data=img_buffer.getvalue(),
            file_name=f"detection_{model_type}_{uploaded_file.name}",
            mime="image/jpeg"
        )
        
    else:
        st.info("No detections found with the current confidence threshold. Try lowering the threshold.")

def image_inference_tab(model_type):
    """Image inference interface"""
    st.subheader(f"🖼️ {model_type.title()} Image Detection")
    
    # Model selection
    if model_type == "binary":
        model_path = "models/yolov11n_binary.pt"
        st.info("🎯 **Binary Classification**: Detects Defect vs No Defect")
    else:
        model_options = {
            "YOLOv11n Multi": "models/yolov11n_multi.pt",
            "YOLOv8n Multi": "models/yolov8n_multi.pt",
            "YOLOv5n Multi": "models/yolov5n_multi.pt"
        }
        selected_model = st.selectbox("Select Model:", list(model_options.keys()))
        model_path = model_options[selected_model]
        st.info("🏷️ **Multiclass Classification**: Detects multiple defect types")
    
    # Load model
    model = load_model(model_path)
    if model is None:
        st.stop()
    
    # Display model classes
    with st.expander("📋 Model Classes"):
        classes = list(model.names.values())
        st.write(f"**Classes ({len(classes)}):** {', '.join(classes)}")
    
    # Sidebar parameters
    confidence_threshold = st.sidebar.slider(
        f"{model_type.title()} Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image file for defect detection",
        key=f"image_{model_type}"
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
            st.write(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Mode:** {image.mode}")
        
        # Process image when button is clicked
        if st.button(f"🔍 Detect {model_type.title()}", type="primary", key=f"detect_{model_type}"):
            with st.spinner(f"Processing image with {model_type} model..."):
                # Process the image
                annotated_image, results = process_image(image, model, confidence_threshold, model_type)
                
                if annotated_image is not None and results is not None:
                    with col2:
                        st.subheader("Detection Results")
                        st.image(annotated_image, caption="Detected Objects", use_container_width=True)
                    
                    # Display detection statistics
                    st.subheader("📊 Detection Summary")
                    display_detection_results(results, model, annotated_image, uploaded_file, model_type)
                        
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload an image file to begin defect detection")

def video_inference_tab(model_type):
    """Video inference interface"""
    st.subheader(f"🎥 {model_type.title()} Video Detection")
    
    # Model selection
    if model_type == "binary":
        model_path = "models/yolov11n_binary.pt"
        st.info("🎯 **Binary Classification**: Detects Defect vs No Defect frame by frame")
    else:
        model_options = {
            "YOLOv11n Multi": "models/yolov11n_multi.pt",
            "YOLOv8n Multi": "models/yolov8n_multi.pt"
        }
        selected_model = st.selectbox("Select Model:", list(model_options.keys()), key=f"video_model_{model_type}")
        model_path = model_options[selected_model]
        st.info("🏷️ **Multiclass Classification**: Detects multiple defect types frame by frame")
    
    # Load model
    model = load_model(model_path)
    if model is None:
        st.stop()
    
    # Display model classes
    with st.expander("📋 Model Classes"):
        classes = list(model.names.values())
        st.write(f"**Classes ({len(classes)}):** {', '.join(classes)}")
    
    # Sidebar parameters
    confidence_threshold = st.sidebar.slider(
        f"{model_type.title()} Video Confidence", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Minimum confidence score for detections",
        key=f"video_conf_{model_type}"
    )
    
    # Warning about processing time
    st.warning("⚠️ Video processing is computationally intensive and may take several minutes depending on video length and resolution.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a video file...", 
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
        help="Upload a video file for defect detection",
        key=f"video_{model_type}"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name
        
        # Display video info
        try:
            # Lazy import cv2 for video info
            try:
                import cv2
            except Exception as e:
                st.error("OpenCV not available for video processing")
                return
                
            cap = cv2.VideoCapture(temp_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps
            cap.release()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Video")
                st.video(uploaded_file)
                
                # Display video info
                st.write(f"**Resolution:** {width} x {height}")
                st.write(f"**FPS:** {fps}")
                st.write(f"**Duration:** {duration:.2f} seconds")
                st.write(f"**Total Frames:** {total_frames}")
            
            # Process video when button is clicked
            if st.button(f"🎬 Process {model_type.title()} Video", type="primary", key=f"process_video_{model_type}"):
                with st.spinner(f"Processing video with {model_type} model..."):
                    # Create progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    start_time = time.time()
                    
                    # Process the video
                    output_path, stats = process_video(
                        temp_video_path, 
                        model, 
                        confidence_threshold,
                        progress_bar,
                        status_text,
                        model_type
                    )
                    
                    processing_time = time.time() - start_time
                    
                    if output_path and stats:
                        with col2:
                            st.subheader("Processed Video")
                            
                            # Display processed video
                            with open(output_path, 'rb') as video_file:
                                video_bytes = video_file.read()
                                st.video(video_bytes)
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Display processing statistics
                        st.subheader("📊 Video Analysis Results")
                        
                        # Summary metrics
                        col3, col4, col5, col6 = st.columns(4)
                        
                        with col3:
                            st.metric("Total Frames", stats['total_frames'])
                        
                        with col4:
                            st.metric("Frames with Detections", stats['frames_with_defects'])
                        
                        with col5:
                            detection_percentage = (stats['frames_with_defects'] / stats['total_frames']) * 100
                            st.metric("Detection Frame %", f"{detection_percentage:.1f}%")
                        
                        with col6:
                            st.metric("Total Detections", stats['total_detections'])
                        
                        # Processing time info
                        st.success(f"⏱️ Processing completed in {processing_time:.1f} seconds")
                        
                        # Class distribution for multiclass
                        if model_type == "multiclass" and stats['class_counts']:
                            st.subheader("📊 Class Distribution in Video")
                            class_video_data = []
                            for class_name, count in stats['class_counts'].items():
                                class_video_data.append({
                                    "Class": class_name.title(),
                                    "Total Detections": count,
                                    "Percentage": f"{(count / stats['total_detections'] * 100):.1f}%"
                                })
                            st.dataframe(class_video_data, use_container_width=True)
                        
                        # Detailed frame statistics
                        if stats['frame_stats']:
                            st.subheader("🔍 Detailed Frame Analysis")
                            
                            # Show first 50 frames with detections
                            frame_data = []
                            for frame_stat in stats['frame_stats'][:50]:
                                frame_data.append({
                                    "Frame #": frame_stat['frame'],
                                    "Timestamp (s)": f"{frame_stat['timestamp']:.2f}",
                                    "Total Detections": frame_stat['detections'],
                                    "Defects": frame_stat['defects']
                                })
                            
                            if frame_data:
                                st.dataframe(frame_data, use_container_width=True)
                                
                                if len(stats['frame_stats']) > 50:
                                    st.info(f"Showing first 50 frames with detections. Total frames with detections: {len(stats['frame_stats'])}")
                            
                            # Download processed video
                            st.subheader("💾 Download Results")
                            
                            st.download_button(
                                label="📥 Download Processed Video",
                                data=video_bytes,
                                file_name=f"processed_{model_type}_{uploaded_file.name}",
                                mime="video/mp4"
                            )
                        
                        else:
                            st.info("No detections found in any frame. Try lowering the confidence threshold.")
                        
                        # Clean up temporary files
                        try:
                            os.unlink(output_path)
                        except:
                            pass
                    
            # Clean up temporary input file
            try:
                os.unlink(temp_video_path)
            except:
                pass
                
        except Exception as e:
            st.error(f"Error reading video file: {str(e)}")
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload a video file to begin defect detection")

def main():
    # Main header
    st.title("🏭 FabGuard - Defect Detection System")
    st.markdown("### AI-Powered Quality Control with YOLOv8/v11 Models")
    
    # Create main tabs
    tab1, tab2 = st.tabs(["🎯 Binary Classification", "🏷️ Multiclass Classification"])
    
    with tab1:
        st.markdown("**Binary Classification** detects whether an object is defective or not (2 classes)")
        
        # Create sub-tabs for binary
        binary_tab1, binary_tab2 = st.tabs(["📷 Image Detection", "🎥 Video Detection"])
        
        with binary_tab1:
            image_inference_tab("binary")
        
        with binary_tab2:
            video_inference_tab("binary")
    
    with tab2:
        st.markdown("**Multiclass Classification** detects different types of defects (multiple classes)")
        
        # Create sub-tabs for multiclass
        multi_tab1, multi_tab2 = st.tabs(["📷 Image Detection", "🎥 Video Detection"])
        
        with multi_tab1:
            image_inference_tab("multiclass")
        
        with multi_tab2:
            video_inference_tab("multiclass")
    
    # Sidebar information
    st.sidebar.header("📊 System Information")
    st.sidebar.success("✅ FabGuard v2.0")
    st.sidebar.info("🤖 Powered by YOLOv8/v11")
    
    st.sidebar.header("📁 Available Models")
    st.sidebar.write("**Binary Models:**")
    st.sidebar.write("• YOLOv11n Binary")
    st.sidebar.write("**Multiclass Models:**")
    st.sidebar.write("• YOLOv11n Multi")
    st.sidebar.write("• YOLOv8n Multi")
    
    st.sidebar.header("ℹ️ Usage Tips")
    st.sidebar.markdown("""
    - **Binary**: Best for simple defect/no-defect classification
    - **Multiclass**: Use when you need to identify specific defect types
    - **Confidence**: Lower values = more detections, higher values = more precision
    - **Video Processing**: May take several minutes for long videos
    """)

if __name__ == "__main__":
    main()