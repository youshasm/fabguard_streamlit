import streamlit as st
import torch
import numpy as np
from PIL import Image
import tempfile
import os
import io
import time

# Lazy import ultralytics/cv2 to provide clearer errors in hosted environments

# Set page configuration
st.set_page_config(
    page_title="Video Inference",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model loading to improve performance
@st.cache_resource
def load_model():
    """Load the trained YOLO model"""
    try:
        try:
            from ultralytics import YOLO
        except Exception as e:
            st.error(
                "System libraries required by OpenCV/Ultralytics are missing on the host. "
                "Please add 'libgl1-mesa-glx' and 'libglib2.0-0' to packages.txt for Streamlit Cloud or install them on your server."
            )
            st.exception(e)
            return None

        model = YOLO('model.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_video(video_path, model, confidence_threshold=0.5, progress_bar=None, status_text=None):
    """
    Process the uploaded video with YOLO model
    
    Args:
        video_path: Path to the video file
        model: Loaded YOLO model
        confidence_threshold: Minimum confidence for detections
        progress_bar: Streamlit progress bar
        status_text: Streamlit status text
    
    Returns:
        output_video_path: Path to the annotated video
        stats: Detection statistics
    """
    try:
        # Lazy import cv2 to allow clearer error messages if system libs are missing
        try:
            import cv2
        except Exception as e:
            st.error(
                "OpenCV (cv2) failed to import. The host may be missing system libraries like libGL. "
                "On Streamlit Cloud add 'libgl1-mesa-glx' and 'libglib2.0-0' to packages.txt or install them on your server."
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
                    if class_name.lower() == 'defect':
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
            'fps': fps,
            'duration': total_frames / fps
        }
        
        return output_path, stats
    
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None, None

def main():
    st.title("🎥 Video Defect Detection")
    st.markdown("Upload a video to detect defects frame by frame using your trained YOLOv11 model")
    
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
    
    # Warning about processing time
    st.sidebar.header("⚠️ Processing Info")
    st.sidebar.warning("Video processing can take several minutes depending on video length and resolution.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a video file...", 
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
        help="Upload a video file for defect detection"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name
        
        # Display video info
        try:
            try:
                import cv2
            except Exception as e:
                st.error(
                    "OpenCV (cv2) failed to import. The host may be missing system libraries like libGL. "
                    "On Streamlit Cloud add 'libgl1-mesa-glx' and 'libglib2.0-0' to packages.txt or install them on your server."
                )
                st.exception(e)
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
            if st.button("🎬 Process Video", type="primary"):
                with st.spinner("Processing video..."):
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
                        status_text
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
                            st.metric("Frames with Defects", stats['frames_with_defects'])
                        
                        with col5:
                            defect_percentage = (stats['frames_with_defects'] / stats['total_frames']) * 100
                            st.metric("Defect Frame %", f"{defect_percentage:.1f}%")
                        
                        with col6:
                            st.metric("Total Detections", stats['total_detections'])
                        
                        # Processing time info
                        st.info(f"⏱️ Processing completed in {processing_time:.1f} seconds")
                        
                        # Detailed frame statistics
                        if stats['frame_stats']:
                            st.subheader("🔍 Detailed Frame Analysis")
                            
                            # Create dataframe for frame stats
                            frame_data = []
                            for frame_stat in stats['frame_stats'][:50]:  # Show first 50 frames with detections
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
                                file_name=f"processed_{uploaded_file.name}",
                                mime="video/mp4"
                            )
                        
                        else:
                            st.info("No defects detected in any frame. Try lowering the confidence threshold.")
                        
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
        
        # Add some example information
        st.subheader("ℹ️ How to Use")
        st.markdown("""
        1. **Upload a Video**: Use the file uploader above to select a video
        2. **Adjust Settings**: Use the sidebar to modify detection parameters
        3. **Process Video**: Click the "Process Video" button to analyze frame by frame
        4. **View Results**: See the processed video with detections and analysis statistics
        5. **Download**: Save the processed video with annotations
        """)
        
        st.subheader("📋 Supported Formats")
        st.markdown("MP4, AVI, MOV, MKV, WMV")
        
        st.subheader("⚠️ Important Notes")
        st.markdown("""
        - Video processing is computationally intensive and may take time
        - Processing time depends on video length, resolution, and frame rate
        - Large videos may require significant processing time
        - The output video will have the same resolution and frame rate as the input
        """)

if __name__ == "__main__":
    main()