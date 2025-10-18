# Defect Detection Streamlit App

This multi-page Streamlit application uses a trained YOLOv11 model to detect defects in both images and videos.

## Features

- �️ **Image Inference**: Upload images for instant defect detection
- 🎥 **Video Inference**: Process videos frame by frame for comprehensive analysis
- 📊 Real-time inference with confidence scoring
- 🎛️ Adjustable confidence threshold
- 📈 Detection statistics and detailed results
- 💾 Download annotated images and videos
- 📱 Responsive multi-page web interface

## Pages

1. **Home**: Overview and navigation
2. **Image Inference**: Single image defect detection
3. **Video Inference**: Video processing with frame-by-frame analysis

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure your trained model file `model.pt` is in the same directory as `app.py`

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

### Image Inference
1. Navigate to "Image Inference" page from the sidebar
2. Upload an image using the file uploader
3. Adjust the confidence threshold if needed
4. Click "Detect Defects" to run inference
5. View results and download the annotated image

### Video Inference
1. Navigate to "Video Inference" page from the sidebar
2. Upload a video file using the file uploader
3. Adjust the confidence threshold if needed
4. Click "Process Video" to analyze frame by frame
5. View the processed video and analysis statistics
6. Download the annotated video

## Model Information

- **Model**: YOLOv11n (Custom Trained)
- **Classes**: Defect, No Defect
- **Input**: Images (PNG, JPG, JPEG, BMP, TIFF) and Videos (MP4, AVI, MOV, MKV, WMV)
- **Output**: Annotated images/videos with bounding boxes

## File Structure

```
├── app.py                           # Main Streamlit application (landing page)
├── pages/
│   ├── 1_🖼️_Image_Inference.py      # Image processing page
│   └── 2_🎥_Video_Inference.py      # Video processing page
├── model.pt                         # Trained YOLOv11 model
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## Important Notes

- **Video Processing**: Video processing is computationally intensive and may take several minutes
- **File Size**: Larger files will take longer to process
- **Model Caching**: The model is cached for better performance across pages
- **Local Processing**: All processing is done locally on your machine
- **Memory Usage**: Video processing may require significant RAM for large files

## Performance Tips

- Use smaller video files for faster processing
- Reduce video resolution if processing is too slow
- Close other applications to free up system resources during video processing
- Consider processing shorter video clips for initial testing