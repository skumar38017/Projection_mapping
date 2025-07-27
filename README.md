# 🎯 Random Object Verification System

A robust real-time object verification system that uses multiple computer vision techniques to match objects from live camera feeds against a database of reference images.

## ✨ Features

### Backend Capabilities
- **Multiple Detection Methods**: ORB feature matching, Deep learning (EfficientNet), and Object detection (Faster R-CNN)
- **Real-time Processing**: Live camera streaming with real-time object verification
- **Multiple Camera Support**: Automatic detection and selection of available cameras
- **Robust Matching**: Combines traditional computer vision with deep learning for accurate results
- **Pinecone Integration**: Optional vector database support for scalable image matching
- **Performance Monitoring**: Real-time FPS counter and processing time metrics

### Frontend Features
- **4-Panel Layout**: 
  - Top-left: Reference images gallery
  - Top-right: Live camera feed
  - Bottom-left: Detection details and match history
  - Bottom-right: Processed video with detection results
- **Camera Controls**: Select camera, start/stop streaming, refresh camera list
- **Real-time Updates**: Live streaming with WebSocket communication
- **Responsive Design**: Works on desktop and mobile devices
- **Visual Feedback**: Color-coded match results and status indicators

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd /home/tagglabs/Videos/Projection_mapping/backend
pip install -r requirements.txt
```

### 2. Add Reference Images
Place your reference images (`.jpg`, `.jpeg`, `.png`) in the `app/assets/` directory:
```bash
cp your_images/* app/assets/
```

### 3. Configure Environment (Optional)
Edit `.env` file to customize settings:
```bash
# Detection thresholds
FEATURE_MATCH_THRESHOLD=0.3
DEEP_MATCH_THRESHOLD=0.7
DETECTION_CONFIDENCE=0.5

# Camera settings
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
FPS=30
```

### 4. Start the Server
```bash
python start_server.py
```

### 5. Open Your Browser
Navigate to: `http://localhost:8000`

## 🎮 How to Use

1. **Select Camera**: Choose from available cameras in the dropdown
2. **Start Streaming**: Click the "▶️ Start" button to begin live detection
3. **View Results**: 
   - See live camera feed in top-right panel
   - Watch detection results in bottom-right panel
   - Monitor match details in bottom-left panel
   - Reference images are shown in top-left panel
4. **Stop Streaming**: Click "⏹️ Stop" when done

## 🔧 System Architecture

### Detection Pipeline
1. **Frame Capture**: Real-time frames from selected camera
2. **Feature Extraction**: Multiple methods (ORB, EfficientNet, Faster R-CNN)
3. **Matching**: Compare against pre-loaded reference features
4. **Scoring**: Confidence-based matching with configurable thresholds
5. **Results**: Real-time display with visual feedback

### Technology Stack
- **Backend**: FastAPI, WebSockets, OpenCV, TensorFlow, PyTorch
- **Frontend**: Vanilla JavaScript, WebSocket API, Responsive CSS
- **Computer Vision**: ORB features, EfficientNet embeddings, Faster R-CNN
- **Optional**: Pinecone vector database for scalable matching

## 📊 Performance Features

- **Real-time FPS Counter**: Monitor streaming performance
- **Processing Time Metrics**: Track detection latency
- **Multiple Detection Methods**: Fallback options for robust matching
- **Configurable Thresholds**: Tune sensitivity for your use case
- **Debug Mode**: Save frames and intermediate results for analysis

## 🛠️ Configuration Options

### Detection Thresholds
- `FEATURE_MATCH_THRESHOLD`: ORB feature matching sensitivity (0.0-1.0)
- `DEEP_MATCH_THRESHOLD`: Deep learning matching sensitivity (0.0-1.0)
- `DETECTION_CONFIDENCE`: Object detection confidence threshold (0.0-1.0)

### Camera Settings
- `CAMERA_WIDTH`: Camera resolution width
- `CAMERA_HEIGHT`: Camera resolution height
- `FPS`: Target frames per second

### Advanced Features
- **Pinecone Integration**: Set `PINECONE_API_KEY` for vector database support
- **Debug Mode**: Frames and processing steps saved to `app/debug/`
- **Logging**: Configurable log levels and file output

## 🔍 Troubleshooting

### Common Issues

1. **No cameras detected**:
   - Check camera permissions
   - Ensure cameras are not in use by other applications
   - Try refreshing the camera list

2. **Poor detection accuracy**:
   - Adjust detection thresholds in `.env`
   - Ensure good lighting conditions
   - Add more reference images from different angles

3. **Slow performance**:
   - Reduce camera resolution
   - Lower FPS settings
   - Check system resources (CPU/GPU usage)

4. **WebSocket connection issues**:
   - Check firewall settings
   - Ensure port 8000 is available
   - Try refreshing the browser

### Debug Information
- Check `logs/server.log` for backend issues
- Use browser developer tools for frontend debugging
- Debug frames are saved to `app/debug/` directory

## 📁 Project Structure

```
backend/
├── app/
│   ├── assets/           # Reference images
│   ├── debug/           # Debug output
│   ├── static/          # Frontend files
│   ├── config.py        # Configuration
│   ├── main.py          # FastAPI application
│   ├── object_detector.py # Detection logic
│   ├── webrtc_signaling.py # WebSocket handling
│   └── utils.py         # Utility functions
├── logs/                # Log files
├── .env                 # Environment configuration
├── requirements.txt     # Python dependencies
└── start_server.py      # Server startup script
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files in `logs/`
3. Open an issue on GitHub
4. Contact the development team

---

**Happy Object Verification! 🎯**
