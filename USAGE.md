# 🎯 Random Object Verification System - Usage Guide

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
cd /home/tagglabs/Videos/Projection_mapping
./run.sh
```

### Option 2: Manual Setup
```bash
cd /home/tagglabs/Videos/Projection_mapping/backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test the system
python test_system.py

# Start the server
python start_server.py
```

## 📱 Using the Web Interface

1. **Open your browser** and navigate to: `http://localhost:8000`

2. **Select a camera** from the dropdown menu (Camera 0 is usually your default webcam)

3. **Click "▶️ Start"** to begin live object detection

4. **View the 4-panel interface**:
   - **Top-left**: Reference images from your assets folder
   - **Top-right**: Live camera feed
   - **Bottom-left**: Detection details and match history
   - **Bottom-right**: Processed video with detection results

5. **Test object detection** by showing objects to the camera that match your reference images

6. **Click "⏹️ Stop"** when finished

## 🖼️ Adding Reference Images

1. **Copy your images** to the assets directory:
   ```bash
   cp your_images/* /home/tagglabs/Videos/Projection_mapping/backend/app/assets/
   ```

2. **Supported formats**: `.jpg`, `.jpeg`, `.png`

3. **Recommended**: Use clear, well-lit images from different angles

4. **Restart the server** to load new images

## 🎮 How It Works

### Detection Process
1. **Camera captures** live video frames
2. **Multiple algorithms** analyze each frame:
   - ORB feature matching (traditional computer vision)
   - EfficientNet deep learning (neural network features)
   - Faster R-CNN object detection (optional)
3. **Comparison** against pre-loaded reference images
4. **Scoring** based on similarity thresholds
5. **Real-time display** of results

### Visual Feedback
- **Green boxes**: Successful match found
- **Red boxes**: No match detected
- **Score percentage**: Confidence level
- **Processing time**: Performance metrics
- **FPS counter**: Real-time frame rate

## ⚙️ Configuration

Edit `/home/tagglabs/Videos/Projection_mapping/backend/.env`:

```bash
# Detection sensitivity (0.0 = very strict, 1.0 = very loose)
FEATURE_MATCH_THRESHOLD=0.3
DEEP_MATCH_THRESHOLD=0.7
DETECTION_CONFIDENCE=0.5

# Camera settings
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
FPS=30
```

## 🔧 Troubleshooting

### Common Issues

**"No cameras detected"**
- Check camera permissions
- Close other applications using the camera
- Try different camera indices
- Run: `ls /dev/video*` to see available cameras

**"Poor detection accuracy"**
- Ensure good lighting
- Use high-quality reference images
- Adjust thresholds in `.env` file
- Add more reference images from different angles

**"Slow performance"**
- Reduce camera resolution
- Lower FPS settings
- Close other resource-intensive applications
- Check CPU/GPU usage

**"Connection issues"**
- Check if port 8000 is available
- Try refreshing the browser
- Check firewall settings
- Look at logs in `backend/logs/server.log`

### Debug Information

**Server logs**: `backend/logs/server.log`
**Debug frames**: `backend/app/debug/` (saved during processing)
**Browser console**: F12 → Console tab for frontend issues

## 📊 Performance Tips

### For Better Accuracy
- Use multiple reference images per object
- Ensure consistent lighting
- Avoid blurry or low-quality images
- Test different camera angles

### For Better Performance
- Reduce camera resolution (e.g., 320x240)
- Lower FPS (e.g., 15 FPS)
- Use fewer reference images
- Close unnecessary applications

## 🎯 Example Use Cases

### Security/Access Control
- Verify authorized objects (ID cards, keys, etc.)
- Monitor restricted items
- Access control based on object recognition

### Quality Control
- Verify product components
- Check assembly completeness
- Detect defects or variations

### Inventory Management
- Track specific items
- Verify item presence
- Automated counting

### Educational/Research
- Computer vision demonstrations
- Algorithm comparison
- Real-time detection experiments

## 🔄 System Status

**✅ Working**: Basic object detection with ORB features
**✅ Working**: Multiple camera support
**✅ Working**: Real-time web interface
**✅ Working**: Reference image management
**⚠️ Optional**: Deep learning features (requires TensorFlow)
**⚠️ Optional**: Pinecone integration (requires API key)

## 📞 Support

**Check logs first**: `backend/logs/server.log`
**Run diagnostics**: `python backend/test_system.py`
**Debug mode**: Frames saved to `backend/app/debug/`

---

**Happy Object Verification! 🎯**

*System created for robust real-time object detection and verification*
