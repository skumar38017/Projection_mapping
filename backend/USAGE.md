# 🎯 Real-Time 3D Object Detection & Matching System - Usage Guide

## 🚀 Quick Start

### 1. Start the System
```bash
# Option 1: Using the startup script (recommended)
python start_system.py

# Option 2: Using the run script
python run.py

# Option 3: Direct conda command
conda run -n .projection-mapping python start_system.py
```

### 2. Access the Web Interface
- **Main Interface**: http://localhost:8000
- **API Health Check**: http://localhost:8000/api/health
- **Network Info**: http://localhost:8000/api/network

## 🎯 How the 60% Threshold Works

### Matching Logic
```
if similarity_score >= 60%:
    result = "Matched"
    broadcast_to_network(match_found=True)
else:
    result = "Not Matched"  
    broadcast_to_network(match_found=False)
```

### Example Results
- **85% similarity** → `"Matched"` ✅
- **45% similarity** → `"Not Matched"` ❌
- **60% similarity** → `"Matched"` ✅ (exactly at threshold)

## 📷 Adding Reference Images

1. **Place images in assets folder:**
```bash
cp your_object.jpg app/assets/
```

2. **Supported formats:** `.jpg`, `.jpeg`, `.png`

3. **Restart system** to load new images

## 📡 Network Broadcasting

The system broadcasts results to multiple protocols simultaneously:

### OSC (Port 8001)
```python
# TouchDesigner, Max/MSP
python network_clients/osc_client_example.py
```

### TCP (Port 8002)
```python
# Persistent connections
python network_clients/tcp_client_example.py --host YOUR_IP
```

### UDP (Port 8003)
```python
# Broadcast messages
python network_clients/udp_client_example.py
```

### Socket.IO (Port 8000)
```javascript
// Web clients
const socket = io('http://YOUR_IP:8000');
socket.on('detection_result', (data) => {
    console.log('Match result:', data.match_result);
});
```

## 📊 Message Format

All network protocols receive this JSON structure:

```json
{
  "timestamp": "2025-01-27T14:00:00Z",
  "detection": {
    "object_detected": true,
    "match_found": true,
    "match_result": "Matched",
    "confidence": 85.3,
    "reference_image": "watch.jpeg",
    "similarity_score": 0.853
  },
  "performance": {
    "processing_time_ms": 15.2,
    "orb_features": 2000,
    "sift_features": 1500,
    "matching_method": "SIFT"
  },
  "system": {
    "threshold": 60.0,
    "threshold_met": true
  }
}
```

## 🧪 Testing the System

### Run the test script:
```bash
conda run -n .projection-mapping python test_matching.py
```

This will test:
1. **Perfect match** (same image) → Should be "Matched"
2. **Modified image** (brightness changed) → Should be "Matched" 
3. **Random noise** → Should be "Not Matched"

## 🔧 Configuration

### Adjust threshold (if needed):
Edit `app/config.py`:
```python
FEATURE_MATCH_THRESHOLD = 0.6  # 60% threshold
DEEP_MATCH_THRESHOLD = 0.6     # 60% threshold
```

### Performance settings:
```python
TARGET_FPS = 30                # Target frame rate
MAX_PROCESSING_TIME = 0.020    # 20ms max processing
```

## 📈 Performance Monitoring

### Real-time logs show:
- **Detection events** with confidence scores
- **Match results** ("Matched" or "Not Matched")
- **Processing times** (target: <20ms)
- **Resource usage** (CPU/GPU/RAM)
- **Network broadcasting** status

### Example terminal output:
```
🔍 [MATCHING] Processing frame at 14:00:00.123
✅ [FEATURES] Extracted ORB=2000, SIFT=1500 features
🎉 [MATCHED] ✅ MATCH FOUND!
   📸 Reference Image: watch.jpeg
   📊 Similarity Score: 75.3% (≥60% = Matched)
   ⏱️ Processing Time: 15.2ms
📡 [BROADCAST] Sent 'Matched' to OSC, TCP, UDP, Socket.IO
```

## 🛠️ Troubleshooting

### No camera detected:
- Check camera permissions
- Ensure camera is not used by another application

### Low performance:
- Check GPU status: `nvidia-smi`
- Monitor resource usage in terminal logs
- Reduce camera resolution if needed

### Network issues:
- Check firewall settings
- Verify IP addresses in client examples
- Test with `ping YOUR_IP`

### No reference images:
- Add images to `app/assets/` folder
- Restart the system
- Check logs for loading confirmation

## 🎬 Live Usage

1. **Start the system**
2. **Open web interface** to see live camera feed
3. **Hold objects** in front of camera
4. **Watch for match results** in real-time
5. **Monitor network clients** for broadcast messages

The system will continuously:
- Detect objects in camera feed
- Compare with stored reference images
- Apply 60% threshold decision
- Broadcast results to all network protocols
- Display results in web interface and terminal

---

**🚀 Ready for real-time 3D object detection and matching!**
