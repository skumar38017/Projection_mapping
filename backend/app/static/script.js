// script.js - Enhanced Object Verification Frontend
class ObjectVerificationApp {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.isStreaming = false;
        this.cameras = [];
        this.referenceImages = [];
        this.matchHistory = [];
        this.fpsCounter = 0;
        this.lastFrameTime = 0;
        
        this.initializeElements();
        this.setupEventListeners();
        this.connectWebSocket();
    }
    
    initializeElements() {
        // Control elements
        this.cameraSelect = document.getElementById('cameraSelect');
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.refreshBtn = document.getElementById('refreshBtn');
        this.status = document.getElementById('status');
        
        // Display elements
        this.referenceImagesContainer = document.getElementById('referenceImages');
        this.originalVideo = document.getElementById('originalVideo');
        this.originalVideoPlaceholder = document.getElementById('originalVideoPlaceholder');
        this.processedVideo = document.getElementById('processedVideo');
        this.processedVideoPlaceholder = document.getElementById('processedVideoPlaceholder');
        this.matchDetails = document.getElementById('matchDetails');
        this.fpsCounterElement = document.getElementById('fpsCounter');
    }
    
    setupEventListeners() {
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.stopBtn.addEventListener('click', () => this.stopCamera());
        this.refreshBtn.addEventListener('click', () => this.refreshCameras());
        
        // Handle window close
        window.addEventListener('beforeunload', () => {
            if (this.ws) {
                this.ws.close();
            }
        });
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        console.log('Attempting to connect to WebSocket:', wsUrl);
        
        try {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected successfully');
                this.updateStatus('connected', '🟢 Connected');
                this.isConnected = true;
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    console.log('Received WebSocket message:', data.type);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };
            
            this.ws.onclose = (event) => {
                console.log('WebSocket disconnected. Code:', event.code, 'Reason:', event.reason);
                this.updateStatus('disconnected', '🔴 Disconnected');
                this.isConnected = false;
                this.isStreaming = false;
                this.updateButtons();
                
                // Attempt to reconnect after 3 seconds
                setTimeout(() => {
                    if (!this.isConnected) {
                        console.log('Attempting to reconnect...');
                        this.connectWebSocket();
                    }
                }, 3000);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateStatus('disconnected', '🔴 Connection Error');
            };
            
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.updateStatus('disconnected', '🔴 Connection Failed');
        }
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'init':
                this.handleInit(data);
                break;
            case 'cameras_list':
                this.handleCamerasList(data.cameras);
                break;
            case 'reference_images':
                this.handleReferenceImages(data.images);
                break;
            case 'camera_started':
                this.handleCameraStarted(data);
                break;
            case 'camera_stopped':
                this.handleCameraStopped(data);
                break;
            case 'stream_frame':
                this.handleStreamFrame(data);
                break;
            case 'single_frame_result':
                this.handleSingleFrameResult(data);
                break;
            case 'match_history':
                this.handleMatchHistory(data.history);
                break;
            case 'error':
                this.handleError(data.message);
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }
    
    handleInit(data) {
        console.log('Received init data:', data);
        console.log('Cameras received:', data.cameras);
        console.log('Reference images received:', data.reference_images);
        
        this.handleCamerasList(data.cameras);
        this.handleReferenceImages(data.reference_images);
        
        // Display settings info
        console.log('Detection settings:', data.settings);
    }
    
    handleCamerasList(cameras) {
        this.cameras = cameras;
        this.updateCameraSelect();
    }
    
    handleReferenceImages(images) {
        this.referenceImages = images;
        this.displayReferenceImages();
    }
    
    handleCameraStarted(data) {
        if (data.success) {
            this.isStreaming = true;
            this.updateStatus('streaming', '🔴 Streaming');
            this.updateButtons();
            console.log(`Camera ${data.camera_id} started successfully`);
        } else {
            this.showError(`Failed to start camera: ${data.error}`);
            this.updateButtons();
        }
    }
    
    handleCameraStopped(data) {
        if (data.success) {
            this.isStreaming = false;
            this.updateStatus('connected', '🟢 Connected');
            this.updateButtons();
            this.hideVideoStreams();
            console.log('Camera stopped successfully');
        }
    }
    
    handleStreamFrame(data) {
        // Update original video
        this.originalVideo.src = `data:image/jpeg;base64,${data.original_frame}`;
        this.originalVideo.style.display = 'block';
        this.originalVideoPlaceholder.style.display = 'none';
        
        // Update processed video
        this.processedVideo.src = `data:image/jpeg;base64,${data.processed_frame}`;
        this.processedVideo.style.display = 'block';
        this.processedVideoPlaceholder.style.display = 'none';
        
        // Update match details
        this.updateMatchDetails(data.result);
        
        // Update FPS counter
        this.updateFPS();
    }
    
    handleSingleFrameResult(data) {
        this.processedVideo.src = `data:image/jpeg;base64,${data.processed_frame}`;
        this.processedVideo.style.display = 'block';
        this.processedVideoPlaceholder.style.display = 'none';
        
        this.updateMatchDetails(data.result);
    }
    
    handleMatchHistory(history) {
        this.matchHistory = history;
        // Could display match history in UI if needed
    }
    
    handleError(message) {
        console.error('Server error:', message);
        this.showError(message);
    }
    
    updateCameraSelect() {
        console.log('Updating camera select with cameras:', this.cameras);
        
        // Clear existing options except the first one
        this.cameraSelect.innerHTML = '<option value="">Select Camera...</option>';
        
        this.cameras.forEach(camera => {
            const option = document.createElement('option');
            option.value = camera.id;
            option.textContent = `${camera.name} (${camera.width}x${camera.height})`;
            this.cameraSelect.appendChild(option);
            console.log(`Added camera option: ${camera.name}`);
        });
        
        console.log(`Found ${this.cameras.length} cameras`);
        this.updateButtons();
    }
    
    displayReferenceImages() {
        console.log('Displaying reference images:', this.referenceImages);
        this.referenceImagesContainer.innerHTML = '';
        
        if (this.referenceImages.length === 0) {
            console.log('No reference images found');
            this.referenceImagesContainer.innerHTML = `
                <div style="grid-column: 1/-1; text-align: center; color: #7f8c8d;">
                    No reference images found.<br>
                    Add images to the assets folder.
                </div>
            `;
            return;
        }
        
        this.referenceImages.forEach(imageInfo => {
            console.log(`Processing reference image: ${imageInfo.filename}`);
            const imageDiv = document.createElement('div');
            imageDiv.className = 'reference-image';
            
            imageDiv.innerHTML = `
                <img src="data:image/jpeg;base64,${imageInfo.image_data}" 
                     alt="${imageInfo.filename}"
                     title="${imageInfo.filename} (${imageInfo.width}x${imageInfo.height})">
                <div class="filename">${imageInfo.filename}</div>
            `;
            
            this.referenceImagesContainer.appendChild(imageDiv);
        });
        
        console.log(`Displayed ${this.referenceImages.length} reference images`);
    }
    
    updateMatchDetails(result) {
        const matchInfo = document.createElement('div');
        matchInfo.className = `match-info ${result.success ? 'success' : 'no-match'}`;
        
        const timestamp = new Date(result.timestamp).toLocaleTimeString();
        
        if (result.success) {
            matchInfo.innerHTML = `
                <h4>✅ Match Found!</h4>
                <p><strong>Image:</strong> ${result.match_path}</p>
                <p><strong>Score:</strong> ${(result.score * 100).toFixed(1)}%</p>
                <p><strong>Method:</strong> ${result.method || 'Unknown'}</p>
                <p><strong>Time:</strong> ${timestamp}</p>
                ${result.processing_time ? `<p><strong>Processing:</strong> ${(result.processing_time * 1000).toFixed(1)}ms</p>` : ''}
            `;
        } else {
            matchInfo.innerHTML = `
                <h4>❌ No Match</h4>
                <p><strong>Best Score:</strong> ${(result.score * 100).toFixed(1)}%</p>
                <p><strong>Time:</strong> ${timestamp}</p>
                ${result.processing_time ? `<p><strong>Processing:</strong> ${(result.processing_time * 1000).toFixed(1)}ms</p>` : ''}
            `;
        }
        
        // Add to top of match details
        this.matchDetails.insertBefore(matchInfo, this.matchDetails.firstChild);
        
        // Keep only last 10 results
        while (this.matchDetails.children.length > 10) {
            this.matchDetails.removeChild(this.matchDetails.lastChild);
        }
    }
    
    updateFPS() {
        const now = performance.now();
        if (this.lastFrameTime > 0) {
            const fps = 1000 / (now - this.lastFrameTime);
            this.fpsCounter = Math.round(fps * 10) / 10;
            this.fpsCounterElement.textContent = `FPS: ${this.fpsCounter}`;
            this.fpsCounterElement.style.display = 'block';
        }
        this.lastFrameTime = now;
    }
    
    updateStatus(type, text) {
        this.status.className = `status ${type}`;
        this.status.textContent = text;
    }
    
    updateButtons() {
        this.startBtn.disabled = !this.isConnected || this.isStreaming || !this.cameraSelect.value;
        this.stopBtn.disabled = !this.isConnected || !this.isStreaming;
        this.refreshBtn.disabled = !this.isConnected;
        this.cameraSelect.disabled = this.isStreaming;
    }
    
    hideVideoStreams() {
        this.originalVideo.style.display = 'none';
        this.originalVideoPlaceholder.style.display = 'block';
        this.processedVideo.style.display = 'none';
        this.processedVideoPlaceholder.style.display = 'block';
        this.fpsCounterElement.style.display = 'none';
    }
    
    startCamera() {
        const cameraId = parseInt(this.cameraSelect.value);
        if (isNaN(cameraId)) {
            this.showError('Please select a camera first');
            return;
        }
        
        if (!this.isConnected) {
            this.showError('Not connected to server');
            return;
        }
        
        console.log(`Starting camera ${cameraId}`);
        this.sendMessage({
            type: 'start_camera',
            camera_id: cameraId
        });
        
        this.updateButtons();
    }
    
    stopCamera() {
        if (!this.isConnected) {
            this.showError('Not connected to server');
            return;
        }
        
        console.log('Stopping camera');
        this.sendMessage({
            type: 'stop_camera'
        });
        
        this.updateButtons();
    }
    
    refreshCameras() {
        if (!this.isConnected) {
            this.showError('Not connected to server');
            return;
        }
        
        console.log('Refreshing cameras');
        this.sendMessage({
            type: 'get_cameras'
        });
    }
    
    sendMessage(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        } else {
            console.error('WebSocket is not connected');
            this.showError('Connection lost. Please refresh the page.');
        }
    }
    
    showError(message) {
        console.error('Error:', message);
        
        // Create error element
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        
        // Add to match details
        this.matchDetails.insertBefore(errorDiv, this.matchDetails.firstChild);
        
        // Remove after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 5000);
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing Object Verification App');
    window.app = new ObjectVerificationApp();
});

// Handle camera select changes
document.addEventListener('DOMContentLoaded', () => {
    const cameraSelect = document.getElementById('cameraSelect');
    cameraSelect.addEventListener('change', () => {
        if (window.app) {
            window.app.updateButtons();
        }
    });
});
