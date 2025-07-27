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
        this.matchStatusIndicator = document.getElementById('matchStatusIndicator');
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
                    console.log('Received WebSocket message:', data.type, data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error, event.data);
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
            
            // Update placeholder text to show streaming has started
            this.originalVideoPlaceholder.textContent = 'Streaming...';
            this.processedVideoPlaceholder.textContent = 'Processing frames...';
            
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
        console.log('=== STREAM FRAME DEBUG ===');
        console.log('Full data received:', data);
        console.log('Result object:', data.result);
        console.log('Result keys:', Object.keys(data.result || {}));
        console.log('match_found:', data.result?.match_found);
        console.log('threshold_met:', data.result?.threshold_met);
        console.log('similarity_score:', data.result?.similarity_score);
        console.log('matched_image:', data.result?.matched_image);
        console.log('method:', data.result?.method);
        console.log('processing_time_ms:', data.result?.processing_time_ms);
        console.log('========================');
        
        // Update original video
        if (data.original_frame) {
            this.originalVideo.src = `data:image/jpeg;base64,${data.original_frame}`;
            this.originalVideo.style.display = 'block';
            this.originalVideoPlaceholder.style.display = 'none';
        }
        
        // Update processed video
        if (data.processed_frame) {
            this.processedVideo.src = `data:image/jpeg;base64,${data.processed_frame}`;
            this.processedVideo.style.display = 'block';
            this.processedVideoPlaceholder.style.display = 'none';
        }
        
        // Update match details
        if (data.result) {
            this.updateMatchDetails(data.result);
        }
        
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
            const objectId = this.generateObjectId(imageInfo.filename);
            
            const imageDiv = document.createElement('div');
            imageDiv.className = 'reference-image';
            
            imageDiv.innerHTML = `
                <img src="data:image/jpeg;base64,${imageInfo.image_data}" 
                     alt="${imageInfo.filename}"
                     title="${imageInfo.filename} (${imageInfo.width}x${imageInfo.height})">
                <div class="filename">
                    <div style="font-weight: bold; margin-bottom: 2px;">${imageInfo.filename}</div>
                    <div style="font-size: 0.6rem; color: #bbb;">🔗 ${objectId}</div>
                </div>
            `;
            
            this.referenceImagesContainer.appendChild(imageDiv);
        });
        
        console.log(`Displayed ${this.referenceImages.length} reference images with Object IDs`);
    }
    
    updateMatchDetails(result) {
        console.log('=== MATCH DETAILS DEBUG ===');
        console.log('UpdateMatchDetails called with:', result);
        console.log('Type of result:', typeof result);
        console.log('Result keys:', Object.keys(result || {}));
        
        // Handle different possible data structures
        let match_found = false;
        let similarity_score = 0;
        let matched_image = null;
        let method = 'Unknown';
        let processing_time_ms = 0;
        let timestamp = new Date().toISOString();
        let threshold_met = false;
        
        if (result) {
            // Try different possible field names
            match_found = result.match_found === true || result.matched === true || result.success === true;
            threshold_met = result.threshold_met === true;
            similarity_score = result.similarity_score || result.confidence || result.score || 0;
            matched_image = result.matched_image || result.match_path || result.image;
            method = result.method || 'Unknown';
            processing_time_ms = result.processing_time_ms || (result.processing_time * 1000) || 0;
            timestamp = result.timestamp || new Date().toISOString();
            
            console.log('Parsed values:');
            console.log('- match_found:', match_found);
            console.log('- threshold_met:', threshold_met);
            console.log('- similarity_score:', similarity_score);
            console.log('- matched_image:', matched_image);
            console.log('- method:', method);
            console.log('- processing_time_ms:', processing_time_ms);
        }
        
        const matchInfo = document.createElement('div');
        
        // Only show as match if both match_found AND threshold_met are true
        const isMatch = match_found && threshold_met;
        console.log('Final isMatch determination:', isMatch);
        console.log('========================');
        
        matchInfo.className = `match-info ${isMatch ? 'success' : 'no-match'}`;
        
        // Update status indicator
        this.updateMatchStatusIndicator(isMatch);
        
        const displayTime = new Date(timestamp).toLocaleTimeString();
        const scorePercent = (similarity_score * 100).toFixed(1);
        
        if (isMatch) {
            // Create enhanced match display with image
            const matchedImageSrc = this.getMatchedImageSrc(matched_image);
            const objectId = this.generateObjectId(matched_image);
            
            matchInfo.innerHTML = `
                <div style="display: flex; gap: 1rem; align-items: flex-start;">
                    <div style="flex-shrink: 0;">
                        ${matchedImageSrc ? `
                            <img src="${matchedImageSrc}" 
                                 alt="${matched_image}" 
                                 style="width: 80px; height: 80px; object-fit: cover; border-radius: 5px; border: 2px solid #27ae60;">
                        ` : `
                            <div style="width: 80px; height: 80px; background: #f0f0f0; border-radius: 5px; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; color: #666;">
                                📷 Image
                            </div>
                        `}
                    </div>
                    <div style="flex: 1;">
                        <h4 style="color: #27ae60; margin: 0 0 0.5rem 0;">✅ Match Found!</h4>
                        <div style="font-size: 0.9rem; line-height: 1.4;">
                            <p style="margin: 0.2rem 0;"><strong>📄 Reference Image:</strong> ${matched_image || 'Unknown'}</p>
                            <p style="margin: 0.2rem 0;"><strong>🕒 Detected At:</strong> ${displayTime}</p>
                            <p style="margin: 0.2rem 0;"><strong>✅ Match Confidence:</strong> ${scorePercent}%</p>
                            <p style="margin: 0.2rem 0;"><strong>🔗 Object ID:</strong> ${objectId}</p>
                            <p style="margin: 0.2rem 0;"><strong>🔍 Method:</strong> ${method}</p>
                            <p style="margin: 0.2rem 0;"><strong>⏱️ Processing:</strong> ${processing_time_ms.toFixed(1)}ms</p>
                            <p style="margin: 0.2rem 0;"><strong>📁 Reference Folder:</strong> assets/${matched_image || 'unknown'}</p>
                        </div>
                    </div>
                </div>
            `;
        } else {
            const bestCandidate = result?.details?.best_candidate || matched_image || 'None';
            matchInfo.innerHTML = `
                <div>
                    <h4 style="color: #e74c3c; margin: 0 0 0.5rem 0;">❌ No Match Found</h4>
                    <div style="font-size: 0.9rem; line-height: 1.4;">
                        <p style="margin: 0.2rem 0;"><strong>📊 Best Score:</strong> ${scorePercent}%</p>
                        <p style="margin: 0.2rem 0;"><strong>📸 Best Candidate:</strong> ${bestCandidate}</p>
                        <p style="margin: 0.2rem 0;"><strong>🎯 Threshold:</strong> 60.0% (Required for match)</p>
                        <p style="margin: 0.2rem 0;"><strong>🔍 Method:</strong> ${method}</p>
                        <p style="margin: 0.2rem 0;"><strong>🕒 Time:</strong> ${displayTime}</p>
                        <p style="margin: 0.2rem 0;"><strong>⏱️ Processing:</strong> ${processing_time_ms.toFixed(1)}ms</p>
                    </div>
                </div>
            `;
        }
        
        // Add to top of match details
        this.matchDetails.insertBefore(matchInfo, this.matchDetails.firstChild);
        
        // Keep only last 10 results
        while (this.matchDetails.children.length > 10) {
            this.matchDetails.removeChild(this.matchDetails.lastChild);
        }
    }
    
    updateMatchStatusIndicator(isMatch) {
        if (!this.matchStatusIndicator) return;
        
        this.matchStatusIndicator.className = 'match-status-indicator';
        
        if (isMatch) {
            this.matchStatusIndicator.classList.add('match-true');
            this.matchStatusIndicator.innerHTML = '<span class="status-text">✅ TRUE</span>';
        } else {
            this.matchStatusIndicator.classList.add('match-false');
            this.matchStatusIndicator.innerHTML = '<span class="status-text">❌ FALSE</span>';
        }
    }
    
    generateObjectId(imageName) {
        if (!imageName) return 'unknown_000';
        
        // Remove file extension and create a clean object ID
        const baseName = imageName.replace(/\.[^/.]+$/, "");
        const cleanName = baseName.toLowerCase().replace(/[^a-z0-9]/g, '_');
        
        // Generate a 3-digit number based on the image name for consistency
        let hash = 0;
        for (let i = 0; i < imageName.length; i++) {
            hash = ((hash << 5) - hash + imageName.charCodeAt(i)) & 0xffffffff;
        }
        const objectNumber = Math.abs(hash) % 1000;
        const paddedNumber = objectNumber.toString().padStart(3, '0');
        
        return `${cleanName}_${paddedNumber}`;
    }
    
    getMatchedImageSrc(imageName) {
        // Find the matched image from reference images
        const matchedRef = this.referenceImages.find(ref => ref.filename === imageName);
        return matchedRef ? `data:image/jpeg;base64,${matchedRef.image_data}` : null;
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
        this.originalVideoPlaceholder.textContent = 'Select a camera and click Start to begin streaming';
        
        this.processedVideo.style.display = 'none';
        this.processedVideoPlaceholder.style.display = 'block';
        this.processedVideoPlaceholder.textContent = 'Processed video will appear here';
        
        this.fpsCounterElement.style.display = 'none';
        
        // Reset status indicator
        if (this.matchStatusIndicator) {
            this.matchStatusIndicator.className = 'match-status-indicator waiting';
            this.matchStatusIndicator.innerHTML = '<span class="status-text">WAITING</span>';
        }
        
        // Reset match details
        this.matchDetails.innerHTML = `
            <div class="match-info">
                <h4>Waiting for detection...</h4>
                <p>Start the camera to begin object verification</p>
            </div>
        `;
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
