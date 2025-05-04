import React, { useRef, useState, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import './WebcamComponent.css';

// Interface for detected objects
interface DetectedObject {
  class: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

// Recyclable trash items
const RECYCLABLE_TRASH_ITEMS = [
  'bottle', 'cup', 'wine glass', 'plastic bag', 'paper', 'cardboard', 'newspaper',
  'book', 'aluminum foil', 'tin can', 'metal can', 'container', 'plastic'
];

// Wrappers are also considered recyclable
const WRAPPER_LIKE_ITEMS = [
  'book', 'tie', 'handbag', 'backpack', 'box', 'suitcase'
];

// Items that should not be classified as trash at all
const NOT_TRASH_ITEMS = [
  'person', 'human', 'man', 'woman', 'child', 'boy', 'girl', 
  'dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 
  'bear', 'zebra', 'giraffe'
];

// React component for webcam
const WebcamComponent: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [detectionStatus, setDetectionStatus] = useState<string>('Loading model...');
  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([]);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [recycleDetected, setRecycleDetected] = useState(false);
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const requestRef = useRef<number | null>(null);
  const [errorCount, setErrorCount] = useState(0);
  const detectionRunningRef = useRef(false);
  const [showDetailsPanel, setShowDetailsPanel] = useState(false);
  const lastDetectionTimeRef = useRef<number>(0);
  const detectionIntervalRef = useRef<number>(150); // Throttle to run ~6-7 times per second

  // Add performance monitoring
  const fpsCountRef = useRef<number>(0);
  const lastFpsUpdateRef = useRef<number>(0);
  const [fps, setFps] = useState<number>(0);
  const frameCountRef = useRef<number>(0); // New ref for actual frame counting

  // Load the TensorFlow model when component mounts
  useEffect(() => {
    let mounted = true;
    let loadAttempts = 0;
    const maxAttempts = 3;
    
    async function loadModel() {
      if (loadAttempts >= maxAttempts) {
        if (mounted) setDetectionStatus(`Failed to load model after ${maxAttempts} attempts. Please refresh the page.`);
        return;
      }
      
      loadAttempts++;
      try {
        if (mounted) setDetectionStatus(`Loading model (attempt ${loadAttempts})...`);
        
        // Reset any previous state
        if (modelRef.current) {
          modelRef.current = null;
        }
        
        // Clear all tensors and TF memory
        try {
          tf.disposeVariables();
          tf.engine().endScope();
          tf.engine().startScope();
        } catch (e) {
          console.warn('Failed to clear TF memory:', e);
        }
        
        // Simple configuration - minimal settings to avoid errors
        await tf.setBackend('webgl');
        await tf.ready();
        console.log('Using TensorFlow backend:', tf.getBackend());
        
        // Simple model loading - using the most basic approach
        const model = await cocoSsd.load({
          base: 'lite_mobilenet_v2'
        });
        
        if (!mounted) return;
        
        modelRef.current = model;
        console.log('Model loaded successfully');
        
        setModelLoaded(true);
        setDetectionStatus('Model loaded. Ready to start.');
      } catch (error) {
        console.error(`Failed to load model (attempt ${loadAttempts}):`, error);
        
        if (mounted) {
          setDetectionStatus(`Error loading model: ${error instanceof Error ? error.message : 'Unknown error'}. Retrying...`);
          
          // Wait before retrying
          setTimeout(() => {
            if (mounted) loadModel();
          }, 2000);
        }
      }
    }
    
    loadModel();
    
    // Cleanup
    return () => {
      mounted = false;
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, []);

  // Convert COCO-SSD detection results to our DetectedObject format
  const processDetections = (predictions: cocoSsd.DetectedObject[]): DetectedObject[] => {
    return predictions.map(prediction => {
      const [x, y, width, height] = prediction.bbox;
      return {
        class: prediction.class,
        confidence: Math.round(prediction.score * 100),
        x,
        y,
        width,
        height
      };
    });
  };

  // Check if an object is recyclable (including wrappers)
  const isRecyclable = (objectClass: string): boolean => {
    const lowerClass = objectClass.toLowerCase();
    return RECYCLABLE_TRASH_ITEMS.includes(lowerClass) || WRAPPER_LIKE_ITEMS.includes(lowerClass);
  };

  // Check if an object is not trash (like humans and animals)
  const isNotTrash = (objectClass: string): boolean => {
    const lowerClass = objectClass.toLowerCase();
    return NOT_TRASH_ITEMS.includes(lowerClass);
  };

  // New function for frame counting separate from detection
  const countFrames = useCallback(() => {
    frameCountRef.current++;
    const now = performance.now();
    
    // Update FPS once per second
    if (now - lastFpsUpdateRef.current >= 1000) {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
      lastFpsUpdateRef.current = now;
    }
    
    // Continue counting frames if streaming
    if (isStreaming) {
      requestAnimationFrame(countFrames);
    }
  }, [isStreaming]);

  // Start frame counting when streaming starts
  useEffect(() => {
    if (isStreaming) {
      frameCountRef.current = 0;
      lastFpsUpdateRef.current = performance.now();
      requestAnimationFrame(countFrames);
    }
    
    return () => {
      frameCountRef.current = 0;
    };
  }, [isStreaming, countFrames]);

  // Main detection function - simplified for reliability
  const detectObjects = useCallback(async () => {
    // Skip if detection is already running or prerequisites not met
    if (detectionRunningRef.current || !isStreaming || !videoRef.current || !modelRef.current) {
      if (isStreaming && modelLoaded) {
        requestRef.current = requestAnimationFrame(detectObjects);
      }
      return;
    }
    
    const now = performance.now();
    
    // Throttle detection frequency
    if (now - lastDetectionTimeRef.current < detectionIntervalRef.current) {
      requestRef.current = requestAnimationFrame(detectObjects);
      return;
    }
    
    lastDetectionTimeRef.current = now;
    detectionRunningRef.current = true;
    
    try {
      // Basic checks for video readiness
      if (!videoRef.current || !videoRef.current.videoWidth || !videoRef.current.videoHeight) {
        detectionRunningRef.current = false;
        requestRef.current = requestAnimationFrame(detectObjects);
        return;
      }
      
      // Simplified detection approach
      try {
        // Wrapped in try/catch for better error handling
        const predictions = await modelRef.current.detect(videoRef.current, 3, 0.6);
        
        // Process results if detection was successful
        const allDetectedItems = processDetections(predictions);
        
        // Separate objects into categories
        const notTrashItems = allDetectedItems.filter(obj => isNotTrash(obj.class));
        const trashItems = allDetectedItems.filter(obj => !isNotTrash(obj.class));
        const recyclableItems = trashItems.filter(obj => isRecyclable(obj.class));
        
        // Update state
        setDetectedObjects(trashItems);
        setRecycleDetected(recyclableItems.length > 0);
        setErrorCount(0);
        
        // Set appropriate status message
        if (notTrashItems.length > 0) {
          setDetectionStatus(`Detected: ${notTrashItems.map(item => item.class).join(', ')} (not trash)`);
        } else if (recyclableItems.length > 0) {
          setDetectionStatus(`Recyclable items detected: ${recyclableItems.map(item => item.class).join(', ')}!`);
        } else if (trashItems.length === 0) {
          setDetectionStatus('No objects detected');
        } else {
          setDetectionStatus(`Non-recyclable items detected: ${trashItems.map(item => item.class).join(', ')}`);
        }
        
        // Draw detections if any were found
        if (allDetectedItems.length > 0) {
          drawDetections(trashItems, allDetectedItems);
        }
      } catch (detectionError) {
        console.error('Detection specific error:', detectionError);
        setErrorCount(prev => prev + 1);
        
        // Simpler error handling
        if (errorCount < 3) {
          setDetectionStatus('Detection error, retrying...');
        } else {
          // Try to recover by forcing model reload if errors persist
          setDetectionStatus('Detection errors occurring. Trying to recover...');
          
          // Attempt to reload model if too many errors
          if (errorCount >= 5) {
            modelRef.current = null;
            setModelLoaded(false);
            setDetectionStatus('Reloading model due to errors...');
            
            // Force reload the model
            setTimeout(() => {
              window.location.reload();
            }, 2000);
            
            detectionRunningRef.current = false;
            return;
          }
        }
      }
      
      // Continue detection loop
      detectionRunningRef.current = false;
      if (isStreaming) {
        requestRef.current = requestAnimationFrame(detectObjects);
      }
    } catch (error) {
      console.error('Detection outer error:', error);
      detectionRunningRef.current = false;
      
      // Continue despite errors
      if (isStreaming) {
        requestRef.current = requestAnimationFrame(detectObjects);
      }
    }
  }, [isStreaming, modelLoaded, errorCount, isNotTrash, isRecyclable]);

  // Draw detection boxes on canvas
  const drawDetections = (objects: DetectedObject[], allPredictions: DetectedObject[] = []) => {
    if (!canvasRef.current || !videoRef.current) return;

    const ctx = canvasRef.current.getContext('2d', { alpha: true }); // Use alpha for transparency
    if (!ctx) return;
    
    try {
      // Get video dimensions
      const videoWidth = videoRef.current.videoWidth;
      const videoHeight = videoRef.current.videoHeight;
      
      // Get display dimensions (canvas/container size)
      const displayWidth = canvasRef.current.clientWidth;
      const displayHeight = canvasRef.current.clientHeight;
      
      // Skip if dimensions are invalid
      if (displayWidth === 0 || displayHeight === 0 || videoWidth === 0 || videoHeight === 0) {
        return;
      }
      
      // Set canvas dimensions to match display area (only if needed)
      if (canvasRef.current.width !== displayWidth || canvasRef.current.height !== displayHeight) {
        canvasRef.current.width = displayWidth;
        canvasRef.current.height = displayHeight;
      }
      
      // Calculate scaling factors
      const scaleX = displayWidth / videoWidth;
      const scaleY = displayHeight / videoHeight;
      
      // For object-fit: cover, we use the larger scale factor
      const scale = Math.max(scaleX, scaleY);
      
      // Calculate offset to center the video
      const offsetX = (displayWidth - videoWidth * scale) / 2;
      const offsetY = (displayHeight - videoHeight * scale) / 2;
      
      // Clear previous drawings - use clearRect to ensure transparency
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      
      // Combine all objects to draw them in one pass
      const allObjects = [
        ...objects.map(obj => ({ ...obj, isTrash: true })),
        ...allPredictions.filter(obj => isNotTrash(obj.class)).map(obj => ({ ...obj, isTrash: false }))
      ];
      
      // Draw all objects in one go for better performance
      drawObjectsOnCanvas(allObjects, scale, offsetX, offsetY);
    } catch (error) {
      console.error('Error drawing detections:', error);
    }
  };
  
  // Helper function to draw objects on canvas with specified color and label
  const drawObjectsOnCanvas = (
    objects: (DetectedObject & { isTrash?: boolean })[], 
    scale: number = 1,
    offsetX: number = 0,
    offsetY: number = 0
  ) => {
    if (!canvasRef.current || !videoRef.current) return;

    const ctx = canvasRef.current.getContext('2d', { alpha: true }); // Use alpha for transparency
    if (!ctx) return;
    
    // Prepare all coordinates first
    const objectsWithCoords = objects.map(obj => {
      // Apply scaling to coordinates and dimensions
      const scaledX = obj.x * scale + offsetX;
      const scaledY = obj.y * scale + offsetY;
      const scaledWidth = obj.width * scale;
      const scaledHeight = obj.height * scale;
      
      // Calculate the mirrored x-coordinate for the box
      const mirroredX = canvasRef.current!.width - scaledX - scaledWidth;
      
      // Determine color based on object type
      let color;
      if (!obj.isTrash) {
        color = '#8E8E93'; // Not trash
      } else if (isRecyclable(obj.class)) {
        color = '#34C759'; // Recyclable
      } else {
        color = '#FF3B30'; // Non-recyclable
      }
      
      // Prepare label text
      let labelText = `${obj.class} (${obj.confidence}%)`;
      
      return {
        ...obj,
        scaledX: scaledX,
        scaledY: scaledY,
        scaledWidth: scaledWidth,
        scaledHeight: scaledHeight,
        screenX: mirroredX,
        screenY: scaledY,
        color,
        labelText
      };
    });
    
    // Draw all bounding boxes first
    ctx.save();
    ctx.scale(-1, 1);
    ctx.translate(-canvasRef.current.width, 0);
    
    ctx.lineWidth = 2;
    
    // Draw all boxes in one path for better performance
    objectsWithCoords.forEach(obj => {
      ctx.strokeStyle = obj.color;
      ctx.strokeRect(obj.scaledX, obj.scaledY, obj.scaledWidth, obj.scaledHeight);
    });
    
    ctx.restore();
    
    // Draw all labels (simplified for performance)
    ctx.font = '600 12px -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif';
    
    objectsWithCoords.forEach(obj => {
      // Simple black background for label
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      const textWidth = ctx.measureText(obj.labelText).width;
      ctx.fillRect(obj.screenX, obj.screenY - 20, textWidth + 6, 20);
      
      // Draw label text
      ctx.fillStyle = obj.color;
      ctx.fillText(obj.labelText, obj.screenX + 3, obj.screenY - 5);
    });
  };

  // Clear canvas
  const clearCanvas = () => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };
  
  // Start/stop detection loop when streaming state changes
  useEffect(() => {
    // Cleanup any existing detection loop
    if (requestRef.current) {
      cancelAnimationFrame(requestRef.current);
      requestRef.current = null;
    }
    
    // If we're streaming and model is loaded, start detection
    if (isStreaming && modelLoaded) {
      // Reset error count on new streaming session
      setErrorCount(0);
      detectionRunningRef.current = false;
      
      // Add a small delay before starting detection to let video initialize
      setTimeout(() => {
        if (isStreaming) {
          requestRef.current = requestAnimationFrame(detectObjects);
        }
      }, 500);
    }
    
    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
        requestRef.current = null;
      }
    };
  }, [isStreaming, modelLoaded, detectObjects]);

  // Handle window resize to update canvas dimensions
  useEffect(() => {
    const handleResize = () => {
      if (canvasRef.current) {
        // Update canvas dimensions when viewport changes
        canvasRef.current.width = canvasRef.current.clientWidth;
        canvasRef.current.height = canvasRef.current.clientHeight;
      }
    };

    // Add resize listener
    window.addEventListener('resize', handleResize);
    
    // Initial call to set dimensions
    handleResize();
    
    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  // Adjust detection interval - use more conservative values
  useEffect(() => {
    if (isStreaming && modelLoaded) {
      const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
      
      // Set very conservative detection rates to ensure stability
      if (isMobile) {
        detectionIntervalRef.current = 300; // ~3 fps on mobile - slower but more stable
      } else {
        detectionIntervalRef.current = 150; // ~6 fps on desktop - slower but more stable
      }
      
      console.log(`Detection interval set to ${detectionIntervalRef.current}ms for stability`);
    }
  }, [isStreaming, modelLoaded]);

  // Start webcam
  const startWebcam = async () => {
    try {
      setDetectionStatus('Starting camera...');
      
      // Reset any previous video state
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
        videoRef.current.srcObject = null;
      }
      
      // Get device constraints that work well on mobile and desktop
      const constraints = {
        video: { 
          width: { ideal: 1280 }, // Higher resolution for better visibility
          height: { ideal: 720 },
          facingMode: 'environment', // Use back camera on mobile when available
          frameRate: { ideal: 30 } // Request higher frame rate
        }
      };
      
      // Request camera access with preferred settings
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      if (videoRef.current) {
        // Set video attributes explicitly
        videoRef.current.width = 1280;
        videoRef.current.height = 720;
        videoRef.current.autoplay = true;
        videoRef.current.playsInline = true;
        videoRef.current.muted = true;
        
        // Ensure video element is visible
        videoRef.current.style.display = 'block';
        videoRef.current.style.opacity = '1';
        
        // Set stream
        videoRef.current.srcObject = stream;
        
        // Ensure video plays on iOS
        videoRef.current.setAttribute('playsinline', 'true');
        
        // Wait for video to be ready before setting streaming state
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            // Force play the video
            const playPromise = videoRef.current.play();
            
            // Handle play promise to avoid uncaught promise errors
            if (playPromise !== undefined) {
              playPromise.then(() => {
                console.log('Video is playing successfully');
                
                // Ensure the video is visible
                if (videoRef.current) {
                  videoRef.current.style.display = 'block';
                  videoRef.current.style.opacity = '1';
                }
                
                setIsStreaming(true);
                setDetectedObjects([]);
                
                // Update canvas dimensions once video is loaded
                if (canvasRef.current) {
                  canvasRef.current.width = canvasRef.current.clientWidth;
                  canvasRef.current.height = canvasRef.current.clientHeight;
                  // Ensure canvas is transparent
                  const ctx = canvasRef.current.getContext('2d', { alpha: true });
                  if (ctx) {
                    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                  }
                }
              }).catch(error => {
                console.error('Error playing video:', error);
                setDetectionStatus('Error starting video stream. Please try again.');
              });
            }
          }
        };
      }
    } catch (err) {
      console.error('Error accessing webcam:', err);
      setDetectionStatus('Error accessing webcam');
    }
  };

  // Stop webcam
  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      // Stop all tracks in the stream
      const stream = videoRef.current.srcObject as MediaStream;
      const tracks = stream.getTracks();
      tracks.forEach(track => track.stop());
      
      // Reset state
      videoRef.current.srcObject = null;
      
      // Clear canvas
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d', { alpha: true });
        if (ctx) {
          ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
      }
      
      setIsStreaming(false);
      setDetectionStatus('Webcam stopped');
      setDetectedObjects([]);
      
      // Reset error count and detection states
      setErrorCount(0);
      setRecycleDetected(false);
    }
  };

  // Toggle webcam
  const toggleWebcam = () => {
    if (isStreaming) {
      stopWebcam();
    } else {
      startWebcam();
    }
  };
  
  // Toggle details panel
  const toggleDetailsPanel = () => {
    setShowDetailsPanel(!showDetailsPanel);
  };

  // Format detection status for display
  const getFormattedStatus = () => {
    if (!modelLoaded) {
      return 'Loading AI model...';
    }
    
    if (!isStreaming) {
      return 'Ready to scan';
    }
    
    const fpsDisplay = fps > 0 ? ` (${fps} FPS)` : '';
    
    if (detectedObjects.length === 0) {
      return `No items detected${fpsDisplay}`;
    }
    
    const recyclableCount = detectedObjects.filter(obj => isRecyclable(obj.class)).length;
    const nonRecyclableCount = detectedObjects.length - recyclableCount;
    
    return recyclableCount > 0 
      ? `Found ${recyclableCount} recyclable item${recyclableCount !== 1 ? 's' : ''}${fpsDisplay}`
      : `Found ${nonRecyclableCount} non-recyclable item${nonRecyclableCount !== 1 ? 's' : ''}${fpsDisplay}`;
  };

  return (
    <div className="ios-container">
      <div className="ios-app">
        <div className="ios-camera-container">
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline
            muted
            className="ios-camera-view"
          />
          <canvas 
            ref={canvasRef} 
            className="ios-detection-overlay"
          />
          
          <div className="video-titles">
            <div className="video-title-left">L.I.F.T</div>
            <div className="video-title-right">Recycling Detector</div>
          </div>
          
          {!isStreaming && (
            <div className="ios-camera-placeholder">
              <div className="ios-placeholder-content">
                <div className="ios-camera-icon"></div>
                <div className="ios-placeholder-text">
                  {modelLoaded ? 'Point camera at recyclable items' : 'Loading AI...'}
                </div>
              </div>
            </div>
          )}
          
          {isStreaming && detectedObjects.length > 0 && (
            <div className="ios-detection-pill">
              {recycleDetected ? (
                <div className="ios-recyclable-pill">
                  <span className="ios-pill-icon">♻️</span>
                  <span className="ios-pill-text">Recyclable</span>
                </div>
              ) : (
                <div className="ios-non-recyclable-pill">
                  <span className="ios-pill-icon">🗑️</span>
                  <span className="ios-pill-text">Non-Recyclable</span>
                </div>
              )}
            </div>
          )}
          
          <div className={`ios-camera-controls ${isStreaming ? 'active' : ''}`}>
            <button 
              className="ios-camera-button"
              onClick={toggleWebcam}
              disabled={!modelLoaded && !isStreaming}
              aria-label={isStreaming ? 'Stop Camera' : 'Start Camera'}
            >
              <div className="ios-button-inner">
                {isStreaming ? 
                  <span className="ios-stop-icon"></span> : 
                  <span className="ios-start-icon"></span>
                }
              </div>
            </button>
            
            <button 
              className="ios-info-button"
              onClick={toggleDetailsPanel}
              aria-label="Toggle Details"
            >
              <div className="ios-info-icon"></div>
            </button>
          </div>
        </div>
        
        <div className="ios-status-display">
          <div className="ios-status-text">
            {getFormattedStatus()}
          </div>
        </div>
        
        {showDetailsPanel && (
          <div className="ios-detail-panel">
            <div className="ios-panel-header">
              <h3>Detection Details</h3>
              <button 
                className="ios-close-button"
                onClick={toggleDetailsPanel}
                aria-label="Close Details"
              >
                <span className="ios-close-icon">×</span>
              </button>
            </div>
            <div className="ios-panel-content">
              <div className="ios-detail-status">{detectionStatus}</div>
              
              {detectedObjects.length > 0 && (
                <div className="ios-object-list">
                  <h4>Detected Items</h4>
                  <ul>
                    {detectedObjects.map((obj, idx) => (
                      <li key={idx} className={isRecyclable(obj.class) ? 'recyclable' : 'non-recyclable'}>
                        <div className="ios-object-name">{obj.class}</div>
                        <div className="ios-object-confidence">{obj.confidence}%</div>
                        <div className="ios-object-status">
                          {isRecyclable(obj.class) ? 'Recyclable' : 'Non-Recyclable'}
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default WebcamComponent; 