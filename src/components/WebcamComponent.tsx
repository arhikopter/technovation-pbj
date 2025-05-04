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

  // Load the TensorFlow model when component mounts
  useEffect(() => {
    let mounted = true;
    
    async function loadModel() {
      try {
        if (mounted) setDetectionStatus('Loading model...');
        
        // Enable optimizations for TensorFlow.js
        console.log('TensorFlow.js version:', tf.version.tfjs);
        
        // Configure TensorFlow.js for optimal performance
        tf.ENV.set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
        tf.ENV.set('WEBGL_FORCE_F16_TEXTURES', true);
        tf.ENV.set('WEBGL_CPU_FORWARD', false);
        tf.ENV.set('WEBGL_PACK', true);
        
        // Set memory efficient backend
        await tf.setBackend('webgl');
        await tf.ready();
        console.log('Using TensorFlow backend:', tf.getBackend());
        
        // Warm up the engine to avoid initial lag
        const dummyTensor = tf.zeros([1, 224, 224, 3]);
        dummyTensor.dispose();
        
        // Load the COCO-SSD model with a faster model for real-time performance
        const model = await cocoSsd.load({
          base: 'lite_mobilenet_v2'  // Using lighter model for better speed
        });
        
        if (!mounted) return;
        
        modelRef.current = model;
        console.log('Model loaded successfully');
        
        // Run garbage collection
        try {
          // @ts-ignore
          if (window.gc) {
            // @ts-ignore
            window.gc();
          }
        } catch (e) {
          console.log('GC not available');
        }
        
        setModelLoaded(true);
        setDetectionStatus('Model loaded. Ready to start.');
      } catch (error) {
        console.error('Failed to load model:', error);
        if (mounted) {
          setDetectionStatus(`Error loading model: ${error instanceof Error ? error.message : 'Unknown error'}`);
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

  // Main detection function using TensorFlow.js
  const detectObjects = useCallback(async () => {
    // Skip if detection is already running or prerequisites not met
    if (detectionRunningRef.current || !isStreaming || !videoRef.current || !modelRef.current) {
      if (isStreaming && modelLoaded) {
        requestRef.current = requestAnimationFrame(detectObjects);
      }
      return;
    }
    
    // Update FPS counter
    fpsCountRef.current++;
    const now = performance.now();
    if (now - lastFpsUpdateRef.current >= 1000) {
      setFps(fpsCountRef.current);
      fpsCountRef.current = 0;
      lastFpsUpdateRef.current = now;
    }
    
    // Throttle detection frequency
    if (now - lastDetectionTimeRef.current < detectionIntervalRef.current) {
      // Not enough time has passed since last detection
      requestRef.current = requestAnimationFrame(detectObjects);
      return;
    }
    
    // Update last detection time
    lastDetectionTimeRef.current = now;
    detectionRunningRef.current = true;
    
    try {
      // Make sure the video is ready
      if (videoRef.current.readyState < 2) {
        detectionRunningRef.current = false;
        requestRef.current = requestAnimationFrame(detectObjects);
        return;
      }
      
      // Ensure TensorFlow backend is ready
      if (!tf.getBackend()) {
        await tf.setBackend('webgl');
        await tf.ready();
      }
      
      // Use tidy to automatically clean up tensors
      tf.tidy(() => {
        // Run detection on current video frame with optimized parameters
        modelRef.current!.detect(videoRef.current!, 5, 0.5).then(predictions => {
          // Process all detected objects
          const allDetectedItems = processDetections(predictions);
          
          // Check for not-trash items (like humans)
          const notTrashItems = allDetectedItems.filter(obj => isNotTrash(obj.class));
          const hasNotTrashItems = notTrashItems.length > 0;

          // Filter out "not trash" items from trash detection
          const trashItems = allDetectedItems.filter(obj => !isNotTrash(obj.class));
          
          // Only update detected objects with items that are classified as trash
          setDetectedObjects(trashItems);
          
          // Check for recyclable items only among trash items
          const recyclableItems = trashItems.filter(obj => isRecyclable(obj.class));
          
          // Update recyclable detection state
          setRecycleDetected(recyclableItems.length > 0);
          
          // Update status messages
          if (hasNotTrashItems) {
            const notTrashNames = notTrashItems.map(item => item.class).join(', ');
            
            if (recyclableItems.length > 0) {
              // Both not-trash and recyclable items detected
              setDetectionStatus(`Detected: ${notTrashNames} (not trash) and recyclable items: ${recyclableItems.map(item => item.class).join(', ')}`);
            } else if (trashItems.length > 0) {
              // Not-trash and non-recyclable trash detected
              setDetectionStatus(`Detected: ${notTrashNames} (not trash) and non-recyclable items`);
            } else {
              // Only not-trash items detected
              setDetectionStatus(`Detected: ${notTrashNames} (not trash)`);
            }
          } else if (recyclableItems.length > 0) {
            // Only recyclable items detected
            setDetectionStatus(`Recyclable items detected: ${recyclableItems.map(item => item.class).join(', ')}!`);
          } else if (trashItems.length === 0) {
            // No objects detected at all
            setDetectionStatus('No objects detected');
          } else {
            // Only non-recyclable trash detected
            setDetectionStatus(`No recyclable items found among ${trashItems.length} detected object(s)`);
          }
          
          // Draw detection boxes for all objects (including not-trash)
          drawDetections(trashItems, allDetectedItems);
          
          // Periodically clean up memory
          if (fpsCountRef.current % 30 === 0) {
            tf.engine().endScope();
            tf.engine().startScope();
          }
          
          // Continue detection loop
          detectionRunningRef.current = false;
          if (isStreaming) {
            requestRef.current = requestAnimationFrame(detectObjects);
          }
        }).catch(error => {
          console.error('Detection error:', error);
          setErrorCount(prev => prev + 1);
          
          // Handle errors
          setDetectionStatus(errorCount < 3 ? 'Detection error, retrying...' : 'Detection error occurred');
          detectionRunningRef.current = false;
          
          // Retry with increasing delay if not too many errors
          if (errorCount < 5 && isStreaming) {
            setTimeout(() => {
              if (isStreaming) {
                requestRef.current = requestAnimationFrame(detectObjects);
              }
            }, errorCount * 100); // Shorter delay for faster recovery
          }
        });
      });
    } catch (error) {
      console.error('Detection error:', error);
      detectionRunningRef.current = false;
      
      // Continue despite errors for more fluid experience
      if (isStreaming) {
        requestRef.current = requestAnimationFrame(detectObjects);
      }
    }
  }, [isStreaming, modelLoaded, errorCount]);

  // Draw detection boxes on canvas
  const drawDetections = (objects: DetectedObject[], allPredictions: DetectedObject[] = []) => {
    if (!canvasRef.current || !videoRef.current) return;

    const ctx = canvasRef.current.getContext('2d', { alpha: false });
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
      
      // Clear previous drawings
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

    const ctx = canvasRef.current.getContext('2d', { alpha: false });
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

  // Adjust detection interval based on device performance
  useEffect(() => {
    if (isStreaming && modelLoaded) {
      // Check if we're on a mobile device
      const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
      
      // Set different intervals based on device type - increase speeds for all devices
      if (isMobile) {
        detectionIntervalRef.current = 100; // ~10 fps on mobile
      } else {
        detectionIntervalRef.current = 50; // ~20 fps on desktop
      }
      
      console.log(`Detection interval set to ${detectionIntervalRef.current}ms`);
    }
  }, [isStreaming, modelLoaded]);

  // Start webcam
  const startWebcam = async () => {
    try {
      setDetectionStatus('Starting camera...');
      
      // Get device constraints that work well on mobile and desktop
      const constraints = {
        video: { 
          width: { ideal: 640 }, // Lower resolution for faster processing
          height: { ideal: 480 },
          facingMode: 'environment', // Use back camera on mobile when available
          frameRate: { ideal: 30 } // Request higher frame rate
        }
      };
      
      // Request camera access with preferred settings
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        
        // Wait for video to be ready before setting streaming state
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            videoRef.current.play();
            setIsStreaming(true);
            setDetectedObjects([]);
            
            // Update canvas dimensions once video is loaded
            if (canvasRef.current) {
              canvasRef.current.width = canvasRef.current.clientWidth;
              canvasRef.current.height = canvasRef.current.clientHeight;
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
      setIsStreaming(false);
      setDetectionStatus('Webcam stopped');
      setDetectedObjects([]);
      clearCanvas();
      
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