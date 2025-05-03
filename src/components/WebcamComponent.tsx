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
  
  // State variables for micro:bit
  const [microbitConnected, setMicrobitConnected] = useState(false);
  const [microbitConnecting, setMicrobitConnecting] = useState(false);
  const [microbitButtonA, setMicrobitButtonA] = useState(false);
  const [microbitButtonB, setMicrobitButtonB] = useState(false);
  const [microbitConnectionStatus, setMicrobitConnectionStatus] = useState('Disconnected');
  
  // useRef for micro:bit device and services
  const microbitDeviceRef = useRef<BluetoothDevice | null>(null);
  const microbitServicesRef = useRef<Record<string, any>>({});
  const microbitGattServerRef = useRef<BluetoothRemoteGATTServer | null>(null);
  
  // No throttling for important state changes
  const lastMicrobitUpdateRef = useRef<number>(0);
  const microbitUpdateIntervalMs = 100; // 100ms between updates (reduced from 500ms)

  // Load the TensorFlow model when component mounts
  useEffect(() => {
    let mounted = true;
    
    // Log recyclable items for debugging
    console.log("RECYCLABLE_TRASH_ITEMS:", RECYCLABLE_TRASH_ITEMS);
    console.log("WRAPPER_LIKE_ITEMS:", WRAPPER_LIKE_ITEMS);
    
    async function loadModel() {
      try {
        if (mounted) setDetectionStatus('Loading model...');
        
        // Enable debug mode for TensorFlow.js
        tf.enableDebugMode();
        console.log('TensorFlow.js version:', tf.version.tfjs);
        
        // Set memory efficient backend
        await tf.setBackend('webgl');
        await tf.ready();
        console.log('Using TensorFlow backend:', tf.getBackend());
        
        // Load the COCO-SSD model with a base model
        const model = await cocoSsd.load({
          base: 'lite_mobilenet_v2'  // Using a lighter model that's faster
        });
        
        if (!mounted) return;
        
        modelRef.current = model;
        console.log('Model loaded successfully');
        
        setModelLoaded(true);
        setDetectionStatus('Model loaded. Click Start to begin.');
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
      
      // Disconnect from micro:bit if connected
      if (microbitGattServerRef.current && microbitConnected) {
        microbitGattServerRef.current.disconnect();
      }
    };
  }, []);

  // Connect to micro:bit
  const connectToMicrobit = async () => {
    setMicrobitConnectionStatus('Connecting...');
    setMicrobitConnecting(true);

    try {
      if (!navigator.bluetooth) {
        console.error('Web Bluetooth API is not available in this browser');
        setMicrobitConnectionStatus('Bluetooth not supported');
        setMicrobitConnecting(false);
        return;
      }

      console.log('Requesting micro:bit device...');
      
      // Request a micro:bit device directly from the browser's Bluetooth API
      const device = await navigator.bluetooth.requestDevice({
        filters: [{ namePrefix: 'BBC micro:bit' }],
        optionalServices: [
          '0000180a-0000-1000-8000-00805f9b34fb', // Device Information Service
          'e95dd91d-251d-470a-a062-fa1922dfa9a8', // LED Service
          'e95d9882-251d-470a-a062-fa1922dfa9a8', // Button Service
          'e95d6100-251d-470a-a062-fa1922dfa9a8'  // Temperature Service
        ]
      });
      
      microbitDeviceRef.current = device;
      
      console.log('Connecting to GATT server...');
      const server = await device.gatt?.connect();
      microbitGattServerRef.current = server || null;
      
      // Store services
      const services: Record<string, any> = {};
      
      // Button Service
      try {
        if (server) {
          const buttonService = await server.getPrimaryService('e95d9882-251d-470a-a062-fa1922dfa9a8');
          const buttonACharacteristic = await buttonService.getCharacteristic('e95dda90-251d-470a-a062-fa1922dfa9a8');
          const buttonBCharacteristic = await buttonService.getCharacteristic('e95dda91-251d-470a-a062-fa1922dfa9a8');
          
          // Start notifications for button A
          await buttonACharacteristic.startNotifications();
          buttonACharacteristic.addEventListener('characteristicvaluechanged', (event: Event) => {
            const target = event.target as BluetoothRemoteGATTCharacteristic;
            const value = target.value;
            if (value) {
              const pressed = value.getUint8(0) === 1;
              console.log(`Button A ${pressed ? 'pressed' : 'released'}`);
              setMicrobitButtonA(pressed);
              
              // Toggle webcam streaming when button A is pressed
              if (pressed) {
                setIsStreaming(prevState => !prevState);
              }
            }
          });
          
          // Start notifications for button B
          await buttonBCharacteristic.startNotifications();
          buttonBCharacteristic.addEventListener('characteristicvaluechanged', (event: Event) => {
            const target = event.target as BluetoothRemoteGATTCharacteristic;
            const value = target.value;
            if (value) {
              const pressed = value.getUint8(0) === 1;
              console.log(`Button B ${pressed ? 'pressed' : 'released'}`);
              setMicrobitButtonB(pressed);
              
              // Toggle detection when button B is pressed
              if (pressed) {
                // Toggle streaming as well
                setIsStreaming(prevState => !prevState);
              }
            }
          });
          
          // Store the service
          services.buttonService = buttonService;
        }
      } catch (buttonError) {
        console.error('Error setting up button service:', buttonError);
      }
      
      // LED Service
      try {
        if (server) {
          const ledService = await server.getPrimaryService('e95dd91d-251d-470a-a062-fa1922dfa9a8');
          const ledMatrixStateCharacteristic = await ledService.getCharacteristic('e95d7b77-251d-470a-a062-fa1922dfa9a8');
          const ledTextCharacteristic = await ledService.getCharacteristic('e95d93ee-251d-470a-a062-fa1922dfa9a8');
          
          // Create a function to write text to the LED display
          const writeText = async (text: string) => {
            const encoder = new TextEncoder();
            const data = encoder.encode(text);
            await ledTextCharacteristic.writeValue(data);
          };
          
          // Create a function to write a pattern to the LED display
          const writeMatrixState = async (pattern: boolean[][]) => {
            const byteArray = new Uint8Array(5);
            for (let i = 0; i < 5; i++) {
              let byte = 0;
              for (let j = 0; j < 5; j++) {
                if (pattern[i][j]) {
                  byte |= 1 << j;
                }
              }
              byteArray[i] = byte;
            }
            await ledMatrixStateCharacteristic.writeValue(byteArray);
          };
          
          // Store the service with functions
          services.ledService = {
            writeText,
            writeMatrixState
          };
          
          // Display a welcome pattern instead of text
          // Show a smile pattern
          const smilePattern = [
            [false, false, false, false, false],
            [false, true, false, true, false],
            [false, false, false, false, false],
            [true, false, false, false, true],
            [false, true, true, true, false]
          ];
          await writeMatrixState(smilePattern);
          
          // We don't need the setTimeout for text now since we're showing the pattern immediately
        }
      } catch (ledError) {
        console.error('Error setting up LED service:', ledError);
      }
      
      // Store all services
      microbitServicesRef.current = services;
      
      // Device disconnect event handler
      device.addEventListener('gattserverdisconnected', () => {
        console.log('micro:bit disconnected');
        setMicrobitConnected(false);
        setMicrobitButtonA(false);
        setMicrobitButtonB(false);
        setMicrobitConnectionStatus('Disconnected');
      });
      
      setMicrobitConnected(true);
      setMicrobitConnectionStatus('Connected');
      console.log('Successfully connected to micro:bit');
      
    } catch (error) {
      console.error('Failed to connect to micro:bit:', error);
      setMicrobitConnectionStatus('Connection failed');
      setMicrobitConnected(false);
    } finally {
      setMicrobitConnecting(false);
    }
  };
  
  // Disconnect from micro:bit
  const disconnectFromMicrobit = () => {
    if (!microbitConnected || !microbitGattServerRef.current) return;
    
    try {
      // Disconnect from GATT server
      microbitGattServerRef.current.disconnect();
      
      setMicrobitConnected(false);
      setMicrobitButtonA(false);
      setMicrobitButtonB(false);
      setMicrobitConnectionStatus('Disconnected');
      console.log('Disconnected from micro:bit');
    } catch (error) {
      console.error('Error disconnecting from micro:bit:', error);
    }
  };

  // Display text on micro:bit
  const displayOnMicrobit = async (text: string) => {
    if (!microbitConnected || !microbitServicesRef.current.ledService) return;
    
    // Check if we should throttle updates
    const now = Date.now();
    if (now - lastMicrobitUpdateRef.current < microbitUpdateIntervalMs) {
      return; // Skip this update if not enough time has passed
    }
    
    // Update the timestamp
    lastMicrobitUpdateRef.current = now;
    
    try {
      console.log(`Displaying "${text}" on micro:bit`);
      
      const ledService = microbitServicesRef.current.ledService;
      if (ledService && ledService.writeText) {
        await ledService.writeText(text);
        console.log(`Sent "${text}" to micro:bit display`);
      }
    } catch (error) {
      console.error('Error displaying text on micro:bit:', error);
    }
  };

  // Display pattern on micro:bit for recyclable status
  const displayRecyclablePatternOnMicrobit = async () => {
    if (!microbitConnected || !microbitServicesRef.current.ledService) return;
    
    const ledService = microbitServicesRef.current.ledService;
    if (!ledService || !ledService.writeMatrixState) return;
    
    try {
      // Right arrow pattern for recyclable items
      const rightArrowPattern = [
        [false, false, true, false, false],
        [false, false, false, true, false],
        [true, true, true, true, true],
        [false, false, false, true, false],
        [false, false, true, false, false]
      ];
      
      // Left arrow pattern for non-recyclable items
      const leftArrowPattern = [
        [false, false, true, false, false],
        [false, true, false, false, false],
        [true, true, true, true, true],
        [false, true, false, false, false],
        [false, false, true, false, false]
      ];
      
      // Empty pattern for when no objects are detected
      const emptyPattern = [
        [false, false, false, false, false],
        [false, false, false, false, false],
        [false, false, false, false, false],
        [false, false, false, false, false],
        [false, false, false, false, false]
      ];
      
      // Choose pattern based on detection state and if objects are present
      let patternToShow;
      
      if (detectedObjects.length === 0) {
        // No objects detected, show nothing
        patternToShow = emptyPattern;
      } else if (recycleDetected) {
        // Recyclable objects detected, show right arrow
        patternToShow = rightArrowPattern;
      } else {
        // Non-recyclable objects detected, show left arrow
        patternToShow = leftArrowPattern;
      }
      
      // Display the pattern
      await ledService.writeMatrixState(patternToShow);
    } catch (error) {
      console.error('Error displaying pattern on micro:bit:', error);
    }
  };

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
      
      // Run detection on current video frame
      const predictions = await modelRef.current.detect(videoRef.current, undefined, 0.5);
      
      // Reset error count on successful detection
      if (errorCount > 0) setErrorCount(0);
      
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
      drawDetections(trashItems);
      
      // Continue detection loop
      detectionRunningRef.current = false;
      if (isStreaming) {
        requestRef.current = requestAnimationFrame(detectObjects);
      }
    } catch (error) {
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
        }, errorCount * 500);
      }
    }
  }, [isStreaming, modelLoaded, errorCount]);

  // Draw detection boxes on canvas
  const drawDetections = (objects: DetectedObject[]) => {
    if (!canvasRef.current || !videoRef.current) return;

    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;
    
    try {
      // Get video dimensions
      const videoWidth = videoRef.current.videoWidth;
      const videoHeight = videoRef.current.videoHeight;
      
      // Set canvas dimensions
      canvasRef.current.width = videoWidth || 640;
      canvasRef.current.height = videoHeight || 480;
      
      // Clear previous drawings
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      
      // Draw trash objects with normal colors
      drawObjectsOnCanvas(objects);
      
      // Find any "not trash" objects and draw them separately
      if (videoRef.current && modelRef.current) {
        modelRef.current.detect(videoRef.current, undefined, 0.5).then(predictions => {
          const notTrashObjects = processDetections(
            predictions.filter(pred => isNotTrash(pred.class))
          );
          
          // Draw not-trash objects (like humans) with blue color
          if (notTrashObjects.length > 0) {
            drawObjectsOnCanvas(notTrashObjects, 'blue', 'Not Trash');
          }
        }).catch(err => {
          console.error('Error detecting not-trash objects:', err);
        });
      }
    } catch (error) {
      console.error('Error drawing detections:', error);
    }
  };
  
  // Helper function to draw objects on canvas with specified color and label
  const drawObjectsOnCanvas = (objects: DetectedObject[], defaultColor?: string, labelPrefix?: string) => {
    if (!canvasRef.current || !videoRef.current) return;

    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;
    
    // Get video dimensions
    const videoWidth = videoRef.current.videoWidth;
    const videoHeight = videoRef.current.videoHeight;
    
    // Save the transformed coordinates for label drawing after restoring context
    const objectsWithScreenCoords = objects.map(obj => {
      // Calculate the mirrored x-coordinate for the box
      const mirroredX = videoWidth - obj.x - obj.width;
      
      return {
        ...obj,
        screenX: mirroredX, // Store the screen coordinates
        screenY: obj.y
      };
    });
    
    // First draw the bounding boxes with mirrored coordinates
    ctx.save();
    
    // Handle mirrored video - flip the context horizontally
    ctx.scale(-1, 1);
    ctx.translate(-canvasRef.current.width, 0);
    
    // Draw only the bounding boxes in the mirrored context
    objects.forEach(obj => {
      // Choose color based on object class or use provided default
      let color;
      if (defaultColor) {
        color = defaultColor;
      } else {
        color = isRecyclable(obj.class) ? 'lime' : 'red';
      }
      
      // Draw bounding box (flipped)
      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      ctx.strokeRect(obj.x, obj.y, obj.width, obj.height);
    });
    
    // Restore context to normal (unflipped) state
    ctx.restore();
    
    // Now draw the labels in normal orientation so they're readable
    objectsWithScreenCoords.forEach(obj => {
      // Choose color based on classification or use provided default
      let color;
      if (defaultColor) {
        color = defaultColor;
      } else {
        color = isRecyclable(obj.class) ? 'lime' : 'red';
      }
      
      // Prepare label text
      let labelText = `${obj.class}: ${obj.confidence}%`;
      
      // Add classification
      if (labelPrefix) {
        labelText = `${obj.class} (${labelPrefix}): ${obj.confidence}%`;
      } else if (isRecyclable(obj.class)) {
        labelText = `${obj.class} (Recyclable): ${obj.confidence}%`;
      } else {
        labelText = `${obj.class} (Non-Recyclable): ${obj.confidence}%`;
      }
      
      // Draw text background in normal orientation
      ctx.font = 'bold 16px Arial';
      const textMetrics = ctx.measureText(labelText);
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(obj.screenX, obj.screenY - 30, textMetrics.width + 10, 30);
      
      // Draw label text in normal orientation
      ctx.fillStyle = color;
      ctx.fillText(labelText, obj.screenX + 5, obj.screenY - 10);
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

  // Update micro:bit display when recyclable detection state changes or objects appear/disappear
  useEffect(() => {
    if (microbitConnected) {
      displayRecyclablePatternOnMicrobit();
    }
  }, [recycleDetected, microbitConnected, detectedObjects]);

  // Start webcam
  const startWebcam = async () => {
    try {
      setDetectionStatus('Starting camera...');
      
      // Request camera access with preferred settings
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 }
        } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        
        // Wait for video to be ready before setting streaming state
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            videoRef.current.play();
            setIsStreaming(true);
            setDetectedObjects([]);
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

  return (
    <div className="webcam-container">
      <div className="status-icons">
        <div className={`icon right-arrow ${detectedObjects.length > 0 && recycleDetected ? 'active' : ''}`}>
          →
        </div>
        <div className={`icon left-arrow ${detectedObjects.length > 0 && !recycleDetected ? 'active' : ''}`}>
          ←
        </div>
      </div>
      <div className="video-container">
        <video 
          ref={videoRef} 
          autoPlay 
          playsInline
          muted
          className="webcam-video"
        />
        <canvas 
          ref={canvasRef} 
          className="detection-canvas"
        />
      </div>
      <div className={`detection-status ${recycleDetected ? 'recyclable-detected' : ''}`}>
        {detectionStatus}
      </div>
      <div className="microbit-connection">
        <button 
          onClick={microbitConnected ? disconnectFromMicrobit : connectToMicrobit}
          disabled={microbitConnecting || (!navigator.bluetooth && !microbitConnected)}
          className={`microbit-button ${microbitConnected ? 'disconnect-button' : 'connect-button'}`}
        >
          {microbitConnecting ? 'Connecting...' : (microbitConnected ? 'Disconnect micro:bit' : 'Connect micro:bit')}
        </button>
        <div className={`connection-status ${microbitConnected ? 'connected' : ''}`}>
          {!navigator.bluetooth && !microbitConnected && 'Web Bluetooth not supported in this browser'}
          {navigator.bluetooth && microbitConnected && (
            <>
              {microbitConnectionStatus}
              {microbitButtonA && <span className="microbit-button-indicator buttonA">A</span>}
              {microbitButtonB && <span className="microbit-button-indicator buttonB">B</span>}
            </>
          )}
        </div>
      </div>
      {detectedObjects.length > 0 && (
        <div className="detected-objects">
          <h3>Detected Objects:</h3>
          <ul>
            {detectedObjects.map((obj, index) => (
              <li 
                key={index} 
                className={isRecyclable(obj.class) ? 'recyclable-object' : 'non-recyclable-object'}
              >
                {obj.class}: {obj.confidence}% confidence
                {isRecyclable(obj.class) ? ' (RECYCLABLE)' : ' (NON-RECYCLABLE)'}
              </li>
            ))}
          </ul>
        </div>
      )}
      <div className="button-container">
        <button 
          onClick={startWebcam}
          disabled={isStreaming || !modelLoaded}
          className="webcam-button start-button"
        >
          Start
        </button>
        <button 
          onClick={stopWebcam}
          disabled={!isStreaming}
          className="webcam-button stop-button"
        >
          Stop
        </button>
      </div>
    </div>
  );
};

export default WebcamComponent; 