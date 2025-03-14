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

// Non-recyclable or compostable trash items
const NON_RECYCLABLE_TRASH_ITEMS = [
  'food', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
  'hot dog', 'pizza', 'donut', 'cake', 'organic waste', 'wet waste'
];

// Items that could represent a toy frog in the video
const FROG_PROXY_CLASSES = ['teddy bear', 'toy', 'doll', 'stuffed animal', 'handbag', 'suitcase'];

// Oranges, apples, and other round fruits could be detected as lemons
const LEMON_PROXY_CLASSES = ['orange', 'apple', 'sports ball'];

// Combined trash items list for general trash detection
const TRASH_ITEMS = [
  ...RECYCLABLE_TRASH_ITEMS,
  ...NON_RECYCLABLE_TRASH_ITEMS,
  // Additional items that might be either depending on material
  'vase', 'teddy bear', 'tv', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket'
];

// Note: COCO-SSD doesn't specifically have "wrapper" as a class, but these
// objects might be visually similar to wrappers and could be used as proxies
const WRAPPER_LIKE_ITEMS = [
  'book', 'tie', 'handbag', 'backpack', 'box', 'suitcase'
];

// We'll use teddy bear from COCO-SSD as a proxy for toy frog
const TOY_FROG_CLASSES = ['teddy bear'];

// React component for webcam
const WebcamComponent: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [detectionStatus, setDetectionStatus] = useState<string>('Loading model...');
  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([]);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [isFrogDetected, setIsFrogDetected] = useState(false);
  const [isLemonDetected, setIsLemonDetected] = useState(false);
  const [recycleDetected, setRecycleDetected] = useState(false);
  const [nonRecycleDetected, setNonRecycleDetected] = useState(false);
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

  // Load the TensorFlow model when component mounts
  useEffect(() => {
    let mounted = true;
    
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
            // Convert the pattern to a byte array
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
          
          // Display a welcome message
          await writeText('Hi!');
          
          // After 2 seconds, show a smile
          setTimeout(async () => {
            // Create a pattern for a smile (5x5 grid)
            const pattern = [
              [0, 0, 0, 0, 0],
              [0, 1, 0, 1, 0],
              [0, 0, 0, 0, 0],
              [1, 0, 0, 0, 1],
              [0, 1, 1, 1, 0]
            ];
            await writeMatrixState(pattern.map(row => row.map(val => val === 1)));
          }, 2000);
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

  // Check if an object is considered trash
  const isTrash = (objectClass: string): boolean => {
    return TRASH_ITEMS.includes(objectClass.toLowerCase());
  };

  // Check if an object is recyclable trash
  const isRecyclable = (objectClass: string): boolean => {
    return RECYCLABLE_TRASH_ITEMS.includes(objectClass.toLowerCase());
  };

  // Check if an object is non-recyclable trash or compost
  const isNonRecyclable = (objectClass: string): boolean => {
    return NON_RECYCLABLE_TRASH_ITEMS.includes(objectClass.toLowerCase());
  };

  // Check if an object is considered a wrapper-like item
  const isWrapper = (objectClass: string): boolean => {
    return WRAPPER_LIKE_ITEMS.includes(objectClass.toLowerCase());
  };

  // Check if an object is considered a toy frog (using teddy bear as proxy)
  const isToyFrog = (objectClass: string): boolean => {
    return TOY_FROG_CLASSES.includes(objectClass.toLowerCase());
  };

  // Check if an object could potentially be a lemon
  const isPotentialLemon = (objectClass: string): boolean => {
    return LEMON_PROXY_CLASSES.includes(objectClass.toLowerCase());
  };

  // Process lemon detection with custom logic
  const processLemonDetection = (objects: DetectedObject[]): boolean => {
    // Look for objects that could be lemons
    const potentialLemons = objects.filter(obj => isPotentialLemon(obj.class));
    
    // If we have potential lemons, we'll consider them lemons for this demo
    return potentialLemons.length > 0;
  };

  // Main detection function using TensorFlow.js
  const detectObjects = useCallback(async () => {
    // Skip if detection is already running to prevent concurrent calls
    if (detectionRunningRef.current || !isStreaming || !videoRef.current || !modelRef.current) {
      if (isStreaming && modelLoaded) {
        // Schedule next frame if we're streaming but skipped this detection
        requestRef.current = requestAnimationFrame(detectObjects);
      }
      return;
    }
    
    detectionRunningRef.current = true;
    
    try {
      // Make sure the video is actually ready
      if (videoRef.current.readyState < 2) {
        console.log('Video not ready yet');
        detectionRunningRef.current = false;
        requestRef.current = requestAnimationFrame(detectObjects);
        return;
      }
      
      // Ensure TensorFlow backend is initialized
      if (!tf.getBackend()) {
        await tf.setBackend('webgl');
        await tf.ready();
      }
      
      // Run detection on current video frame
      const predictions = await modelRef.current.detect(videoRef.current, undefined, 0.5);
      
      // Reset error count on successful detection
      if (errorCount > 0) setErrorCount(0);
      
      // Process and update detected objects
      const detectedItems = processDetections(predictions);
      setDetectedObjects(detectedItems);
      
      // Find recyclable and non-recyclable items
      const recyclableItems = detectedItems.filter(obj => isRecyclable(obj.class));
      const nonRecyclableItems = detectedItems.filter(obj => isNonRecyclable(obj.class));
      
      // Update recycling state
      setRecycleDetected(recyclableItems.length > 0);
      setNonRecycleDetected(nonRecyclableItems.length > 0);
      
      // Wrappers are also considered recyclable
      const wrapperItems = detectedItems.filter(obj => isWrapper(obj.class));
      if (wrapperItems.length > 0) {
        setRecycleDetected(true);
      }
      
      // Check for toy frog
      const foundFrog = detectedItems.some(obj => isToyFrog(obj.class));
      setIsFrogDetected(foundFrog);
      
      // Check for lemon
      const foundLemon = processLemonDetection(detectedItems);
      setIsLemonDetected(foundLemon);
      
      // Update status based on detection results
      if (foundFrog) {
        setDetectionStatus(`Toy frog detected!`);
        if (microbitConnected) displayOnMicrobit('FROG');
      } else if (foundLemon) {
        setDetectionStatus(`Lemon detected!`);
        if (microbitConnected) displayOnMicrobit('LEMON');
      } else if (recyclableItems.length > 0) {
        const recyclableNames = recyclableItems.map(item => item.class).join(', ');
        setDetectionStatus(`Recyclable items detected: ${recyclableNames}!`);
        // Send to micro:bit - we'll just show the first item
        if (microbitConnected && recyclableItems.length > 0) {
          const shortName = recyclableItems[0].class.substring(0, 5); // First 5 chars
          displayOnMicrobit(shortName);
        }
      } else if (nonRecyclableItems.length > 0) {
        const nonRecyclableNames = nonRecyclableItems.map(item => item.class).join(', ');
        setDetectionStatus(`Non-recyclable items detected: ${nonRecyclableNames}!`);
        // Send to micro:bit - we'll just show the first item
        if (microbitConnected && nonRecyclableItems.length > 0) {
          const shortName = nonRecyclableItems[0].class.substring(0, 5); // First 5 chars
          displayOnMicrobit(shortName);
        }
      } else if (wrapperItems.length > 0) {
        const wrapperNames = wrapperItems.map(item => item.class).join(', ');
        setDetectionStatus(`Recyclable wrappers detected: ${wrapperNames}!`);
        if (microbitConnected) displayOnMicrobit('WRAP');
      } else if (detectedItems.length === 0) {
        setDetectionStatus('No objects detected');
        if (microbitConnected) displayOnMicrobit('NONE');
      } else {
        setDetectionStatus(`Detected ${detectedItems.length} object(s)`);
        // If there's a person, tell micro:bit
        const person = detectedItems.find(obj => obj.class === 'person');
        if (microbitConnected && person) {
          displayOnMicrobit('PERSON');
        } else if (microbitConnected) {
          // Show the first detected object
          const shortName = detectedItems[0].class.substring(0, 5); // First 5 chars
          displayOnMicrobit(shortName);
        }
      }
      
      // Draw detection boxes
      drawDetections(detectedItems);
      
      // Continue detection loop
      detectionRunningRef.current = false;
      if (isStreaming) {
        requestRef.current = requestAnimationFrame(detectObjects);
      }
    } catch (error) {
      console.error('Detection error:', error);
      
      // Increment error count
      setErrorCount(prev => prev + 1);
      
      // Handle error based on count
      if (errorCount < 3) {
        setDetectionStatus('Detection error, retrying...');
      } else {
        setDetectionStatus('Detection error occurred');
        if (microbitConnected) displayOnMicrobit('ERROR');
      }
      
      // Release this detection cycle
      detectionRunningRef.current = false;
      
      // If we've had multiple errors but still want to keep trying
      if (errorCount < 5 && isStreaming) {
        // Delay longer between retries as error count increases
        setTimeout(() => {
          if (isStreaming) {
            requestRef.current = requestAnimationFrame(detectObjects);
          }
        }, errorCount * 500);
      }
    }
  }, [isStreaming, modelLoaded, errorCount, microbitConnected]);

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
        // Choose color based on object class
        let color = 'yellow';
        if (obj.class === 'person') color = 'cyan';
        if (isRecyclable(obj.class)) color = 'lime';
        if (isNonRecyclable(obj.class)) color = 'red';
        if (isWrapper(obj.class)) color = 'orange';
        if (isToyFrog(obj.class)) color = 'magenta';
        if (isPotentialLemon(obj.class)) color = 'yellow';
        
        // Draw bounding box (flipped)
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.strokeRect(obj.x, obj.y, obj.width, obj.height);
      });
      
      // Restore context to normal (unflipped) state
      ctx.restore();
      
      // Now draw the labels in normal orientation so they're readable
      objectsWithScreenCoords.forEach(obj => {
        // Choose color based on object class
        let color = 'yellow';
        if (obj.class === 'person') color = 'cyan';
        if (isRecyclable(obj.class)) color = 'lime';
        if (isNonRecyclable(obj.class)) color = 'red';
        if (isWrapper(obj.class)) color = 'orange';
        if (isToyFrog(obj.class)) color = 'magenta';
        if (isPotentialLemon(obj.class)) color = 'yellow';
        
        // Prepare label text
        let labelText = `${obj.class}: ${obj.confidence}%`;
        
        // Add classification to label
        if (isRecyclable(obj.class)) {
          labelText = `${obj.class} (Recyclable): ${obj.confidence}%`;
        } else if (isNonRecyclable(obj.class)) {
          labelText = `${obj.class} (Non-Recyclable): ${obj.confidence}%`;
        } else if (isPotentialLemon(obj.class)) {
          labelText = `Lemon (${obj.class}): ${obj.confidence}%`;
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
    } catch (error) {
      console.error('Error drawing detections:', error);
    }
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
      setIsFrogDetected(false);
      setIsLemonDetected(false);
      setRecycleDetected(false);
      setNonRecycleDetected(false);
    }
  };

  // Check if any trash is detected (for legacy status coloring)
  const isTrashDetected = detectedObjects.some(obj => isTrash(obj.class) || isWrapper(obj.class));

  // Determine the container classes based on detections
  const containerClasses = [
    'webcam-container',
    isFrogDetected ? 'frog-detected' : '',
    isLemonDetected ? 'lemon-detected' : '',
  ].filter(Boolean).join(' ');

  return (
    <div className={containerClasses}>
      <div className="status-icons">
        <div className={`icon checkmark ${recycleDetected ? 'active' : ''}`}>
          ✓
        </div>
        <div className={`icon x-mark ${nonRecycleDetected ? 'active' : ''}`}>
          ✕
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
      <div className={`detection-status ${isTrashDetected ? 'bottle-detected' : ''} ${isFrogDetected ? 'frog-detected' : ''} ${isLemonDetected ? 'lemon-detected' : ''}`}>
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
                className={`
                  ${isRecyclable(obj.class) ? 'recyclable-object' : ''}
                  ${isNonRecyclable(obj.class) ? 'non-recyclable-object' : ''}
                  ${isWrapper(obj.class) ? 'wrapper-object' : ''}
                  ${isPotentialLemon(obj.class) ? 'lemon-object' : ''}
                  ${isToyFrog(obj.class) ? 'frog-object' : ''}
                `}
              >
                {isPotentialLemon(obj.class) ? 'Lemon' : obj.class}: {obj.confidence}% confidence
                {isRecyclable(obj.class) && ' (RECYCLABLE)'}
                {isNonRecyclable(obj.class) && ' (NON-RECYCLABLE)'}
                {isWrapper(obj.class) && ' (RECYCLABLE WRAPPER)'}
                {isPotentialLemon(obj.class) && ' (LEMON)'}
                {isToyFrog(obj.class) && ' (TOY FROG)'}
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