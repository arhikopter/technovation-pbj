# L.I.F.T - Recycling Detector

A real-time recycling detection application that uses TensorFlow.js and computer vision to identify recyclable items through your device's camera.

## Features

- **Real-time Object Detection**: Uses TensorFlow.js and COCO-SSD model to detect objects in real-time
- **Recycling Classification**: Automatically classifies detected objects as recyclable or non-recyclable
- **High Performance**: Optimized for real-time operation on both mobile and desktop devices
- **Clean Interface**: Modern iOS-inspired interface with real-time feedback
- **Performance Monitoring**: Shows FPS counter to monitor performance

## How It Works

L.I.F.T (Litter Identification For Tomorrow) uses a pre-trained machine learning model to detect objects in the camera view. It then classifies these objects as recyclable or non-recyclable based on a predefined list of recyclable materials. The application runs entirely in the browser, with no server-side processing needed.

### Recyclable Items

The application can identify common recyclable items, including:
- Bottles
- Cups
- Paper
- Cardboard
- Metal cans
- Containers
- And more

## Technologies Used

- React (JavaScript/TypeScript)
- TensorFlow.js
- COCO-SSD Object Detection model
- HTML5 Camera API
- CSS3 for modern UI

## Getting Started

### Prerequisites

- Modern web browser (Chrome, Firefox, Safari, Edge)
- Camera access
- JavaScript enabled

### Local Development

1. Clone the repository
2. Install dependencies: `npm install`
3. Start the development server: `npm start`
4. Open http://localhost:3000 in your browser

### Building for Production

To create a production build:

```
npm run build
```

## Usage

1. Open the application in your browser
2. Allow camera access when prompted
3. Point your device's camera at objects you want to identify
4. The application will identify objects and classify them as recyclable or non-recyclable

## Privacy

The application processes all images locally in your browser. No images or video streams are sent to any server.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
