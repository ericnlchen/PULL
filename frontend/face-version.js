import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

// Global settings:
const DIST_TO_WEIGHT_RATIO = 50;
const NUM_NODES = 1000;
const ALPHA_DECAY = 0;      // 1 - Math.pow(0.001, 1 / 300) default
const VELOCITY_DECAY = 0.01; // 0.4 default
const SPRING_STIFFNESS = 1;
const SPRING_DAMPING = 0.02;

let IMAGE_SIZE = 150;

async function loadVideo() {
    const videoElement = document.getElementById('webcam-video');

    // Create a canvas element
    const buffer = document.createElement("canvas");
    const bufferContainer = document.getElementById("buffer-container");
    bufferContainer.appendChild(buffer);

    // Set canvas dimensions to match the image
    buffer.width = IMAGE_SIZE;
    buffer.height = IMAGE_SIZE;

    // Load video
    videoElement.srcObject = await navigator.mediaDevices.getUserMedia({ video: true });

    // Wait for a few seconds to let it initialize
    await new Promise(resolve => setTimeout(resolve, 1000));

    return [videoElement, buffer];
}

// Load video
const [videoElement, buffer] = await loadVideo();

// Get height and width
let videoWidth = videoElement.offsetWidth;
let videoHeight = videoElement.offsetHeight;

videoElement.style.display = 'none';

// Draw video to buffer
const buffCtx = buffer.getContext("2d");
buffCtx.drawImage(videoElement, 0, 0, IMAGE_SIZE, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
buffCtx.translate(buffer.width, 0);
buffCtx.scale(-1,1);

// Wait for first frame to be encoded
let latentZ = await encodeImage(buffer);

// Set up canvas for the resulting image
const canvas = document.createElement("canvas");
const context = canvas.getContext("2d", {willReadFrequently: true});
const container = document.getElementById("canvas-container");
container.appendChild(canvas)
canvas.width = IMAGE_SIZE;
canvas.height = IMAGE_SIZE;

async function encodeImage(canvas) {
    try {
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));

        // Create FormData and append the Blob
        const formData = new FormData();
        formData.append('image', blob, 'imageToEncode.png');
        const response = await fetch('http://127.0.0.1:8000/api/encode-face', {
            method: "POST",
            body: formData,
        });
        const result = await response.json();
        const encodedArray = result.encoded_array;
        return encodedArray[0];
    } catch (error) {
        console.error('Error fetching data:', error);
    }
}

// Update the canvas pixels with link lengths (called every 0.05 seconds for 30hz)
async function decodeEdgeWeights() {
    // Get new latent z from lengths of edges
    let newLatentZ = latentZ;
    // Decode latent z to image
    try {
        // Send POST
        const response = await fetch('http://127.0.0.1:8000/api/decode-face', {
            method: "POST",
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ array: newLatentZ }),
        });
        const blob = await response.blob();
        const imgURL = URL.createObjectURL(blob);
        const decodedImg = new Image();
        decodedImg.src = imgURL;
        await new Promise((resolve) => (decodedImg.onload = resolve));
        context.drawImage(decodedImg, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
    } catch (error) {
        console.error('Error fetching data:', error);
    }
}

async function periodicDecodeEdgeWeights(fps) {
    let s = 1 / fps;
    let ms = s * 1000;
    while (true) {
        await new Promise(resolve => setTimeout(resolve, ms));
        await decodeEdgeWeights();
    }
}

let fpsInterval, startTime, now, then, elapsed;

function startAnimating(fps) {
    fpsInterval = 1000 / fps;
    then = Date.now();
    startTime = then;
    animate();
}

async function animate() {
    requestAnimationFrame(animate);

    now = Date.now();
    elapsed = now - then;

    if (elapsed > fpsInterval) {
        then = now - (elapsed % fpsInterval);

        // Draw video feed to canvas
        buffCtx.drawImage(
            videoElement,
            (videoWidth - videoHeight) / 2, 0,
            videoHeight, videoHeight,
            0, 0,
            IMAGE_SIZE, IMAGE_SIZE
        );
        latentZ = await encodeImage(buffer);

    }
}

startAnimating(10); // fps
periodicDecodeEdgeWeights(10); // fps