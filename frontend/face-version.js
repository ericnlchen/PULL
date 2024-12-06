import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

// Global settings:
const IMAGE_SIZE = 150;
const LATENT_BAR_MAX = 5;
const LATENT_BAR_MIN = -5;

const PARAM_BAR_MIN = -10;
const PARAM_BAR_MAX = 10;

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

function barChart(data, containerId, yMin=LATENT_BAR_MIN, yMax=LATENT_BAR_MAX, interactive=false, color="white") {
    // D3 Bar Chart
    const width = 1200;
    const height = 200;
    const margin = { top: 10, right: 10, bottom: 10, left: 10 };
    const svg = d3.select(containerId)
        .append("svg")
        .attr("width", width)
        .attr("height", height)

    // Step 4: Define scales
    const xScale = d3.scaleBand()
        .domain(data.map((_, i) => i)) // Map each data index to a bar
        .range([margin.left, width - margin.right])
        .padding(0.1); // Add space between bars

    const yScale = d3.scaleLinear()
        .domain([yMin, yMax]) // Account for both positive and negative values
        .range([height - margin.bottom, margin.top]);

    // Step 5: Draw bars
    const bars = svg.selectAll(".bar")
        .data(data)
        .enter()
        .append("rect")
        .attr("class", "bar")
        .attr("x", (_, i) => xScale(i)) // Set x-position using the index
        .attr("y", d => (d >= 0 ? yScale(d) : yScale(0))) // Bars start at yScale(0) for negative values
        .attr("width", xScale.bandwidth()) // Bar width
        .attr("height", d => Math.abs(yScale(d) - yScale(0))) // Height depends on the distance from yScale(0)
        .attr("fill", color);

    if (interactive) {
        let isTracing = false;

        svg.on("mousedown", function (event) {
            isTracing = true; // Start tracing
            const [mouseX, mouseY] = d3.pointer(event);
            manipulateBars(mouseX, mouseY);
        })
        svg.on("mousemove", function (event) {
            if (isTracing) {
                const [mouseX, mouseY] = d3.pointer(event);
                // Update bar heights dynamically as the user moves the mouse
                manipulateBars(mouseX, mouseY);
            }
        });
        svg.on("mouseup", function () {
            isTracing = false; // End tracing
        });
        svg.on("mouseleave", function() {
            isTracing = false;
        })
    
        function manipulateBars(mouseX, mouseY) {
            const mouseValue = yScale.invert(mouseY); // Map mouse Y position to the data value
            // Adjust each bar's height if the mouse is within its x-domain
            bars.each(function (d, i) {
                const barXStart = xScale(i);
                const barXEnd = barXStart + xScale.bandwidth();
        
                // Check if the mouse is over the bar
                if (mouseX >= barXStart && mouseX <= barXEnd) {
                    // Update the corresponding data value
                    data[i] = Math.max(yMin, Math.min(yMax, mouseValue));
                
                    // Update the bar's position and height
                    d3.select(this)
                        .attr("y", () => (data[i] >= 0 ? yScale(data[i]) : yScale(0)))
                        .attr("height", () => Math.abs(yScale(data[i]) - yScale(0)));
                }
            });
        }
    }

    return [svg, yScale];
}

let params = new Array(latentZ.length).fill(1)

let [latentBarSvg, latentBarYScale] = barChart(latentZ, '#d3-container-0');
let [paramBarSvg, paramBarYScale] = barChart(params, '#d3-container-1', PARAM_BAR_MIN, PARAM_BAR_MAX, true, '#2bff69');

function updateChart(svg, yScale, newData) {
    // Bind the new data to the bars
    const bars = svg.selectAll(".bar")
      .data(newData);

    // Update bars
    bars
      .attr("y", d => (d >= 0 ? yScale(d) : yScale(0)))
      .attr("height", d => Math.abs(yScale(d) - yScale(0)))
}

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
    let newLatentZ = latentZ.map((elt, index) => elt * params[index]);
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
        updateChart(latentBarSvg, latentBarYScale, latentZ);
    }
}

const resetButton = document.getElementById("reset-button");
resetButton.addEventListener('click', () => {
    params.fill(1);
    updateChart(paramBarSvg, paramBarYScale, params);
})

startAnimating(10); // fps
periodicDecodeEdgeWeights(10); // fps