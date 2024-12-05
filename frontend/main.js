import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

// Global settings:
const DIST_TO_WEIGHT_RATIO = 50;
const NUM_NODES = 800;
const ALPHA_DECAY = 0;      // 1 - Math.pow(0.001, 1 / 300) default
const VELOCITY_DECAY = 0.02; // 0.4 default
const SPRING_STIFFNESS = 2;
const SPRING_DAMPING = 0.01;

async function loadVideo() {
    const videoElement = document.getElementById('webcam-video');

    // Create a canvas element
    const buffer = document.createElement("canvas");
    const bufferContainer = document.getElementById("buffer-container");
    bufferContainer.appendChild(buffer);

    // Set canvas dimensions to match the image
    buffer.width = 64;
    buffer.height = 64;

    // Load video
    videoElement.srcObject = await navigator.mediaDevices.getUserMedia({ video: true });

    // Wait for a few seconds to let it initialize
    await new Promise(resolve => setTimeout(resolve, 2000));

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
buffCtx.drawImage(videoElement, 0, 0, 64, 64, 0, 0, 64, 64);
buffCtx.translate(buffer.width, 0);
buffCtx.scale(-1,1);

// Wait for first frame to be encoded
const latentZ = await encodeImage(buffer);

console.log(latentZ)

// Set up the simulation
const k = latentZ.length;
const graph = generateRandomGraph(NUM_NODES, k, latentZ);
let nodes = graph.nodes;
let links = graph.links;

// Set up canvas for the resulting image
const canvas = document.createElement("canvas");
const context = canvas.getContext("2d", {willReadFrequently: true});
const container = document.getElementById("canvas-container");
container.appendChild(canvas)
canvas.width = 64;
canvas.height = 64;

// Set up the SVG canvas dimensions
const width = 600;
const height = 500;
const svg = d3.select("#d3-container").append("svg")
    .attr("width", width)
    .attr("height", height);

// // Add pointer node
// nodes.push({id: 'pointer'});

// A custom spring force
function springForce(links, stiffness, damping) {
    return function (alpha) {
      links.forEach(link => {
        // Get source and target nodes
        const source = link.source;
        const target = link.target;

        const dx = target.x - source.x;
        const dy = target.y - source.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const displacement = distance - link.weight * DIST_TO_WEIGHT_RATIO;
  
        // Normalize direction
        const nx = dx / distance;
        const ny = dy / distance;
  
        // Compute spring force magnitude
        const springForce = -stiffness * displacement;
  
        // Apply damping based on relative velocity
        const relativeVelocityX = target.vx - source.vx;
        const relativeVelocityY = target.vy - source.vy;
        const dampingForce = -damping * (relativeVelocityX * nx + relativeVelocityY * ny);
  
        const force = -0.1*(springForce + dampingForce);
  
        // Apply forces to source and target
        source.vx += force * nx;
        source.vy += force * ny;
        target.vx -= force * nx;
        target.vy -= force * ny;
      });
    };
  }
  

// Define the simulation
const simulation = d3.forceSimulation(nodes)
    // .force("link", d3.forceLink(links).id((d) => d.id).distance(link => link.weight * DIST_TO_WEIGHT_RATIO))
    .force("spring", springForce(links, SPRING_STIFFNESS, SPRING_DAMPING))
    .force("charge", d3.forceManyBody().strength(-0.5))
    // .force("charge", d3.forceManyBody().strength((n) => n.id === 'pointer' ? 50 : 0 ))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .on("tick", ticked)
    .alphaDecay(ALPHA_DECAY)
    .velocityDecay(VELOCITY_DECAY);

// Draw the links
const link = svg.selectAll(".link")
    .data(links)
    .enter().append("line")
    .attr("class", "link")
    .attr("stroke-width", 0.55) // d => d.weight / 3
    .attr("stroke", "grey");

// Draw the nodes
const node = svg.selectAll(".node")
    .data(nodes.filter(d => d.id !== 'pointer'))
    .enter().append("circle")
    .attr("class", "node")
    .attr("r", 4)
    .attr("fill", "#E0E0E0")
    .call(drag(simulation));

// d3.select("#d3-container")
//     .on("pointermove", pointermoved);

// function pointermoved(event) {
//     const [x, y] = d3.pointer(event);
//     nodes[nodes.length - 1].fx = x;
//     nodes[nodes.length - 1].fy = y;
// }

// Update positions on each tick
function ticked() {
    link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);
    node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);
}

function generateRandomGraph(n, k, weights) {
    if (k > n * (n - 1) / 2) {
        throw new Error("Too many edges for the number of nodes");
    }
    if (k < n - 1) {
        throw new Error("Too many nodes for the number of edges");
    }
    if (weights.length !== k) {
        throw new Error("Length of weights array must be k");
    }
    // Create nodes
    const nodes = Array.from({ length: n }, (_, i) => ({ id: i }));
    let weightIndex = 0;

    // Create a minimal connected structure (tree) to ensure connectivity
    const links = [];
    for (let i = 1; i < n; i++) {
        const source = i;
        const target = Math.floor(Math.random() * i); // Connect to a random earlier node
        links.push({
            "source": nodes[source],
            "target": nodes[target],
            "weight": weights[weightIndex],
        });
        weightIndex++;
    }
    // Add random edges to reach exactly k edges
    const linkSet = new Set(links.map(e => `${e.source}-${e.target}`));
    while (links.length < k) {
        const source = Math.floor(Math.random() * n);
        let target = Math.floor(Math.random() * n);
        while (source === target || linkSet.has(`${source}-${target}`) || linkSet.has(`${target}-${source}`)) {
            target = Math.floor(Math.random() * n);
        }
        links.push({
            "source": nodes[source],
            "target": nodes[target],
            "weight": weights[weightIndex]
        });
        weightIndex++;
        linkSet.add(`${source}-${target}`);
    }
    return {nodes, links};
}

async function encodeImage(canvas) {
    try {
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));

        // Create FormData and append the Blob
        const formData = new FormData();
        formData.append('image', blob, 'imageToEncode.png');
        const response = await fetch('http://127.0.0.1:8000/api/encode', {
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
    let newLatentZ = links.map((link) => (
        (1 / DIST_TO_WEIGHT_RATIO) *
        Math.sqrt(
            Math.pow(link.target.x - link.source.x, 2) +
            Math.pow(link.target.y - link.source.y, 2)
        )
    ));
    // Decode latent z to image
    try {
        // Send POST
        const response = await fetch('http://127.0.0.1:8000/api/decode', {
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
        context.drawImage(decodedImg, 0, 0, 64, 64);

        // let imageData = context.getImageData(0, 0, canvas.width, canvas.height);
        // let data = imageData.data;
        // context.putImageData(data)
    } catch (error) {
        console.error('Error fetching data:', error);
    }
}

async function periodicDecodeEdgeWeights() {
    while (true) {
        await new Promise(resolve => setTimeout(resolve, 50));
        await decodeEdgeWeights();
    }
}

// Dragging functionality
function drag(simulation) {
    return d3.drag()
        .on("start", event => {
            // if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        })
        .on("drag", event => {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        })
        .on("end", event => {
            // if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        });
}

let start;
async function loop(timestamp) {
    if (start === undefined) {
        start = timestamp;
    }
    const elapsed = timestamp - start; // ms

    buffCtx.drawImage(
        videoElement,
        (videoWidth - videoHeight) / 2, 0,
        videoHeight, videoHeight,
        0, 0,
        64, 64
    );
    const latentZ = await encodeImage(buffer);

    // Dynamically update edges of the graph
    // update link weights
    links.forEach((link, i) => {
        link.weight = latentZ[i];
    })
    // simulation.force("link").distance(link => link.weight * DIST_TO_WEIGHT_RATIO);

    requestAnimationFrame(loop);
}

requestAnimationFrame(loop);
periodicDecodeEdgeWeights();