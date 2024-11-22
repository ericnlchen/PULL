import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

// Global settings:
const DIST_TO_WEIGHT_RATIO = 50
const NUM_NODES = 200
const ALPHA_DECAY = 0      // 1 - Math.pow(0.001, 1 / 300) default
const VELOCITY_DECAY = 0.1 // 0.4 default

// Load image data into graph data structure
const img = new Image();
img.src = "./data/dogtest-64.png";

img.onload = async () => {
    // Get latent z
    const latentZ = await encodeImage(img);
    console.log(latentZ);

    // Create a canvas element
    const buffer = document.createElement("canvas");
    const bufferContainer = document.getElementById("buffer-container");
    bufferContainer.appendChild(buffer);
    const buffCtx = buffer.getContext("2d");

    // Set canvas dimensions to match the image
    buffer.width = img.width;
    buffer.height = img.height;

    // Draw the image on the buffer
    buffCtx.drawImage(img, 0, 0);

    // Set up graph from latents
    const k = latentZ.length;
    const graph = generateRandomGraph(NUM_NODES, k, latentZ);
    let nodes = graph.nodes;
    let links = graph.links;

    // Set up canvas for the resulting image
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d", {willReadFrequently: true});
    const container = document.getElementById("canvas-container");
    container.appendChild(canvas)
    canvas.width = img.width;
    canvas.height = img.height;

    // Set up the SVG canvas dimensions
    const width = 700;
    const height = 500;
    const svg = d3.select("#d3-container").append("svg")
        .attr("width", width)
        .attr("height", height);

    // Define the simulation
    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id((d) => d.id).distance(link => link.weight * DIST_TO_WEIGHT_RATIO))
        .force("charge", d3.forceManyBody().strength(0))
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
        .data(nodes)
        .enter().append("circle")
        .attr("class", "node")
        .attr("r", 4)
        .attr("fill", "#E0E0E0")
        .call(drag(simulation));

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
                "source": source,
                "target": target,
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
                "source": source,
                "target": target,
                "weight": weights[weightIndex]
            });
            weightIndex++;
            linkSet.add(`${source}-${target}`);
        }

        return {nodes, links};
    }

    async function encodeImage(img) {
        try {
            const imgResponse = await fetch(img.src);
            const blob = await imgResponse.blob();

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
            context.drawImage(decodedImg, 0, 0)

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


    periodicDecodeEdgeWeights();

};