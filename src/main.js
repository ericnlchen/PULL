import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

// Global settings:
const breakingThreshold = 150;

// Load image data into graph data structure
const img = new Image();
img.src = "data/dogtest-small.png";

img.onload = () => {
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

    // Get the pixel data
    const imageData = buffCtx.getImageData(0, 0, buffer.width, buffer.height);
    const pixels = imageData.data;

    let nodes = [];
    for (let i = 0; i < img.width; i++) {
        nodes.push({"id": `${i}`});
    }

    let links = [];

    // Access pixel values
    for (let i = 0; i < pixels.length; i += 4) {
        const red = pixels[i];
        const green = pixels[i + 1];
        const blue = pixels[i + 2];
        let grey = 0.299 * red + 0.587 * green + 0.114 * blue;

        let pixelIndex = i / 4;
        let rowNum = Math.floor(pixelIndex / img.width);
        let colNum = pixelIndex % img.width;

        if (rowNum === colNum) {
            continue;
        }

        let distance = grey;

        if (distance > breakingThreshold) {
            continue;
        }

        links.push({
            "source": nodes[colNum],
            "target": nodes[rowNum],
            "weight": distance
        });
    }

    // Set up canvas for the resulting image
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d", {willReadFrequently: true});
    const container = document.getElementById("canvas-container");
    container.appendChild(canvas)
    canvas.width = img.width;
    canvas.height = img.height;

    // Set up the SVG canvas dimensions
    const width = 500;
    const height = 500;
    const svg = d3.select("#d3-container").append("svg")
        .attr("width", width)
        .attr("height", height);

    // Define the simulation
    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).distance(link => link.weight))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .on("tick", ticked);

    // Draw the links
    const link = svg.selectAll(".link")
        .data(links)
        .enter().append("line")
        .attr("class", "link")
        .attr("stroke-width", d => d.weight / 2000)
        .attr("stroke", "gray");

    // Draw the nodes
    const node = svg.selectAll(".node")
        .data(nodes)
        .enter().append("circle")
        .attr("class", "node")
        .attr("r", 4)
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

        // Update the canvas pixels with link lengths
        let imageData = context.getImageData(0, 0, canvas.width, canvas.height);
        let data = imageData.data;
        links.forEach(link => {
            const length = Math.sqrt(
                Math.pow(link.target.x - link.source.x, 2) +
                Math.pow(link.target.y - link.source.y, 2)
              );
            let rowNum = parseInt(link.target.id);
            let colNum = parseInt(link.source.id);
            let dataIndex = 4 * (rowNum * canvas.width + colNum);
            let gray = length;
            data[dataIndex] = gray;
            data[dataIndex+1] = gray;
            data[dataIndex+2] = gray;
            data[dataIndex+3] = 255; // alpha
        })
        context.putImageData(imageData, 0, 0);
    }

    // Dragging functionality
    function drag(simulation) {
        return d3.drag()
            .on("start", event => {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            })
            .on("drag", event => {
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            })
            .on("end", event => {
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            });
    }

};
