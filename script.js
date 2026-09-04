import { MnistData } from './data.js';

console.log("TensorFlow.js loaded!");

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let drawing = false;
canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mousemove", draw);

function startDrawing(event) {
    drawing = true;
    draw(event);
}

function stopDrawing() {
    drawing = false;
    ctx.beginPath();
}

function draw(event) {

    if (!drawing) return;

    ctx.lineWidth = 20;
    ctx.lineCap = "round";
    ctx.strokeStyle = "white";

    const rect = canvas.getBoundingClientRect();

    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(x, y);
}
function createModel() {

    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu'
    }));

    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));

    model.add(tf.layers.flatten());

    model.add(tf.layers.dense({
        units: 10,
        activation: 'softmax'
    }));

    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}
async function trainModel(model, data) {

    const BATCH_SIZE = 128;
    const TRAIN_BATCHES = 100;

    for (let i = 0; i < TRAIN_BATCHES; i++) {

        const batch = data.nextTrainBatch(BATCH_SIZE);

        await model.fit(batch.xs, batch.labels, {
            batchSize: BATCH_SIZE,
            epochs: 1
        });

        batch.xs.dispose();
        batch.labels.dispose();

        console.log(`Training batch ${i + 1}/${TRAIN_BATCHES}`);
    }

    console.log("Training complete!");
}
async function startTraining() {

    console.log("Loading MNIST...");

    const data = new MnistData();

    await data.load();

    console.log("Creating model...");

    const model = createModel();

    console.log("Starting training...");

    await trainModel(model, data);

    console.log("Training finished!");

    window.model = model;
}