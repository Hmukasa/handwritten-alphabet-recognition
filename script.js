"use strict";

const MODEL_URL = "models/resnet_emnist_letters.onnx";
const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");

const predictButton = document.getElementById("predictButton");
const clearButton = document.getElementById("clearButton");

const predictionEl = document.getElementById("prediction");
const confidenceEl = document.getElementById("confidence");
const statusEl = document.getElementById("status");
const top5El = document.getElementById("top5");
const probabilityGrid = document.getElementById("probabilityGrid");

let session = null;
let drawing = false;
let hasInk = false;
let isPredicting = false;

ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";


function setupCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "white";
  ctx.lineWidth = 22;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}


function getPointerPosition(event) {
  const rect = canvas.getBoundingClientRect();

  return {
    x: (event.clientX - rect.left) * canvas.width / rect.width,
    y: (event.clientY - rect.top) * canvas.height / rect.height
  };
}


function startDrawing(event) {
  event.preventDefault();

  drawing = true;
  hasInk = true;

  if (canvas.setPointerCapture) {
    canvas.setPointerCapture(event.pointerId);
  }

  const position = getPointerPosition(event);

  ctx.beginPath();
  ctx.moveTo(position.x, position.y);
  ctx.lineTo(position.x + 0.01, position.y + 0.01);
  ctx.stroke();

  updatePredictButton();
}


function draw(event) {
  if (!drawing) {
    return;
  }

  event.preventDefault();

  const position = getPointerPosition(event);

  ctx.lineTo(position.x, position.y);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(position.x, position.y);
}


function stopDrawing(event) {
  if (!drawing) {
    return;
  }

  event.preventDefault();

  drawing = false;
  ctx.beginPath();
}


function updatePredictButton() {
  predictButton.disabled =
    !session || !hasInk || isPredicting;
}


function setStatus(message, type = "") {
  statusEl.textContent = message;
  statusEl.className = type ? `status ${type}` : "status";
}


function percent(probability) {
  return `${(probability * 100).toFixed(1)}%`;
}


function makeBarRow(letter, probability, className) {
  const row = document.createElement("div");
  row.className = className;

  const label = document.createElement("strong");
  label.textContent = letter;

  const track = document.createElement("div");
  track.className = "bar-track";

  const fill = document.createElement("div");
  fill.className = "bar-fill";
  fill.style.width = `${probability * 100}%`;

  const value = document.createElement("span");
  value.className = "percent";
  value.textContent = percent(probability);

  track.appendChild(fill);

  row.appendChild(label);
  row.appendChild(track);
  row.appendChild(value);

  return row;
}


function clearProbabilityGrid() {
  probabilityGrid.innerHTML = "";

  for (const letter of LETTERS) {
    probabilityGrid.appendChild(
      makeBarRow(letter, 0, "prob-row")
    );
  }
}


function clearCanvas() {
  setupCanvas();

  hasInk = false;

  predictionEl.textContent = "?";
  confidenceEl.textContent = "--%";
  top5El.innerHTML = "";

  clearProbabilityGrid();

  if (session) {
    setStatus("A〜Zを1文字書いてください", "ok");
  }

  updatePredictButton();
}


// 描画部分を切り出し、28×28の入力画像に変換する
function preprocessCanvas() {
  const width = canvas.width;
  const height = canvas.height;

  const image = ctx.getImageData(
    0,
    0,
    width,
    height
  );

  const data = image.data;

  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = (y * width + x) * 4;

      const gray =
        0.299 * data[index] +
        0.587 * data[index + 1] +
        0.114 * data[index + 2];

      if (gray > 20) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX < minX || maxY < minY) {
    return new Float32Array(28 * 28);
  }

  const cropWidth = maxX - minX + 1;
  const cropHeight = maxY - minY + 1;

  const scale = Math.min(
    20 / cropWidth,
    20 / cropHeight
  );

  const newWidth = Math.max(
    1,
    Math.round(cropWidth * scale)
  );

  const newHeight = Math.max(
    1,
    Math.round(cropHeight * scale)
  );

  const resizedCanvas = document.createElement("canvas");
  resizedCanvas.width = 28;
  resizedCanvas.height = 28;

  const resizedCtx = resizedCanvas.getContext("2d");

  resizedCtx.fillStyle = "black";
  resizedCtx.fillRect(0, 0, 28, 28);
  resizedCtx.imageSmoothingEnabled = true;

  const left = Math.floor((28 - newWidth) / 2);
  const top = Math.floor((28 - newHeight) / 2);

  resizedCtx.drawImage(
    canvas,
    minX,
    minY,
    cropWidth,
    cropHeight,
    left,
    top,
    newWidth,
    newHeight
  );

  const pixels = resizedCtx.getImageData(
    0,
    0,
    28,
    28
  ).data;

  const input = new Float32Array(28 * 28);

  for (let i = 0; i < input.length; i++) {
    const r = pixels[i * 4];
    const g = pixels[i * 4 + 1];
    const b = pixels[i * 4 + 2];

    const gray =
      (0.299 * r + 0.587 * g + 0.114 * b) / 255;

    input[i] = (gray - 0.5) / 0.5;
  }

  return input;
}


function softmax(values) {
  const maxValue = Math.max(...values);

  const expValues = values.map(
    value => Math.exp(value - maxValue)
  );

  const sum = expValues.reduce(
    (total, value) => total + value,
    0
  );

  return expValues.map(
    value => value / sum
  );
}


function showResults(probabilities) {
  const ranked = probabilities
    .map((probability, index) => ({
      letter: LETTERS[index],
      probability
    }))
    .sort(
      (a, b) => b.probability - a.probability
    );

  predictionEl.textContent = ranked[0].letter;
  confidenceEl.textContent =
    percent(ranked[0].probability);

  top5El.innerHTML = "";

  for (const item of ranked.slice(0, 5)) {
    top5El.appendChild(
      makeBarRow(
        item.letter,
        item.probability,
        "top-row"
      )
    );
  }

  probabilityGrid.innerHTML = "";

  for (let i = 0; i < LETTERS.length; i++) {
    probabilityGrid.appendChild(
      makeBarRow(
        LETTERS[i],
        probabilities[i],
        "prob-row"
      )
    );
  }
}


async function loadModel() {
  try {
    setStatus("ResNetを読み込み中...");

    session = await ort.InferenceSession.create(
      MODEL_URL,
      {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all"
      }
    );

    setStatus(
      "A〜Zを1文字書いてください",
      "ok"
    );

  } catch (error) {
    console.error(error);

    session = null;

    setStatus(
      "モデルを読み込めません。models/resnet_emnist_letters.onnx を確認してください。",
      "error"
    );
  }

  updatePredictButton();
}


async function predict() {
  if (!session || !hasInk || isPredicting) {
    return;
  }

  isPredicting = true;
  updatePredictButton();

  setStatus("ResNet が認識中...");

  try {
    const input = preprocessCanvas();

    const tensor = new ort.Tensor(
      "float32",
      input,
      [1, 1, 28, 28]
    );

    const results = await session.run({
      [session.inputNames[0]]: tensor
    });

    const logits = Array.from(
      results[session.outputNames[0]].data
    );

    const probabilities = softmax(logits);

    showResults(probabilities);

    setStatus(
      "ResNet による予測",
      "ok"
    );

  } catch (error) {
    console.error(error);

    setStatus(
      `認識に失敗しました: ${error.message}`,
      "error"
    );

  } finally {
    isPredicting = false;
    updatePredictButton();
  }
}


canvas.addEventListener(
  "pointerdown",
  startDrawing
);

canvas.addEventListener(
  "pointermove",
  draw
);

canvas.addEventListener(
  "pointerup",
  stopDrawing
);

canvas.addEventListener(
  "pointercancel",
  stopDrawing
);

predictButton.addEventListener(
  "click",
  predict
);

clearButton.addEventListener(
  "click",
  clearCanvas
);


setupCanvas();
clearProbabilityGrid();
loadModel();
