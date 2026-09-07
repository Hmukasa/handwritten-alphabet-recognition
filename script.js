"use strict";

const MODEL_URL = "models/resnet_emnist_letters.onnx";
const LETTERS = Array.from("ABCDEFGHIJKLMNOPQRSTUVWXYZ");
const REALTIME_DELAY_MS = 350;

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
let predictTimer = null;
let isPredicting = false;

// ONNX Runtime WebのWASM本体もCDNから取得する。
if (typeof ort !== "undefined") {
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
}

function setupCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "white";
  ctx.lineWidth = 22;
}

function pointerPosition(event) {
  const rect = canvas.getBoundingClientRect();

  return {
    x: (event.clientX - rect.left) * (canvas.width / rect.width),
    y: (event.clientY - rect.top) * (canvas.height / rect.height)
  };
}

function startDrawing(event) {
  event.preventDefault();

  drawing = true;
  hasInk = true;

  if (canvas.setPointerCapture) {
    canvas.setPointerCapture(event.pointerId);
  }

  const p = pointerPosition(event);

  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
  ctx.lineTo(p.x + 0.01, p.y + 0.01);
  ctx.stroke();
}

function draw(event) {
  if (!drawing) return;

  event.preventDefault();

  const p = pointerPosition(event);

  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);

  clearTimeout(predictTimer);

  if (session) {
    predictTimer = setTimeout(() => {
      predict();
    }, REALTIME_DELAY_MS);
  }
}

function stopDrawing(event) {
  if (!drawing) return;

  event.preventDefault();
  drawing = false;
  ctx.beginPath();

  clearTimeout(predictTimer);

  if (hasInk && session) {
    predictTimer = setTimeout(() => {
      predict();
    }, 100);
  }
}

function percent(p) {
  return `${(p * 100).toFixed(1)}%`;
}

function makeBarRow(letter, probability, rowClass = "prob-row") {
  const row = document.createElement("div");
  row.className = rowClass;

  const label = document.createElement("strong");
  label.textContent = letter;

  const track = document.createElement("div");
  track.className = "bar-track";

  const fill = document.createElement("div");
  fill.className = "bar-fill";
  fill.style.width =
    `${Math.max(0, Math.min(100, probability * 100))}%`;

  const value = document.createElement("span");
  value.className = "percent";
  value.textContent = percent(probability);

  track.appendChild(fill);
  row.appendChild(label);
  row.appendChild(track);
  row.appendChild(value);

  return row;
}

function createEmptyProbabilityGrid() {
  probabilityGrid.innerHTML = "";

  for (const letter of LETTERS) {
    probabilityGrid.appendChild(
      makeBarRow(letter, 0)
    );
  }
}

function clearCanvas() {
  clearTimeout(predictTimer);

  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  hasInk = false;

  predictionEl.textContent = "?";
  confidenceEl.textContent = "--%";
  top5El.innerHTML = "";
  createEmptyProbabilityGrid();

  if (session) {
    setStatus("文字を書いてください", "ok");
  }
}

function setStatus(message, type = "") {
  statusEl.textContent = message;
  statusEl.className = type ? `status ${type}` : "status";
}

/*
  Canvas -> EMNIST Letters用 1x1x28x28 Float32

  Python版アプリと同じ考え方:
  1. 黒背景から白い描画領域のbounding boxを検出
  2. アスペクト比を保ったまま最大20x20へ縮小
  3. 28x28黒背景の中央へ配置
  4. [0,1] -> [-1,1] に正規化

  ブラウザで書いた文字は既に正立しているため、
  EMNIST生データ読み込み時のrotate/flip処理は適用しない。
*/
function preprocessCanvas() {
  const W = canvas.width;
  const H = canvas.height;
  const image = ctx.getImageData(0, 0, W, H);
  const data = image.data;

  let minX = W;
  let minY = H;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;

      const gray =
        0.299 * data[i] +
        0.587 * data[i + 1] +
        0.114 * data[i + 2];

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

  const cropW = maxX - minX + 1;
  const cropH = maxY - minY + 1;
  const scale = Math.min(20 / cropW, 20 / cropH);

  const newW = Math.max(1, Math.round(cropW * scale));
  const newH = Math.max(1, Math.round(cropH * scale));

  const off = document.createElement("canvas");
  off.width = 28;
  off.height = 28;

  const offCtx = off.getContext("2d");
  offCtx.fillStyle = "black";
  offCtx.fillRect(0, 0, 28, 28);
  offCtx.imageSmoothingEnabled = true;
  offCtx.imageSmoothingQuality = "high";

  const left = Math.floor((28 - newW) / 2);
  const top = Math.floor((28 - newH) / 2);

  offCtx.drawImage(
    canvas,
    minX, minY, cropW, cropH,
    left, top, newW, newH
  );

  const pixels =
    offCtx.getImageData(0, 0, 28, 28).data;

  const input = new Float32Array(28 * 28);

  for (let i = 0; i < 28 * 28; i++) {
    const r = pixels[i * 4];
    const g = pixels[i * 4 + 1];
    const b = pixels[i * 4 + 2];

    const gray =
      (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;

    input[i] = (gray - 0.5) / 0.5;
  }

  return input;
}

function softmax(logits) {
  const maxValue = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - maxValue));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

async function loadModel() {
  try {
    if (typeof ort === "undefined") {
      throw new Error("ONNX Runtime Webを読み込めませんでした。");
    }

    setStatus("ResNetを読み込み中...");

    session = await ort.InferenceSession.create(
      MODEL_URL,
      {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all"
      }
    );

    predictButton.disabled = false;
    setStatus("A〜Zを1文字書いてください", "ok");
  } catch (error) {
    console.error(error);
    session = null;
    predictButton.disabled = true;

    setStatus(
      "モデルを読み込めません。models/resnet_emnist_letters.onnx を確認してください。",
      "error"
    );
  }
}

async function predict() {
  if (!session || !hasInk || isPredicting) return;

  isPredicting = true;
  predictButton.disabled = true;
  setStatus("ResNet が認識中...");

  try {
    const inputArray = preprocessCanvas();

    const inputTensor =
      new ort.Tensor("float32", inputArray, [1, 1, 28, 28]);

    const inputName = session.inputNames[0];

    const results = await session.run({
      [inputName]: inputTensor
    });

    const outputName = session.outputNames[0];
    const logits = Array.from(results[outputName].data);
    const probs = softmax(logits);

    const ranked = probs
      .map((probability, index) => ({
        letter: LETTERS[index],
        probability
      }))
      .sort((a, b) => b.probability - a.probability);

    predictionEl.textContent = ranked[0].letter;
    confidenceEl.textContent = percent(ranked[0].probability);

    setStatus("ResNet による予測", "ok");

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
          probs[i],
          "prob-row"
        )
      );
    }
  } catch (error) {
    console.error(error);
    setStatus(`認識に失敗しました: ${error.message}`, "error");
  } finally {
    isPredicting = false;
    predictButton.disabled = !session;
  }
}

canvas.addEventListener("pointerdown", startDrawing);
canvas.addEventListener("pointermove", draw);
canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);

predictButton.addEventListener("click", predict);
clearButton.addEventListener("click", clearCanvas);

setupCanvas();
createEmptyProbabilityGrid();
loadModel();
