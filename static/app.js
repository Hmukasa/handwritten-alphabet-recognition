const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");

const predictButton = document.getElementById("predictButton");
const clearButton = document.getElementById("clearButton");
const predictionEl = document.getElementById("prediction");
const confidenceEl = document.getElementById("confidence");
const statusEl = document.getElementById("status");
const top5El = document.getElementById("top5");
const probabilityGrid = document.getElementById("probabilityGrid");

let drawing = false;
let hasInk = false;
let predictTimer = null;

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

  canvas.setPointerCapture(event.pointerId);

  const p = pointerPosition(event);
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
}

function draw(event) {
  if (!drawing) return;

  event.preventDefault();

  const p = pointerPosition(event);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
}

function stopDrawing(event) {
  if (!drawing) return;

  event.preventDefault();
  drawing = false;
  ctx.closePath();

  clearTimeout(predictTimer);

  predictTimer = setTimeout(() => {
    if (hasInk) predict();
  }, 350);
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
  fill.style.width = `${Math.max(0, Math.min(100, probability * 100))}%`;

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

  for (const letter of "ABCDEFGHIJKLMNOPQRSTUVWXYZ") {
    probabilityGrid.appendChild(makeBarRow(letter, 0));
  }
}

function clearCanvas() {
  clearTimeout(predictTimer);

  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  hasInk = false;

  predictionEl.textContent = "?";
  confidenceEl.textContent = "--%";
  statusEl.textContent = "文字を書いてください";
  top5El.innerHTML = "";

  createEmptyProbabilityGrid();
}

async function predict() {
  if (!hasInk) {
    statusEl.textContent = "まず文字を書いてください";
    return;
  }

  predictButton.disabled = true;
  statusEl.textContent = "AI が認識中...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        image: canvas.toDataURL("image/png")
      })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data.error || "認識に失敗しました。"
      );
    }

    predictionEl.textContent = data.prediction;
    confidenceEl.textContent = percent(data.confidence);

    const modelName =
      data.model && data.model.model_type
        ? data.model.model_type
        : "AI";

    statusEl.textContent = `${modelName} による予測`;

    top5El.innerHTML = "";

    for (const item of data.top5) {
      top5El.appendChild(
        makeBarRow(
          item.letter,
          item.probability,
          "top-row"
        )
      );
    }

    probabilityGrid.innerHTML = "";

    for (const item of data.probabilities) {
      probabilityGrid.appendChild(
        makeBarRow(
          item.letter,
          item.probability
        )
      );
    }

  } catch (error) {
    statusEl.textContent = error.message;
  } finally {
    predictButton.disabled = false;
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
