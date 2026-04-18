import "./style.css";
import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";

const MP_VERSION = "0.10.14";
const WASM_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const INDEX_TIP = 8;
const THUMB_TIP = 4;

type PaintMode = "pinch" | "always";

function dist2D(
  a: { x: number; y: number },
  b: { x: number; y: number }
): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function mount(): void {
  const root = document.querySelector<HTMLDivElement>("#app");
  if (!root) return;

  root.innerHTML = `
    <h1>Touching Paint</h1>
    <div class="stage">
      <video id="cam" playsinline muted autoplay></video>
      <canvas id="layer"></canvas>
    </div>
    <div class="toolbar">
      <label><input type="radio" name="mode" value="pinch" checked /> Pinch to paint</label>
      <label><input type="radio" name="mode" value="always" /> Always paint</label>
      <label>Color <input type="color" id="color" value="#7c5cff" /></label>
      <label>Size <input type="range" id="size" min="2" max="48" value="14" /></label>
      <button type="button" id="clear">Clear</button>
    </div>
    <p class="status" id="status">Starting camera…</p>
  `;

  const video = root.querySelector<HTMLVideoElement>("#cam")!;
  const canvas = root.querySelector<HTMLCanvasElement>("#layer")!;
  const ctx = canvas.getContext("2d")!;
  const statusEl = root.querySelector<HTMLParagraphElement>("#status")!;
  const colorInput = root.querySelector<HTMLInputElement>("#color")!;
  const sizeInput = root.querySelector<HTMLInputElement>("#size")!;
  const clearBtn = root.querySelector<HTMLButtonElement>("#clear")!;

  let paintMode: PaintMode = "pinch";
  root.querySelectorAll<HTMLInputElement>('input[name="mode"]').forEach((r) => {
    r.addEventListener("change", () => {
      if (r.checked) paintMode = r.value as PaintMode;
    });
  });

  let lastX: number | null = null;
  let lastY: number | null = null;
  let handLandmarker: HandLandmarker | null = null;
  let raf = 0;

  function setStatus(text: string, isError = false): void {
    statusEl.textContent = text;
    statusEl.classList.toggle("error", isError);
  }

  function resizeCanvas(): void {
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return;
    if (canvas.width !== w || canvas.height !== h) {
      const prev = document.createElement("canvas");
      prev.width = canvas.width;
      prev.height = canvas.height;
      if (canvas.width && canvas.height) {
        const pctx = prev.getContext("2d");
        if (pctx) pctx.drawImage(canvas, 0, 0);
      }
      canvas.width = w;
      canvas.height = h;
      if (prev.width && prev.height) {
        ctx.drawImage(prev, 0, 0, prev.width, prev.height, 0, 0, w, h);
      }
    }
  }

  function toCanvasXY(nx: number, ny: number): [number, number] {
    const w = canvas.width;
    const h = canvas.height;
    return [nx * w, ny * h];
  }

  function strokeTo(x: number, y: number, draw: boolean): void {
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = colorInput.value;
    ctx.lineWidth = Number(sizeInput.value);
    if (draw && lastX != null && lastY != null) {
      ctx.beginPath();
      ctx.moveTo(lastX, lastY);
      ctx.lineTo(x, y);
      ctx.stroke();
    }
    if (draw) {
      lastX = x;
      lastY = y;
    } else {
      lastX = null;
      lastY = null;
    }
  }

  function resetStroke(): void {
    lastX = null;
    lastY = null;
  }

  clearBtn.addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    resetStroke();
  });

  function onFrame(): void {
    raf = requestAnimationFrame(onFrame);
    if (video.readyState < 2 || !handLandmarker) return;
    resizeCanvas();
    let result: HandLandmarkerResult;
    try {
      result = handLandmarker.detectForVideo(video, performance.now());
    } catch {
      return;
    }
    const hands = result.landmarks;
    if (!hands.length) {
      resetStroke();
      setStatus("Show your hand to the camera.");
      return;
    }
    const lm = hands[0];
    const tip = lm[INDEX_TIP];
    const thumb = lm[THUMB_TIP];
    const [x, y] = toCanvasXY(tip.x, tip.y);
    const pinching = dist2D(tip, thumb) < 0.06;
    const shouldDraw =
      paintMode === "always" ? true : pinching;
    strokeTo(x, y, shouldDraw);
    setStatus(
      paintMode === "pinch"
        ? pinching
          ? "Painting. Release pinch to move without drawing."
          : "Pinch thumb and index finger together to paint."
        : "Painting follows your index finger."
    );
  }

  async function start(): Promise<void> {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
    } catch {
      setStatus("Camera permission denied or unavailable.", true);
      return;
    }

    setStatus("Loading hand model…");
    const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
    const opts = {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: "GPU" as const,
      },
      runningMode: "VIDEO" as const,
      numHands: 1,
    };
    try {
      handLandmarker = await HandLandmarker.createFromOptions(fileset, opts);
    } catch {
      try {
        handLandmarker = await HandLandmarker.createFromOptions(fileset, {
          ...opts,
          baseOptions: { ...opts.baseOptions, delegate: "CPU" },
        });
      } catch {
        setStatus("Could not load hand tracking. Check network and reload.", true);
        return;
      }
    }

    setStatus("Ready.");
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(onFrame);
  }

  video.addEventListener("loadedmetadata", () => {
    resizeCanvas();
  });

  void start();
}

mount();
