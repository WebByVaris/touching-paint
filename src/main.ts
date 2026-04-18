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
const PINCH_CLOSE = 0.052;
const PINCH_OPEN = 0.074;
const HAND_LOST_FRAMES = 14;
const BRUSH_SMOOTH = 0.42;

const HAND_BONES: readonly [number, number][] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [5, 9],
  [9, 13],
  [13, 17],
];

type PaintMode = "pinch" | "always";

type Lm = { x: number; y: number };

function drawHandTracking(
  hctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  landmarks: Lm[][],
  pinchTip: Lm,
  pinchThumb: Lm,
  pinching: boolean
): void {
  hctx.clearRect(0, 0, w, h);
  if (!w || !h) return;
  const palette = [
    "rgba(90, 200, 255, 0.9)",
    "rgba(255, 140, 220, 0.9)",
  ];
  for (let hIdx = 0; hIdx < landmarks.length; hIdx++) {
    const lm = landmarks[hIdx];
    const stroke = palette[hIdx % palette.length];
    hctx.strokeStyle = stroke;
    hctx.lineWidth = 2.25;
    hctx.lineCap = "round";
    hctx.lineJoin = "round";
    hctx.beginPath();
    for (const [a, b] of HAND_BONES) {
      const pa = lm[a];
      const pb = lm[b];
      hctx.moveTo(pa.x * w, pa.y * h);
      hctx.lineTo(pb.x * w, pb.y * h);
    }
    hctx.stroke();
    hctx.fillStyle = "rgba(255, 255, 255, 0.92)";
    for (let i = 0; i < lm.length; i++) {
      const p = lm[i];
      const rad = i === 8 || i === 4 ? 4.5 : 3.2;
      hctx.beginPath();
      hctx.arc(p.x * w, p.y * h, rad, 0, Math.PI * 2);
      hctx.fill();
    }
  }
  if (pinching) {
    const mx = ((pinchTip.x + pinchThumb.x) * 0.5) * w;
    const my = ((pinchTip.y + pinchThumb.y) * 0.5) * h;
    const pr = dist2D(pinchTip, pinchThumb) * Math.min(w, h) * 0.5;
    hctx.strokeStyle = "rgba(255, 220, 120, 0.95)";
    hctx.lineWidth = 2;
    hctx.beginPath();
    hctx.arc(mx, my, Math.max(6, pr * 0.85), 0, Math.PI * 2);
    hctx.stroke();
  }
}

function dist2D(
  a: { x: number; y: number },
  b: { x: number; y: number }
): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function tipToClient(
  video: HTMLVideoElement,
  tip: { x: number; y: number }
): { x: number; y: number } {
  const r = video.getBoundingClientRect();
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return { x: 0, y: 0 };
  const scale = Math.max(r.width / vw, r.height / vh);
  const dispW = vw * scale;
  const dispH = vh * scale;
  const ox = r.left + (r.width - dispW) / 2;
  const oy = r.top + (r.height - dispH) / 2;
  const xUn = ox + tip.x * vw * scale;
  const y = oy + tip.y * vh * scale;
  const x = r.right - (xUn - r.left);
  return { x, y };
}

function pointInRectPad(
  px: number,
  py: number,
  rect: DOMRect,
  pad: number
): boolean {
  return (
    px >= rect.left - pad &&
    px <= rect.right + pad &&
    py >= rect.top - pad &&
    py <= rect.bottom + pad
  );
}

function mount(): void {
  const root = document.querySelector<HTMLDivElement>("#app");
  if (!root) return;

  root.innerHTML = `
    <div class="stage">
      <video id="cam" playsinline muted autoplay></video>
      <canvas id="hand"></canvas>
      <canvas id="layer"></canvas>
    </div>
    <div class="hud" aria-hidden="false">
      <div class="glass-panel">
        <div class="segmented" role="radiogroup" aria-label="Paint mode">
          <label class="seg" title="Pinch to paint"><input type="radio" name="mode" value="pinch" checked /><span>P</span></label>
          <label class="seg" title="Always paint"><input type="radio" name="mode" value="always" /><span>A</span></label>
        </div>
        <div class="list-group">
          <label class="cell cell-swatch">
            <input type="color" id="color" value="#5e9eff" aria-label="Color" title="Color" />
          </label>
          <div class="cell cell-size-row">
            <button
              type="button"
              id="size-trigger"
              class="size-icon-btn"
              aria-label="Brush size"
              aria-expanded="false"
              aria-controls="size-popover"
              title="Brush size"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" aria-hidden="true"><path d="M4 14h16M7 10h10M10 6h4" /></svg>
            </button>
          </div>
        </div>
        <button type="button" id="clear" class="secondary-btn" aria-label="Clear canvas" title="Clear">×</button>
      </div>
      <div id="size-popover" class="size-popover" hidden>
        <input type="range" id="size" min="2" max="48" value="14" aria-label="Brush size" />
      </div>
    </div>
  `;

  const video = root.querySelector<HTMLVideoElement>("#cam")!;
  const handCanvas = root.querySelector<HTMLCanvasElement>("#hand")!;
  const hctx = handCanvas.getContext("2d")!;
  const canvas = root.querySelector<HTMLCanvasElement>("#layer")!;
  const ctx = canvas.getContext("2d")!;
  const colorInput = root.querySelector<HTMLInputElement>("#color")!;
  const sizeInput = root.querySelector<HTMLInputElement>("#size")!;
  const sizeTrigger = root.querySelector<HTMLButtonElement>("#size-trigger")!;
  const sizePopover = root.querySelector<HTMLDivElement>("#size-popover")!;
  const glassPanel = root.querySelector<HTMLDivElement>(".glass-panel")!;
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
  let vfcHandle = 0;
  let prevPinch = false;
  let pinchLatched = false;
  let handLostFrames = 0;

  function setSizeOpen(open: boolean): void {
    sizePopover.hidden = !open;
    sizeTrigger.setAttribute("aria-expanded", String(open));
    if (open) {
      requestAnimationFrame(() => {
        updatePopoverPos();
        sizeInput.focus();
      });
    }
  }

  function updatePopoverPos(): void {
    if (sizePopover.hidden) return;
    const pr = glassPanel.getBoundingClientRect();
    const tr = sizeTrigger.getBoundingClientRect();
    const pad = 8;
    const w = sizePopover.offsetWidth || 140;
    const h = sizePopover.offsetHeight || 48;
    let top = tr.top + (tr.height - h) / 2;
    top = Math.max(pad, Math.min(top, window.innerHeight - h - pad));
    let left = pr.right + 6;
    if (left + w > window.innerWidth - pad) {
      left = pr.left - w - 6;
    }
    sizePopover.style.left = `${Math.max(pad, left)}px`;
    sizePopover.style.top = `${top}px`;
  }

  sizeTrigger.addEventListener("click", (e) => {
    e.stopPropagation();
    setSizeOpen(sizePopover.hidden);
  });

  document.addEventListener(
    "mousedown",
    (e) => {
      if (sizePopover.hidden) return;
      const t = e.target as Node;
      if (sizePopover.contains(t) || sizeTrigger.contains(t)) return;
      setSizeOpen(false);
    },
    true
  );

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !sizePopover.hidden) setSizeOpen(false);
  });

  window.addEventListener("resize", () => updatePopoverPos());

  function resizeCanvas(): void {
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return;
    if (handCanvas.width !== w || handCanvas.height !== h) {
      handCanvas.width = w;
      handCanvas.height = h;
    }
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

  function strokeTo(x: number, y: number, draw: boolean): void {
    const lw = Number(sizeInput.value);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = colorInput.value;
    ctx.lineWidth = lw;
    if (draw && lastX != null && lastY != null) {
      const dx = x - lastX;
      const dy = y - lastY;
      const len = Math.hypot(dx, dy);
      const step = Math.max(1.2, lw * 0.2);
      const n = Math.max(1, Math.ceil(len / step));
      ctx.beginPath();
      ctx.moveTo(lastX, lastY);
      for (let i = 1; i <= n; i++) {
        const t = i / n;
        ctx.lineTo(lastX + dx * t, lastY + dy * t);
      }
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

  function scheduleFrame(): void {
    if (typeof video.requestVideoFrameCallback === "function") {
      vfcHandle = video.requestVideoFrameCallback((now) => {
        onVideoFrame(now);
      });
    } else {
      raf = requestAnimationFrame(() => {
        onVideoFrame(performance.now());
      });
    }
  }

  function onVideoFrame(mediaTime: number): void {
    try {
      if (video.readyState < 2 || !handLandmarker) return;
      resizeCanvas();
      let result: HandLandmarkerResult;
      try {
        result = handLandmarker.detectForVideo(video, mediaTime);
      } catch {
        return;
      }
      const hands = result.landmarks;
      const cw0 = canvas.width;
      const ch0 = canvas.height;
      if (!hands.length) {
        if (cw0 && ch0) {
          hctx.clearRect(0, 0, cw0, ch0);
        }
        handLostFrames++;
        if (handLostFrames >= HAND_LOST_FRAMES) {
          resetStroke();
          pinchLatched = false;
          prevPinch = false;
        }
        return;
      }
      handLostFrames = 0;
      const lm = hands[0];
      const tip = lm[INDEX_TIP];
      const thumb = lm[THUMB_TIP];
      const cw = canvas.width;
      const ch = canvas.height;
      const rawX = tip.x * cw;
      const rawY = tip.y * ch;
      const dThumb = dist2D(tip, thumb);
      if (pinchLatched) {
        if (dThumb > PINCH_OPEN) pinchLatched = false;
      } else {
        if (dThumb < PINCH_CLOSE) pinchLatched = true;
      }
      const pinching = pinchLatched;
      const tipClient = tipToClient(video, tip);
      const tr = sizeTrigger.getBoundingClientRect();
      const onSizeIcon = pointInRectPad(tipClient.x, tipClient.y, tr, 14);
      if (!prevPinch && pinching && onSizeIcon) {
        setSizeOpen(sizePopover.hidden);
      }
      const suppressDraw = pinching && onSizeIcon;
      if (!suppressDraw) {
        const shouldDraw =
          paintMode === "always" ? true : pinching;
        if (shouldDraw) {
          let px = rawX;
          let py = rawY;
          if (lastX != null && lastY != null) {
            const a = BRUSH_SMOOTH;
            px = lastX + (rawX - lastX) * a;
            py = lastY + (rawY - lastY) * a;
          }
          strokeTo(px, py, true);
        } else {
          strokeTo(0, 0, false);
        }
      } else {
        resetStroke();
      }
      prevPinch = pinching;
      drawHandTracking(
        hctx,
        cw,
        ch,
        hands as Lm[][],
        tip,
        thumb,
        pinching
      );
    } finally {
      scheduleFrame();
    }
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
      window.alert("Camera permission denied or unavailable.");
      return;
    }

    const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
    const opts = {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: "GPU" as const,
      },
      runningMode: "VIDEO" as const,
      numHands: 2,
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
        window.alert("Could not load hand tracking. Check your network and reload.");
        return;
      }
    }

    cancelAnimationFrame(raf);
    if (typeof video.cancelVideoFrameCallback === "function") {
      video.cancelVideoFrameCallback(vfcHandle);
    }
    scheduleFrame();
  }

  video.addEventListener("loadedmetadata", () => {
    resizeCanvas();
  });

  void start();
}

mount();
