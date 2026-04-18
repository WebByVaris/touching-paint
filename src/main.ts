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
const PINKY_TIP = 20;
const ERASE_THUMB_PINKY_MAX_STRONG = 0.142;
const ERASE_THUMB_PINKY_MAX_WEAK = 0.162;
const PAINT_TRP_PINCH_MAX_STRONG = 0.096;
const PAINT_TRP_PINCH_MAX_WEAK = 0.114;
const PAINT_PINCH_ON_POINT = 0.068;
const SIZE_PINCH_MAX = 0.054;
const SIZE_SLIDER_MIN = 2;
const SIZE_SLIDER_MAX = 48;
const SIZE_BRUSH_PINCH_MAX = 0.148;
const SIZE_BRUSH_PINCH_MIN = 0.022;
const SIZE_TRACK_HIT_PAD = 22;
const SIZE_UI_LEAVE_PAD = 16;
const ERASER_WIDTH_MULT = 3;
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

type PaintMode = "point" | "always";

type Lm = { x: number; y: number };

function drawHandTracking(
  hctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  landmarks: Lm[][],
  paintPinchRing: boolean[],
  eraserActive: boolean[],
  fistClenched: boolean[],
  palmOpen: boolean[],
  paintLineWidth: number
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
    const fc = fistClenched[hIdx] ?? false;
    const po = palmOpen[hIdx] ?? false;
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
      const rad = fc || po || (i !== 8 && i !== 4) ? 3.2 : 4.5;
      hctx.beginPath();
      hctx.arc(p.x * w, p.y * h, rad, 0, Math.PI * 2);
      hctx.fill();
    }
  }
  for (let hIdx = 0; hIdx < landmarks.length; hIdx++) {
    const lm = landmarks[hIdx];
    const fc = fistClenched[hIdx] ?? false;
    const po = palmOpen[hIdx] ?? false;
    const er = eraserActive[hIdx] ?? false;
    const pr = paintPinchRing[hIdx] ?? false;
    if (er) {
      const c = threeFingerTipsCentroidLm(lm);
      const tx = c.x * w;
      const ty = c.y * h;
      const rr = Math.max(
        12,
        paintLineWidth * ERASER_WIDTH_MULT * 0.52
      );
      hctx.strokeStyle = "rgba(255, 180, 90, 0.95)";
      hctx.lineWidth = 2;
      hctx.beginPath();
      hctx.arc(tx, ty, rr, 0, Math.PI * 2);
      hctx.stroke();
    } else if (pr && !fc && !po) {
      const tx = lm[INDEX_TIP].x * w;
      const ty = lm[INDEX_TIP].y * h;
      const rr = Math.max(8, paintLineWidth * 0.52);
      hctx.strokeStyle = "rgba(120, 255, 200, 0.92)";
      hctx.lineWidth = 2;
      hctx.beginPath();
      hctx.arc(tx, ty, rr, 0, Math.PI * 2);
      hctx.stroke();
    }
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

function thumbRingPinkyPinchCluster(lm: Lm[], maxPairSpan: number): boolean {
  const th = lm[THUMB_TIP];
  const ri = lm[16];
  const pi = lm[PINKY_TIP];
  const dTr = dist2D(th, ri);
  const dTp = dist2D(th, pi);
  const dRp = dist2D(ri, pi);
  return Math.max(dTr, dTp, dRp) < maxPairSpan;
}

function indexMiddlePaintPoseStrong(lm: Lm[]): boolean {
  if (!thumbRingPinkyPinchCluster(lm, PAINT_TRP_PINCH_MAX_STRONG)) {
    return false;
  }
  const d58 = dist2D(lm[5], lm[8]);
  const d08 = dist2D(lm[0], lm[8]);
  const d06 = dist2D(lm[0], lm[6]);
  const d912 = dist2D(lm[9], lm[12]);
  const d012 = dist2D(lm[0], lm[12]);
  const d010 = dist2D(lm[0], lm[10]);
  return (
    d58 > 0.066 &&
    d08 > 0.084 &&
    d08 >= d06 * 0.95 &&
    d912 > 0.064 &&
    d012 > 0.082 &&
    d012 >= d010 * 0.95
  );
}

function indexMiddlePaintPoseWeak(lm: Lm[]): boolean {
  if (!thumbRingPinkyPinchCluster(lm, PAINT_TRP_PINCH_MAX_WEAK)) {
    return false;
  }
  const d58 = dist2D(lm[5], lm[8]);
  const d08 = dist2D(lm[0], lm[8]);
  const d06 = dist2D(lm[0], lm[6]);
  const d912 = dist2D(lm[9], lm[12]);
  const d012 = dist2D(lm[0], lm[12]);
  const d010 = dist2D(lm[0], lm[10]);
  return (
    d58 > 0.056 &&
    d08 > 0.074 &&
    d08 >= d06 * 0.92 &&
    d912 > 0.056 &&
    d012 > 0.072 &&
    d012 >= d010 * 0.92
  );
}

function threeFingerErasePoseStrong(lm: Lm[]): boolean {
  const d58 = dist2D(lm[5], lm[8]);
  const d912 = dist2D(lm[9], lm[12]);
  const d1316 = dist2D(lm[13], lm[16]);
  const d08 = dist2D(lm[0], lm[8]);
  const d012 = dist2D(lm[0], lm[12]);
  const d016 = dist2D(lm[0], lm[16]);
  const d06 = dist2D(lm[0], lm[6]);
  const d010 = dist2D(lm[0], lm[10]);
  const d014 = dist2D(lm[0], lm[14]);
  const dThumbPinky = dist2D(lm[THUMB_TIP], lm[PINKY_TIP]);
  return (
    d58 > 0.070 &&
    d912 > 0.066 &&
    d1316 > 0.058 &&
    d08 >= d06 * 0.96 &&
    d012 >= d010 * 0.96 &&
    d016 >= d014 * 0.96 &&
    dThumbPinky < ERASE_THUMB_PINKY_MAX_STRONG
  );
}

function threeFingerErasePoseWeak(lm: Lm[]): boolean {
  const d58 = dist2D(lm[5], lm[8]);
  const d912 = dist2D(lm[9], lm[12]);
  const d1316 = dist2D(lm[13], lm[16]);
  const d08 = dist2D(lm[0], lm[8]);
  const d012 = dist2D(lm[0], lm[12]);
  const d016 = dist2D(lm[0], lm[16]);
  const d06 = dist2D(lm[0], lm[6]);
  const d010 = dist2D(lm[0], lm[10]);
  const d014 = dist2D(lm[0], lm[14]);
  const dThumbPinky = dist2D(lm[THUMB_TIP], lm[PINKY_TIP]);
  return (
    d58 > 0.062 &&
    d912 > 0.058 &&
    d1316 > 0.050 &&
    d08 >= d06 * 0.94 &&
    d012 >= d010 * 0.94 &&
    d016 >= d014 * 0.94 &&
    dThumbPinky < ERASE_THUMB_PINKY_MAX_WEAK
  );
}

function threeFingerTipsCentroidLm(lm: Lm[]): Lm {
  return {
    x: (lm[INDEX_TIP].x + lm[12].x + lm[16].x) / 3,
    y: (lm[INDEX_TIP].y + lm[12].y + lm[16].y) / 3,
  };
}

function fistStrong(lm: Lm[]): boolean {
  const d04 = dist2D(lm[0], lm[4]);
  if (d04 > 0.112) return false;
  const d08 = dist2D(lm[0], lm[8]);
  const d012 = dist2D(lm[0], lm[12]);
  const d016 = dist2D(lm[0], lm[16]);
  const d020 = dist2D(lm[0], lm[20]);
  return (
    d08 < 0.096 &&
    d012 < 0.096 &&
    d016 < 0.102 &&
    d020 < 0.102
  );
}

function fistWeak(lm: Lm[]): boolean {
  const d04 = dist2D(lm[0], lm[4]);
  if (d04 > 0.125) return false;
  const d08 = dist2D(lm[0], lm[8]);
  const d012 = dist2D(lm[0], lm[12]);
  const d016 = dist2D(lm[0], lm[16]);
  const d020 = dist2D(lm[0], lm[20]);
  return (
    d08 < 0.112 &&
    d012 < 0.112 &&
    d016 < 0.118 &&
    d020 < 0.118
  );
}

function palmOpenStrong(lm: Lm[]): boolean {
  const d58 = dist2D(lm[5], lm[8]);
  const d912 = dist2D(lm[9], lm[12]);
  const d1316 = dist2D(lm[13], lm[16]);
  const d1720 = dist2D(lm[17], lm[20]);
  const d04 = dist2D(lm[0], lm[4]);
  const tip812 = dist2D(lm[8], lm[12]);
  return (
    d58 > 0.098 &&
    d912 > 0.094 &&
    d1316 > 0.086 &&
    d1720 > 0.082 &&
    d04 > 0.096 &&
    tip812 > 0.062
  );
}

function palmOpenWeak(lm: Lm[]): boolean {
  const d58 = dist2D(lm[5], lm[8]);
  const d912 = dist2D(lm[9], lm[12]);
  const d1316 = dist2D(lm[13], lm[16]);
  const d1720 = dist2D(lm[17], lm[20]);
  const d04 = dist2D(lm[0], lm[4]);
  const tip812 = dist2D(lm[8], lm[12]);
  return (
    d58 > 0.088 &&
    d912 > 0.084 &&
    d1316 > 0.078 &&
    d1720 > 0.074 &&
    d04 > 0.088 &&
    tip812 > 0.052
  );
}

function threeStraightClearPose(lm: Lm[]): boolean {
  const d58 = dist2D(lm[5], lm[8]);
  const d912 = dist2D(lm[9], lm[12]);
  const d1316 = dist2D(lm[13], lm[16]);
  const d08 = dist2D(lm[0], lm[8]);
  const d012 = dist2D(lm[0], lm[12]);
  const d016 = dist2D(lm[0], lm[16]);
  const d06 = dist2D(lm[0], lm[6]);
  const d010 = dist2D(lm[0], lm[10]);
  const d014 = dist2D(lm[0], lm[14]);
  const d1720 = dist2D(lm[17], lm[20]);
  const d04 = dist2D(lm[0], lm[4]);
  const threeUp =
    d58 > 0.06 &&
    d912 > 0.057 &&
    d1316 > 0.053 &&
    d08 >= d06 * 0.96 &&
    d012 >= d010 * 0.96 &&
    d016 >= d014 * 0.96;
  if (!threeUp) return false;
  if (d1720 > 0.08 && d04 > 0.092) return false;
  return true;
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

function pinchMidClient(
  video: HTMLVideoElement,
  tip: { x: number; y: number },
  thumb: { x: number; y: number }
): { x: number; y: number } {
  const a = tipToClient(video, tip);
  const b = tipToClient(video, thumb);
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

type RectLike = Pick<DOMRect, "left" | "top" | "right" | "bottom">;

function pointInRectPad(
  px: number,
  py: number,
  rect: RectLike,
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
        <div class="segmented segmented--row" role="radiogroup" aria-label="Paint mode">
          <label class="seg" title="Index and middle open; thumb, ring, and pinky tips pinched together; tighter pose to paint"><input type="radio" name="mode" value="point" checked /><span>P</span></label>
          <label class="seg" title="Index and middle open; thumb, ring, and pinky tips pinched together; relaxed pose keeps stroke"><input type="radio" name="mode" value="always" /><span>A</span></label>
        </div>
        <div class="palette" role="group" aria-label="Colors, pinch or tap to select">
          <button type="button" class="palette-swatch" data-color="#5e9eff" style="background:#5e9eff" aria-label="Blue" title="Blue"></button>
          <button type="button" class="palette-swatch" data-color="#ff6b6b" style="background:#ff6b6b" aria-label="Coral" title="Coral"></button>
          <button type="button" class="palette-swatch" data-color="#34d399" style="background:#34d399" aria-label="Mint" title="Mint"></button>
          <button type="button" class="palette-swatch" data-color="#a78bfa" style="background:#a78bfa" aria-label="Violet" title="Violet"></button>
          <button type="button" class="palette-swatch" data-color="#fbbf24" style="background:#fbbf24" aria-label="Amber" title="Amber"></button>
          <button type="button" class="palette-swatch" data-color="#f472b6" style="background:#f472b6" aria-label="Pink" title="Pink"></button>
        </div>
        <input type="hidden" id="color" value="#5e9eff" />
        <div class="tools-row">
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
          <button type="button" id="clear" class="secondary-btn" aria-label="Clear canvas" title="Clear">×</button>
        </div>
      </div>
      <div id="size-popover" class="size-popover" hidden>
        <div class="size-popover__top">
          <span class="size-popover__label">Brush</span>
          <span id="size-readout" class="size-popover__readout" aria-live="polite">14</span>
        </div>
        <div class="size-popover__dial" aria-hidden="true">
          <svg class="size-popover__dial-svg" viewBox="0 0 80 80" width="100%" height="80" preserveAspectRatio="xMidYMid meet">
            <defs>
              <linearGradient id="sizeDialGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#5ac8fa" />
                <stop offset="100%" stop-color="#0a84ff" />
              </linearGradient>
            </defs>
            <circle id="size-dial-brush" class="size-popover__dial-brush" cx="40" cy="40" r="6" fill="url(#sizeDialGrad)" />
          </svg>
        </div>
        <div id="size-track-hit" class="size-popover__track">
          <div class="size-popover__track-rail" aria-hidden="true"></div>
          <div class="size-popover__track-fill" aria-hidden="true"></div>
          <input type="range" id="size" class="size-popover__range" min="2" max="48" value="14" aria-label="Brush size" />
        </div>
        <p class="size-popover__hint">Pinch the brush icon to open · move your index away to close · pinch the bar to set size · fist rotates anytime</p>
      </div>
    </div>
    <div id="aim-dot" class="aim-dot" hidden></div>
    <div id="aim-dot-b" class="aim-dot aim-dot--b" hidden></div>
  `;

  const video = root.querySelector<HTMLVideoElement>("#cam")!;
  const handCanvas = root.querySelector<HTMLCanvasElement>("#hand")!;
  const hctx = handCanvas.getContext("2d")!;
  const canvas = root.querySelector<HTMLCanvasElement>("#layer")!;
  const ctx = canvas.getContext("2d")!;
  const colorInput = root.querySelector<HTMLInputElement>("#color")!;
  const swatches = root.querySelectorAll<HTMLButtonElement>(".palette-swatch");
  const sizeInput = root.querySelector<HTMLInputElement>("#size")!;
  const sizeTrigger = root.querySelector<HTMLButtonElement>("#size-trigger")!;
  const sizePopover = root.querySelector<HTMLDivElement>("#size-popover")!;
  const sizeReadout = root.querySelector<HTMLSpanElement>("#size-readout")!;
  const sizeTrackHit = root.querySelector<HTMLDivElement>("#size-track-hit")!;
  const sizeDialBrush = root.querySelector<SVGCircleElement>("#size-dial-brush")!;
  const clearBtn = root.querySelector<HTMLButtonElement>("#clear")!;
  const aimDot = root.querySelector<HTMLDivElement>("#aim-dot")!;
  const aimDotB = root.querySelector<HTMLDivElement>("#aim-dot-b")!;

  const handColor: [string, string] = [
    (colorInput.value || "#5e9eff").trim(),
    (colorInput.value || "#5e9eff").trim(),
  ];

  function syncSwatchSelection(): void {
    const c0 = handColor[0].trim().toLowerCase();
    const c1 = handColor[1].trim().toLowerCase();
    swatches.forEach((b) => {
      const bc = (b.dataset.color || "").trim().toLowerCase();
      b.classList.toggle("pick-h0", bc === c0);
      b.classList.toggle("pick-h1", bc === c1);
    });
  }

  swatches.forEach((b) => {
    b.addEventListener("click", () => {
      const v = (b.dataset.color || "#5e9eff").trim();
      handColor[0] = v;
      handColor[1] = v;
      colorInput.value = v;
      syncSwatchSelection();
    });
  });
  syncSwatchSelection();

  let paintMode: PaintMode = "point";
  root.querySelectorAll<HTMLInputElement>('input[name="mode"]').forEach((r) => {
    r.addEventListener("change", () => {
      if (r.checked) paintMode = r.value as PaintMode;
    });
  });

  const lastX: (number | null)[] = [null, null];
  const lastY: (number | null)[] = [null, null];
  let handLandmarker: HandLandmarker | null = null;
  let raf = 0;
  let vfcHandle = 0;
  const prevUiPinch: boolean[] = [false, false];
  const paintPinchLatched: boolean[] = [false, false];
  const eraserLatched: boolean[] = [false, false];
  const fistLatched: boolean[] = [false, false];
  const palmLatched: boolean[] = [false, false];
  const prevTool: ("n" | "p" | "e")[] = ["n", "n"];
  let handLostFrames = 0;
  let wasFistAdjustPrev = false;
  let fistRotPrevAng: number | null = null;
  let fistRotStartSize = 14;
  let fistRotAccum = 0;
  let dualClearStreak = 0;

  function syncSizeUi(): void {
    const v = Math.max(
      SIZE_SLIDER_MIN,
      Math.min(
        SIZE_SLIDER_MAX,
        Math.round(Number(sizeInput.value) || SIZE_SLIDER_MIN)
      )
    );
    if (Number(sizeInput.value) !== v) sizeInput.value = String(v);
    sizeReadout.textContent = String(v);
    const span = SIZE_SLIDER_MAX - SIZE_SLIDER_MIN;
    const p = span > 0 ? (v - SIZE_SLIDER_MIN) / span : 0;
    sizeTrackHit.style.setProperty("--p", String(p));
    const brushR = 4 + p * 30;
    sizeDialBrush.setAttribute("r", String(brushR));
    sizeDialBrush.style.opacity = String(0.78 + p * 0.22);
  }

  sizeInput.addEventListener("input", () => syncSizeUi());
  syncSizeUi();

  function setSizeOpen(open: boolean): void {
    sizePopover.hidden = !open;
    sizeTrigger.setAttribute("aria-expanded", String(open));
    if (open) {
      requestAnimationFrame(() => {
        updatePopoverPos();
        syncSizeUi();
        sizeInput.focus();
      });
    }
  }

  function updatePopoverPos(): void {
    if (sizePopover.hidden) return;
    const tr = sizeTrigger.getBoundingClientRect();
    const pad = 8;
    const w = sizePopover.offsetWidth || 140;
    const h = sizePopover.offsetHeight || 48;
    let top = tr.top - h - 8;
    if (top < pad) {
      top = Math.min(tr.bottom + 8, window.innerHeight - h - pad);
    }
    top = Math.max(pad, Math.min(top, window.innerHeight - h - pad));
    let left = tr.left + tr.width / 2 - w / 2;
    left = Math.max(pad, Math.min(left, window.innerWidth - w - pad));
    sizePopover.style.left = `${left}px`;
    sizePopover.style.top = `${top}px`;
  }

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

  function strokeTo(
    x: number,
    y: number,
    draw: boolean,
    erase: boolean,
    handIdx: number
  ): void {
    const lwPaint = Math.max(
      SIZE_SLIDER_MIN,
      Math.min(SIZE_SLIDER_MAX, Number(sizeInput.value) || SIZE_SLIDER_MIN)
    );
    const lw = erase ? lwPaint * ERASER_WIDTH_MULT : lwPaint;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.globalCompositeOperation = erase ? "destination-out" : "source-over";
    ctx.strokeStyle = erase ? "#000" : handColor[handIdx] || handColor[0];
    ctx.lineWidth = lw;
    const lx = lastX[handIdx];
    const ly = lastY[handIdx];
    if (draw && lx != null && ly != null) {
      const dx = x - lx;
      const dy = y - ly;
      const len = Math.hypot(dx, dy);
      const step = Math.max(1.2, lw * 0.2);
      const n = Math.max(1, Math.ceil(len / step));
      ctx.beginPath();
      ctx.moveTo(lx, ly);
      for (let i = 1; i <= n; i++) {
        const t = i / n;
        ctx.lineTo(lx + dx * t, ly + dy * t);
      }
      ctx.stroke();
    }
    ctx.globalCompositeOperation = "source-over";
    if (draw) {
      lastX[handIdx] = x;
      lastY[handIdx] = y;
    } else {
      lastX[handIdx] = null;
      lastY[handIdx] = null;
    }
  }

  function resetStroke(): void {
    lastX[0] = null;
    lastY[0] = null;
    lastX[1] = null;
    lastY[1] = null;
  }

  clearBtn.addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    resetStroke();
    prevTool[0] = "n";
    prevTool[1] = "n";
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
        aimDot.hidden = true;
        aimDotB.hidden = true;
        if (cw0 && ch0) {
          hctx.clearRect(0, 0, cw0, ch0);
        }
        paintPinchLatched[0] = false;
        paintPinchLatched[1] = false;
        eraserLatched[0] = false;
        eraserLatched[1] = false;
        fistLatched[0] = false;
        fistLatched[1] = false;
        palmLatched[0] = false;
        palmLatched[1] = false;
        wasFistAdjustPrev = false;
        fistRotPrevAng = null;
        dualClearStreak = 0;
        prevUiPinch[0] = false;
        prevUiPinch[1] = false;
        handLostFrames++;
        if (handLostFrames >= HAND_LOST_FRAMES) {
          resetStroke();
          prevTool[0] = "n";
          prevTool[1] = "n";
        }
        return;
      }
      handLostFrames = 0;
      if (hands.length >= 2) {
        const dualClear =
          threeStraightClearPose(hands[0]) &&
          threeStraightClearPose(hands[1]);
        if (dualClear) {
          dualClearStreak++;
          if (dualClearStreak === 4) {
            if (canvas.width && canvas.height) {
              ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
            resetStroke();
            prevTool[0] = "n";
            prevTool[1] = "n";
          }
        } else {
          dualClearStreak = 0;
        }
      } else {
        dualClearStreak = 0;
      }
      const nHands = Math.min(2, hands.length);
      for (let hi = nHands; hi < 2; hi++) {
        paintPinchLatched[hi] = false;
        eraserLatched[hi] = false;
        fistLatched[hi] = false;
        palmLatched[hi] = false;
        lastX[hi] = null;
        lastY[hi] = null;
        prevTool[hi] = "n";
        prevUiPinch[hi] = false;
      }
      const cw = canvas.width;
      const ch = canvas.height;
      const paintByPinch: boolean[] = [false, false];
      const erasing: boolean[] = [false, false];
      const fistClenched: boolean[] = [false, false];
      const palmOpen: boolean[] = [false, false];
      const tip: Lm[] = [];
      const thumb: Lm[] = [];
      const threeFingerEraseAim: Lm[] = [];
      for (let hi = 0; hi < nHands; hi++) {
        const lm = hands[hi];
        const t = lm[INDEX_TIP];
        const th = lm[THUMB_TIP];
        tip.push(t);
        thumb.push(th);
        threeFingerEraseAim.push(threeFingerTipsCentroidLm(lm));
        const thumbIndexSpread = dist2D(t, th);
        if (
          thumbIndexSpread < PAINT_PINCH_ON_POINT &&
          !palmOpenStrong(lm)
        ) {
          eraserLatched[hi] = false;
        } else {
          if (eraserLatched[hi]) {
            if (!threeFingerErasePoseWeak(lm)) eraserLatched[hi] = false;
          } else {
            if (threeFingerErasePoseStrong(lm)) eraserLatched[hi] = true;
          }
        }
        erasing[hi] = eraserLatched[hi];
        if (erasing[hi]) {
          paintPinchLatched[hi] = false;
        } else {
          const indexHold =
            paintMode === "always"
              ? indexMiddlePaintPoseWeak(lm)
              : indexMiddlePaintPoseStrong(lm);
          if (paintPinchLatched[hi]) {
            if (!indexHold || fistWeak(lm)) paintPinchLatched[hi] = false;
          } else if (indexMiddlePaintPoseStrong(lm) && !fistWeak(lm)) {
            paintPinchLatched[hi] = true;
          }
        }
        paintByPinch[hi] = paintPinchLatched[hi];
        if (fistLatched[hi]) {
          if (!fistWeak(lm)) fistLatched[hi] = false;
        } else {
          if (fistStrong(lm)) fistLatched[hi] = true;
        }
        fistClenched[hi] = fistLatched[hi];
        if (palmLatched[hi]) {
          if (!palmOpenWeak(lm)) palmLatched[hi] = false;
        } else {
          if (palmOpenStrong(lm)) palmLatched[hi] = true;
        }
        palmOpen[hi] = palmLatched[hi];
      }
      let rotHand: number | null = null;
      for (let hi = 0; hi < nHands; hi++) {
        if (fistClenched[hi] && !erasing[hi]) {
          rotHand = hi;
          break;
        }
      }
      if (rotHand !== null) {
        const rlm = hands[rotHand];
        if (!wasFistAdjustPrev) {
          fistRotStartSize = Math.max(
            SIZE_SLIDER_MIN,
            Math.min(
              SIZE_SLIDER_MAX,
              Number(sizeInput.value) || SIZE_SLIDER_MIN
            )
          );
          fistRotAccum = 0;
          fistRotPrevAng = null;
        }
        const ang = Math.atan2(rlm[9].y - rlm[0].y, rlm[9].x - rlm[0].x);
        if (fistRotPrevAng !== null) {
          let d = ang - fistRotPrevAng;
          if (d > Math.PI) d -= 2 * Math.PI;
          if (d < -Math.PI) d += 2 * Math.PI;
          fistRotAccum += d;
          const span = SIZE_SLIDER_MAX - SIZE_SLIDER_MIN;
          const sens = span / (2 * Math.PI);
          const next = Math.max(
            SIZE_SLIDER_MIN,
            Math.min(
              SIZE_SLIDER_MAX,
              Math.round(fistRotStartSize + fistRotAccum * sens)
            )
          );
          sizeInput.value = String(next);
        }
        fistRotPrevAng = ang;
        wasFistAdjustPrev = true;
      } else {
        fistRotPrevAng = null;
        wasFistAdjustPrev = false;
      }
      const tr = sizeTrigger.getBoundingClientRect();
      const sizePopoverVisible = !sizePopover.hidden;
      const trkRect = sizePopoverVisible
        ? sizeTrackHit.getBoundingClientRect()
        : null;
      const suppressDraw: boolean[] = [false, false];
      const onPaletteSwatch: boolean[] = [false, false];
      const onSizeIcon: boolean[] = [false, false];
      const onSizeTrackPinch: boolean[] = [false, false];
      for (let hi = 0; hi < nHands; hi++) {
        const t = tip[hi];
        const th = thumb[hi];
        const thumbIndexSpread = dist2D(t, th);
        const tipClient = tipToClient(video, t);
        onSizeIcon[hi] = pointInRectPad(tipClient.x, tipClient.y, tr, 14);
        let pal = false;
        for (let si = 0; si < swatches.length; si++) {
          const sr = swatches[si].getBoundingClientRect();
          if (pointInRectPad(tipClient.x, tipClient.y, sr, 14)) {
            pal = true;
            break;
          }
        }
        onPaletteSwatch[hi] = pal;
        const uiPinch = thumbIndexSpread < SIZE_PINCH_MAX;
        const brushPinchForSize =
          thumbIndexSpread < SIZE_BRUSH_PINCH_MAX &&
          thumbIndexSpread > SIZE_BRUSH_PINCH_MIN;
        if (
          sizePopoverVisible &&
          brushPinchForSize &&
          trkRect &&
          trkRect.width > 2
        ) {
          const pinchMid = pinchMidClient(video, t, th);
          onSizeTrackPinch[hi] = pointInRectPad(
            pinchMid.x,
            pinchMid.y,
            trkRect,
            SIZE_TRACK_HIT_PAD
          );
        }
        if (
          onSizeTrackPinch[hi] &&
          !(fistClenched[hi] && !erasing[hi]) &&
          trkRect
        ) {
          const pinchMid = pinchMidClient(video, t, th);
          const u = Math.max(
            0,
            Math.min(
              1,
              (pinchMid.x - trkRect.left) / Math.max(1, trkRect.width)
            )
          );
          const nv = Math.round(
            Math.max(
              SIZE_SLIDER_MIN,
              Math.min(
                SIZE_SLIDER_MAX,
                SIZE_SLIDER_MIN +
                  u * (SIZE_SLIDER_MAX - SIZE_SLIDER_MIN)
              )
            )
          );
          if (nv !== Number(sizeInput.value)) sizeInput.value = String(nv);
        }
        if (!prevUiPinch[hi] && uiPinch) {
          if (onSizeIcon[hi] && sizePopover.hidden) {
            setSizeOpen(true);
            void sizePopover.offsetHeight;
            updatePopoverPos();
          } else if (onPaletteSwatch[hi]) {
            const tcp = tipToClient(video, t);
            for (let si = 0; si < swatches.length; si++) {
              const el = swatches[si];
              const sr = el.getBoundingClientRect();
              if (pointInRectPad(tcp.x, tcp.y, sr, 14)) {
                const nv = (el.dataset.color || handColor[hi]).trim();
                handColor[hi] = nv;
                if (hi === 0) colorInput.value = nv;
                syncSwatchSelection();
                break;
              }
            }
          }
        }
        prevUiPinch[hi] = uiPinch;
        suppressDraw[hi] =
          (uiPinch && (onSizeIcon[hi] || onPaletteSwatch[hi])) ||
          onSizeTrackPinch[hi];
      }
      let anyHandScrubbingSizeTrack = false;
      for (let hi = 0; hi < nHands; hi++) {
        if (onSizeTrackPinch[hi]) {
          anyHandScrubbingSizeTrack = true;
          break;
        }
      }
      if (sizePopoverVisible && !anyHandScrubbingSizeTrack) {
        const pr = sizePopover.getBoundingClientRect();
        const unionBounds: RectLike = {
          left: Math.min(tr.left, pr.left),
          top: Math.min(tr.top, pr.top),
          right: Math.max(tr.right, pr.right),
          bottom: Math.max(tr.bottom, pr.bottom),
        };
        let anyTipInside = false;
        for (let hi = 0; hi < nHands; hi++) {
          const tipClient = tipToClient(video, tip[hi]);
          if (
            pointInRectPad(
              tipClient.x,
              tipClient.y,
              unionBounds,
              SIZE_UI_LEAVE_PAD
            )
          ) {
            anyTipInside = true;
            break;
          }
        }
        if (!anyTipInside) setSizeOpen(false);
      }
      const hot0 =
        onPaletteSwatch[0] || onSizeIcon[0] || onSizeTrackPinch[0];
      const hot1 =
        nHands >= 2 &&
        (onPaletteSwatch[1] || onSizeIcon[1] || onSizeTrackPinch[1]);
      const aimLm0 = erasing[0] ? threeFingerEraseAim[0] : tip[0];
      const aimClient0 = tipToClient(video, aimLm0);
      aimDot.hidden = false;
      aimDot.style.left = `${aimClient0.x}px`;
      aimDot.style.top = `${aimClient0.y}px`;
      aimDot.classList.toggle("aim-dot--hot", hot0);
      aimDot.classList.toggle("aim-dot--erase", erasing[0]);
      if (nHands >= 2) {
        const useTrackB = onSizeTrackPinch[1] && sizePopoverVisible;
        const aimClient1 = useTrackB
          ? pinchMidClient(video, tip[1], thumb[1])
          : tipToClient(
              video,
              erasing[1] ? threeFingerEraseAim[1] : tip[1]
            );
        aimDotB.hidden = false;
        aimDotB.style.left = `${aimClient1.x}px`;
        aimDotB.style.top = `${aimClient1.y}px`;
        aimDotB.classList.toggle("aim-dot--hot", hot1 || useTrackB);
        aimDotB.classList.toggle("aim-dot--erase", erasing[1]);
      } else {
        aimDotB.hidden = true;
      }
      for (let hi = 0; hi < nHands; hi++) {
        if (!suppressDraw[hi]) {
          const indexPaintOk =
            paintMode === "always"
              ? indexMiddlePaintPoseWeak(hands[hi])
              : indexMiddlePaintPoseStrong(hands[hi]);
          const painting =
            !erasing[hi] &&
            !fistClenched[hi] &&
            !palmOpen[hi] &&
            paintByPinch[hi] &&
            indexPaintOk;
          const tool: "n" | "p" | "e" = erasing[hi] ? "e" : painting ? "p" : "n";
          if (tool !== prevTool[hi]) {
            lastX[hi] = null;
            lastY[hi] = null;
          }
          prevTool[hi] = tool;
          if (erasing[hi]) {
            const rawTx = threeFingerEraseAim[hi].x * cw;
            const rawTy = threeFingerEraseAim[hi].y * ch;
            let px = rawTx;
            let py = rawTy;
            const lx = lastX[hi];
            const ly = lastY[hi];
            if (lx != null && ly != null) {
              const a = BRUSH_SMOOTH;
              px = lx + (rawTx - lx) * a;
              py = ly + (rawTy - ly) * a;
            }
            strokeTo(px, py, true, true, hi);
          } else if (painting) {
            const rawX = tip[hi].x * cw;
            const rawY = tip[hi].y * ch;
            let px = rawX;
            let py = rawY;
            const lx = lastX[hi];
            const ly = lastY[hi];
            if (lx != null && ly != null) {
              const a = BRUSH_SMOOTH;
              px = lx + (rawX - lx) * a;
              py = ly + (rawY - ly) * a;
            }
            strokeTo(px, py, true, false, hi);
          } else {
            strokeTo(0, 0, false, false, hi);
          }
        } else {
          strokeTo(0, 0, false, false, hi);
          prevTool[hi] = "n";
        }
      }
      syncSizeUi();
      const paintRing: boolean[] = [];
      const eraseRing: boolean[] = [];
      for (let hi = 0; hi < nHands; hi++) {
        paintRing.push(
          paintByPinch[hi] &&
            !erasing[hi] &&
            !fistClenched[hi] &&
            !palmOpen[hi] &&
            (paintMode === "always"
              ? indexMiddlePaintPoseWeak(hands[hi])
              : indexMiddlePaintPoseStrong(hands[hi]))
        );
        eraseRing.push(erasing[hi]);
      }
      drawHandTracking(
        hctx,
        cw,
        ch,
        hands as Lm[][],
        paintRing,
        eraseRing,
        fistClenched.slice(0, hands.length),
        palmOpen.slice(0, hands.length),
        Math.max(
          SIZE_SLIDER_MIN,
          Math.min(
            SIZE_SLIDER_MAX,
            Number(sizeInput.value) || SIZE_SLIDER_MIN
          )
        )
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
