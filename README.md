# Touching Paint

Web app that uses your webcam and **MediaPipe Hand Landmarker** so you can paint on a mirrored live view with hand gestures—no mouse or stylus required.

---

## Requirements

- **Node.js** (LTS recommended) and **npm**
- A **webcam**
- A **modern browser** with WebGL and `getUserMedia` (Chrome, Edge, Safari, Firefox)
- **Network access** on first load: the app loads MediaPipe WebAssembly from jsDelivr and the hand model from Google Cloud Storage (see `src/main.ts` URLs). After that, chunks may be cached by the browser.

---

## Quick start

From the project root:

```bash
npm install
npm run dev
```

Vite prints a local URL (usually `http://localhost:5173`). Open it in your browser and **allow camera access** when prompted.

### Other commands

| Command | Purpose |
|--------|---------|
| `npm run dev` | Development server with hot reload |
| `npm run build` | Typecheck (`tsc`) and production bundle (`vite build`) into `dist/` |
| `npm run preview` | Serve the production build locally to test `dist/` |

---

## How the UI is laid out

- **Video** fills the screen; the **drawing layer** and a **hand skeleton overlay** sit on top (mirrored horizontally so it feels like a mirror).
- **Bottom bar (HUD):** paint mode (**P** / **A**), color swatches, **brush size** control, and **×** clear.
- **Aim dots** track each hand: normal cursor position; they highlight when you are over a UI target or using the brush-size track; eraser mode uses a distinct style.

---

## Gestures (reference)

All distances are in **normalized landmark space** (roughly “fraction of hand size” in the frame). Lighting, motion, and camera angle affect detection—use **deliberate, steady poses** when learning.

### Paint mode: **P** (point) vs **A** (always)

- **P — Point:** Uses a **stricter** “paint pose” to start and maintain strokes. Good when you want strokes only while the pose is crisp.
- **A — Always:** Uses a **relaxed** paint pose so the stroke is easier to keep once started.

Shared idea for painting:

1. **Thumb, ring, and pinky tips** are **pinched together** (a tight cluster).
2. **Index and middle fingers** stay **up and open** in the paint pose (not a fist, not a flat open palm in the “palm open” sense used by the app).

Painting happens while the app has **latched** that paint pinch: you **enter** the pose to arm painting, then move your **index tip**; the stroke follows in camera space (with light smoothing).

**Blocked while painting:** **fist** (hand clenched) and **palm open** (wide open hand) cancel the paint latch for that hand.

### Eraser

- Use the **three-finger erase pose:** index, middle, and ring extended in a similar “up” configuration, with **thumb and pinky close together** (see erase thresholds in `src/main.ts`).
- The erase brush is wider than the paint brush (fixed multiple of the current brush size).
- Erasing uses the **centroid of the three fingertips** (index, middle, ring), smoothed slightly while you move.

### Brush size

You can change size in several ways:

1. **Fist rotate:** With a **fist** (and not in eraser mode), **rotate your wrist**. Brush size tracks the change in orientation of the hand and updates the numeric size.
2. **Brush size panel:** **Pinch** (thumb and index close) with your **index tip on the brush icon** while the panel is closed to **open** the popover. **Move your index tip** outside the combined area of the icon and the popover to **close** it (unless you are pinching on the slider track).
3. With the panel open, **pinch** and move along the **horizontal slider** so the **midpoint** of the pinch scrubs the value (wide pinch span, not a tiny tap pinch—see `SIZE_BRUSH_PINCH_*` in code).

You can still use the **range input with the mouse** when the popover is open.

### Colors

- **Mouse:** click a swatch.
- **Hand:** **start a pinch** (thumb–index) while your **index tip** is over a swatch to pick that color for that hand.

With **two hands**, each hand can use its **own** last-selected color.

### Clear the canvas

1. **Button:** tap **×** in the HUD (clears the drawing layer).
2. **Gesture:** **Both** hands must hold the **“three straight” clear pose** at the same time (index, middle, ring extended upward in a stable configuration—see `threeStraightClearPose` in `src/main.ts`). The app clears after that pose is detected on **several consecutive frames** in a row (a very short simultaneous hold). If either hand drops the pose, the streak resets.

---

## Troubleshooting

- **No camera:** Check browser permissions; use **HTTPS** or **localhost** (browsers block camera on insecure origins except localhost).
- **Model or WASM fails to load:** Check firewall/ad-blockers and that the CDN and `storage.googleapis.com` URLs are reachable.
- **Hands not detected:** Improve lighting, keep hands in frame, and avoid motion blur.
- **GPU issues:** The app tries WebGPU first for MediaPipe and can fall back to CPU if creation fails (see `start()` in `src/main.ts`).

---

## Project structure (short)

- `index.html` — entry, mounts `#app`
- `src/main.ts` — camera, MediaPipe loop, gesture logic, drawing
- `src/style.css` — layout and HUD styling
- `vite.config.ts` — Vite configuration
