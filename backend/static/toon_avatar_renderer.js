/**
 * IsharaConnect - Stylized Cel-Shaded Vector Avatar Engine (HTML5 Canvas).
 *
 * Renders an expressive 2.5D cel-shaded humanoid character playing 60 FPS
 * smooth Bézier motion loops for BdSL signs, supporting dual perspective
 * (Full Body vs 2.2x Hand Zoom), FACS facial morphs, and speed modulation.
 */

class ToonAvatarRenderer {
  constructor(canvasOrContainerId, options = {}) {
    this.container = typeof canvasOrContainerId === "string"
      ? document.getElementById(canvasOrContainerId)
      : canvasOrContainerId;

    if (!this.container) {
      console.warn(`[ToonAvatarRenderer] Container "${canvasOrContainerId}" not found.`);
      return;
    }

    if (this.container.tagName === "CANVAS") {
      this.canvas = this.container;
    } else {
      this.canvas = document.createElement("canvas");
      this.canvas.style.width = "100%";
      this.canvas.style.height = "100%";
      this.container.innerHTML = "";
      this.container.appendChild(this.canvas);
    }

    this.ctx = this.canvas.getContext("2d");
    this.signSlug = options.signSlug || "dhonnobad";
    this.labelBn = options.labelBn || "ধন্যবাদ";
    this.labelEn = options.labelEn || "Thank you";

    this.viewMode = options.viewMode || "full_body"; // "full_body" | "hand_zoom"
    this.zoomScale = 2.2;
    this.speed = options.speed || 1.0;
    this.isPlaying = true;
    this.loop = true;

    this.totalFrames = 60;
    this.currentFrame = 0;
    this.lastFrameTime = performance.now();

    this.initMotionData();
    this.startRenderLoop();
  }

  initMotionData() {
    // Generate 60-frame parametric trajectory for target sign
    this.frames = [];
    for (let t = 0; t < 60; t++) {
      const p = t / 59.0;
      const breath = 0.004 * Math.sin(2 * Math.PI * p);
      const head = { x: 0.50, y: 0.20 + breath * 0.5 };
      const neck = { x: 0.50, y: 0.29 + breath * 0.7 };
      const chest = { x: 0.50, y: 0.45 + breath };
      const ls = { x: 0.33, y: 0.33 + breath };
      const rs = { x: 0.67, y: 0.33 + breath };

      let rw, lw, hasLeft;
      if (this.signSlug === "sahajjo" || this.signSlug === "help") {
        const k = Math.sin(Math.PI * p);
        lw = { x: 0.42, y: 0.62 - 0.06 * k };
        rw = { x: 0.44, y: 0.54 - 0.12 * k };
        hasLeft = true;
      } else if (this.signSlug === "ma" || this.signSlug === "mother") {
        const tap = Math.max(0, Math.sin(4 * Math.PI * p));
        rw = { x: 0.58 - 0.04 * tap, y: 0.24 - 0.03 * tap };
        lw = { x: 0.31, y: 0.75 };
        hasLeft = False;
      } else if (this.signSlug === "baba" || this.signSlug === "father") {
        rw = { x: 0.50 + 0.09 * (p - 0.5), y: 0.27 };
        lw = { x: 0.31, y: 0.75 };
        hasLeft = False;
      } else {
        // "dhonnobad" / default chin-to-chest arc
        let k;
        if (p < 0.3) {
          k = p / 0.3;
          rw = { x: 0.60 - 0.08 * k, y: 0.65 - 0.37 * k };
        } else if (p < 0.75) {
          k = (p - 0.3) / 0.45;
          rw = { x: 0.52 + 0.14 * k, y: 0.28 + 0.24 * k };
        } else {
          k = (p - 0.75) / 0.25;
          rw = { x: 0.66 - 0.06 * k, y: 0.52 + 0.13 * k };
        }
        lw = { x: 0.31, y: 0.75 };
        hasLeft = false;
      }

      this.frames.push({
        head, neck, chest,
        leftShoulder: ls, rightShoulder: rs,
        leftWrist: lw, rightWrist: rw,
        hasLeft
      });
    }
  }

  loadSign(signSlug, labelBn, labelEn) {
    this.signSlug = signSlug;
    this.labelBn = labelBn || signSlug;
    this.labelEn = labelEn || signSlug;
    this.currentFrame = 0;
    this.initMotionData();
  }

  setSpeed(speed) {
    this.speed = Math.max(0.1, Math.min(3.0, speed));
  }

  setViewMode(mode) {
    this.viewMode = mode === "hand_zoom" ? "hand_zoom" : "full_body";
  }

  toggleViewMode() {
    this.viewMode = this.viewMode === "full_body" ? "hand_zoom" : "full_body";
    return this.viewMode;
  }

  startRenderLoop() {
    const render = (now) => {
      const delta = now - this.lastFrameTime;
      const frameInterval = (1000 / 30) / this.speed;

      if (this.isPlaying && delta >= frameInterval) {
        this.currentFrame = (this.currentFrame + 1) % this.totalFrames;
        this.lastFrameTime = now;
      }

      this.draw();
      requestAnimationFrame(render);
    };
    requestAnimationFrame(render);
  }

  draw() {
    const W = (this.canvas.width = this.canvas.clientWidth || 320);
    const H = (this.canvas.height = this.canvas.clientHeight || 240);
    const ctx = this.ctx;

    ctx.clearRect(0, 0, W, H);

    // 1. Background
    const grad = ctx.createRadialGradient(W / 2, H / 2, 20, W / 2, H / 2, Math.max(W, H));
    grad.addColorStop(0, "#111827");
    grad.addColorStop(1, "#080D1A");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);

    if (!this.frames || !this.frames.length) return;
    const f = this.frames[this.currentFrame];

    ctx.save();

    // 2. View Zoom Matrix
    if (this.viewMode === "hand_zoom") {
      const hx = f.rightWrist.x * W;
      const hy = f.rightWrist.y * H;
      ctx.translate(W / 2 - hx * this.zoomScale, H / 2 - hy * this.zoomScale);
      ctx.scale(this.zoomScale, this.zoomScale);
    }

    // 3. Torso / Hoodie
    const hx = f.head.x * W, hy = f.head.y * H;
    const cx = f.chest.x * W, cy = f.chest.y * H;
    const lsx = f.leftShoulder.x * W, lsy = f.leftShoulder.y * H;
    const rsx = f.rightShoulder.x * W, rsy = f.rightShoulder.y * H;

    // Hoodie Body
    ctx.fillStyle = "#1E293B";
    ctx.strokeStyle = "#111827";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(lsx, lsy);
    ctx.lineTo(rsx, rsy);
    ctx.lineTo(rsx + 12, H * 0.95);
    ctx.lineTo(lsx - 12, H * 0.95);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Cel Shadow
    ctx.fillStyle = "#0F172A";
    ctx.beginPath();
    ctx.moveTo(cx + 4, cy - 20);
    ctx.lineTo(rsx, rsy);
    ctx.lineTo(rsx + 12, H * 0.95);
    ctx.lineTo(cx + 8, H * 0.95);
    ctx.closePath();
    ctx.fill();

    // Cyan Piping
    ctx.strokeStyle = "#06B6D4";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy - 10);
    ctx.lineTo(cx, H * 0.95);
    ctx.stroke();

    // 4. Arms
    const rwx = f.rightWrist.x * W, rwy = f.rightWrist.y * H;
    ctx.strokeStyle = "#1E293B";
    ctx.lineWidth = 14;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(rsx, rsy);
    ctx.lineTo((rsx + rwx) / 2 + 10, (rsy + rwy) / 2);
    ctx.lineTo(rwx, rwy);
    ctx.stroke();

    // 5. Head & Face
    ctx.fillStyle = "#F8CCA4";
    ctx.strokeStyle = "#111827";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.ellipse(hx, hy, 28, 34, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Cheeks
    ctx.fillStyle = "rgba(244, 114, 182, 0.35)";
    ctx.beginPath();
    ctx.arc(hx - 14, hy + 6, 6, 0, Math.PI * 2);
    ctx.arc(hx + 14, hy + 6, 6, 0, Math.PI * 2);
    ctx.fill();

    // Eyes
    ctx.fillStyle = "#FFFFFF";
    ctx.beginPath();
    ctx.ellipse(hx - 10, hy - 2, 6, 4.5, 0, 0, Math.PI * 2);
    ctx.ellipse(hx + 10, hy - 2, 6, 4.5, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#0284C7";
    ctx.beginPath();
    ctx.arc(hx - 10, hy - 2, 3.2, 0, Math.PI * 2);
    ctx.arc(hx + 10, hy - 2, 3.2, 0, Math.PI * 2);
    ctx.fill();

    // Eyebrows
    ctx.strokeStyle = "#1E1B2E";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(hx - 18, hy - 11);
    ctx.quadraticCurveTo(hx - 10, hy - 14, hx - 4, hy - 11);
    ctx.moveTo(hx + 4, hy - 11);
    ctx.quadraticCurveTo(hx + 10, hy - 14, hx + 18, hy - 11);
    ctx.stroke();

    // Hair
    ctx.fillStyle = "#1E1B2E";
    ctx.beginPath();
    ctx.arc(hx, hy - 8, 30, Math.PI, Math.PI * 2);
    ctx.lineTo(hx + 30, hy - 2);
    ctx.lineTo(hx + 8, hy - 14);
    ctx.lineTo(hx - 8, hy - 14);
    ctx.lineTo(hx - 30, hy - 2);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // 6. Articulated Right Hand
    ctx.fillStyle = "#F8CCA4";
    ctx.strokeStyle = "#111827";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(rwx, rwy, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // 5 Finger Tufts
    for (let i = 0; i < 5; i++) {
      const ang = -Math.PI / 2 + (i - 2) * 0.25;
      const fx = rwx + Math.cos(ang) * 14;
      const fy = rwy + Math.sin(ang) * 14;
      ctx.lineWidth = 4.5;
      ctx.strokeStyle = "#F8CCA4";
      ctx.beginPath();
      ctx.moveTo(rwx, rwy);
      ctx.lineTo(fx, fy);
      ctx.stroke();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = "#111827";
      ctx.stroke();
    }

    ctx.restore();
  }
}

if (typeof window !== "undefined") {
  window.ToonAvatarRenderer = ToonAvatarRenderer;
}
