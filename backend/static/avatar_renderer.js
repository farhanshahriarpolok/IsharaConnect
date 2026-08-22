/**
 * IsharaConnect - 3D Avatar Viewport Engine
 * Renders real-time 3D skeletal landmarks and sign animations using Three.js & OrbitControls.
 */

class Ishara3DAvatarEngine {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.frames = [];
    this.currentFrameIdx = 0;
    this.isPlaying = true;
    this.fps = 30;
    this.lastFrameTime = 0;

    this.initScene();
    this.initLighting();
    this.buildSkeletalMesh();
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }

  initScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x090d16);

    // Camera setup
    this.camera = new THREE.PerspectiveCamera(
      45,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    this.camera.position.set(0, 1.2, 2.2);

    // WebGL Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    if (this.container) {
      this.container.appendChild(this.renderer.domElement);
    } else {
      document.body.appendChild(this.renderer.domElement);
    }

    // Interactive OrbitControls
    if (typeof THREE.OrbitControls !== "undefined") {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.target.set(0, 1.0, 0);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.05;
      this.controls.maxPolarAngle = Math.PI / 2 + 0.1;
      this.controls.minDistance = 0.8;
      this.controls.maxDistance = 6.0;
    }

    // Grid Floor Helper
    const grid = new THREE.GridHelper(10, 20, 0x1e293b, 0x0f172a);
    grid.position.y = 0;
    this.scene.add(grid);

    window.addEventListener("resize", () => this.onResize());
  }

  initLighting() {
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    this.scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0x38bdf8, 1.2);
    dirLight.position.set(2, 4, 3);
    dirLight.castShadow = true;
    this.scene.add(dirLight);

    const fillLight = new THREE.DirectionalLight(0xa855f7, 0.6);
    fillLight.position.set(-2, 2, -2);
    this.scene.add(fillLight);
  }

  buildSkeletalMesh() {
    this.jointMeshes = [];

    // Joint Spheres Material
    const jointGeo = new THREE.SphereGeometry(0.015, 16, 16);
    const jointMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      roughness: 0.3,
      metalness: 0.2,
      emissive: 0x0284c7,
      emissiveIntensity: 0.2,
    });

    // Create nodes for 75 landmarks (Pose 33 + Left Hand 21 + Right Hand 21)
    for (let i = 0; i < 75; i++) {
      const mesh = new THREE.Mesh(jointGeo, jointMat);
      mesh.visible = false;
      this.scene.add(mesh);
      this.jointMeshes.push(mesh);
    }
  }

  loadLandmarks(landmarksPayload, glossName = "BdSL Sign") {
    const hudSign = document.getElementById("active-gloss");
    if (hudSign) {
      hudSign.innerText = glossName;
    }
    this.frames = landmarksPayload.frames || [];
    this.fps = landmarksPayload.fps || 30;
    this.currentFrameIdx = 0;
  }

  updateFrame(frameIdx) {
    if (!this.frames || this.frames.length === 0) return;
    const frame = this.frames[frameIdx];

    // Right Hand and Upper Body joint mapping
    if (frame.right_hand) {
      frame.right_hand.forEach((pt, i) => {
        const mesh = this.jointMeshes[i];
        if (mesh) {
          const px = pt.x !== undefined ? pt.x : pt[0];
          const py = pt.y !== undefined ? pt.y : pt[1];
          const pz = pt.z !== undefined ? pt.z : (pt[2] || 0.0);

          // 2D/3D space normalization to Three.js world coordinates
          mesh.position.set((px - 0.5) * 1.5, (1.0 - py) * 1.5 + 0.5, -pz * 1.5);
          mesh.visible = true;
        }
      });
    }

    if (frame.pose) {
      frame.pose.forEach((pt, i) => {
        const mesh = this.jointMeshes[21 + i];
        if (mesh) {
          const px = pt.x !== undefined ? pt.x : pt[0];
          const py = pt.y !== undefined ? pt.y : pt[1];
          const pz = pt.z !== undefined ? pt.z : (pt[2] || 0.0);
          mesh.position.set((px - 0.5) * 1.5, (1.0 - py) * 1.5 + 0.5, -pz * 1.5);
          mesh.visible = true;
        }
      });
    }
  }

  animate(timestamp) {
    requestAnimationFrame(this.animate);
    if (this.controls) {
      this.controls.update();
    }

    if (this.isPlaying && this.frames.length > 0) {
      if (timestamp - this.lastFrameTime > 1000 / this.fps) {
        this.updateFrame(this.currentFrameIdx);
        this.currentFrameIdx = (this.currentFrameIdx + 1) % this.frames.length;
        this.lastFrameTime = timestamp;
      }
    }

    this.renderer.render(this.scene, this.camera);
  }

  onResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }
}

// ----------------- Runtime Initialization -----------------
document.addEventListener("DOMContentLoaded", () => {
  const engine = new Ishara3DAvatarEngine("viewport-container");

  // Load landmark data from API / static cache
  fetch("/static/data/dhonnobad.json")
    .then((res) => res.json())
    .then((data) => {
      engine.loadLandmarks(data, "ধন্যবাদ (Dhonnobad)");
    })
    .catch(() => {
      console.log("Using synthetic procedural keypoints...");
      const syntheticFrames = [];
      for (let i = 0; i < 30; i++) {
        const t = i / 30;
        const arc = Math.sin(t * Math.PI);
        syntheticFrames.push({
          right_hand: [
            { x: 0.5 + arc * 0.1, y: 0.3 + arc * 0.15, z: arc * 0.05 },
            { x: 0.51 + arc * 0.1, y: 0.28 + arc * 0.15, z: arc * 0.05 },
            { x: 0.52 + arc * 0.1, y: 0.26 + arc * 0.15, z: arc * 0.05 },
            { x: 0.52 + arc * 0.1, y: 0.24 + arc * 0.15, z: arc * 0.05 },
            { x: 0.52 + arc * 0.1, y: 0.22 + arc * 0.15, z: arc * 0.05 }
          ]
        });
      }
      engine.loadLandmarks({ frames: syntheticFrames, fps: 30 }, "ধন্যবাদ (Dhonnobad)");
    });

  const playBtn = document.getElementById("btn-play");
  if (playBtn) {
    playBtn.addEventListener("click", (e) => {
      engine.isPlaying = !engine.isPlaying;
      e.target.innerText = engine.isPlaying ? "Pause" : "Play";
    });
  }

  const resetBtn = document.getElementById("btn-reset");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      engine.currentFrameIdx = 0;
      engine.updateFrame(0);
    });
  }
});
