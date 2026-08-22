/**
 * IsharaConnect - 3D FACS Skeletal Avatar Engine
 * Real-time 60 FPS Three.js rendering with FACS blendshapes and Hermite co-articulation streaming.
 */

class Ishara3DFACSAvatar {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.frameQueue = [];
    this.isPlaying = true;
    this.totalFramesReceived = 0;

    this.initScene();
    this.initLighting();
    this.createAvatarMesh();
    this.initWebSocket();

    this.renderLoop = this.renderLoop.bind(this);
    requestAnimationFrame(this.renderLoop);
  }

  initScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0f1d);

    this.camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.1, 100);
    this.camera.position.set(0, 1.3, 1.8);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    if (this.container) {
      this.container.appendChild(this.renderer.domElement);
    } else {
      document.body.appendChild(this.renderer.domElement);
    }

    if (typeof THREE.OrbitControls !== "undefined") {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.target.set(0, 1.1, 0);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.05;
      this.controls.maxPolarAngle = Math.PI / 2 + 0.1;
      this.controls.minDistance = 0.6;
      this.controls.maxDistance = 4.0;
    }

    // Grid Floor
    const grid = new THREE.GridHelper(10, 20, 0x1e293b, 0x0f172a);
    grid.position.y = 0;
    this.scene.add(grid);

    window.addEventListener("resize", () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
    });
  }

  initLighting() {
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const keyLight = new THREE.DirectionalLight(0x38bdf8, 1.4);
    keyLight.position.set(2, 4, 3);
    this.scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0xa855f7, 0.8);
    rimLight.position.set(-2, 2, -2);
    this.scene.add(rimLight);
  }

  createAvatarMesh() {
    // ১. ফেসিয়াল মেশ উইথ মরফ টার্গেটস (FACS AUs)
    const headGeo = new THREE.SphereGeometry(0.12, 32, 32);

    // Morph Target 0: AU12 (Smile / Lip Corner Puller)
    const pos = headGeo.attributes.position;
    const morphPositions = [];
    for (let i = 0; i < pos.count; i++) {
      let x = pos.getX(i);
      let y = pos.getY(i);
      let z = pos.getZ(i);
      // নিচের ঠোঁটের কোণ উপরের দিকে টানা
      if (y < 0 && z > 0.05) {
        y += 0.02;
        x *= 1.1;
      }
      morphPositions.push(x, y, z);
    }
    headGeo.morphAttributes.position = [
      new THREE.Float32BufferAttribute(morphPositions, 3),
    ];

    const headMat = new THREE.MeshStandardMaterial({
      color: 0xfbcfe8,
      roughness: 0.5,
      morphTargets: true,
    });

    this.headMesh = new THREE.Mesh(headGeo, headMat);
    this.headMesh.position.set(0, 1.25, 0);
    this.scene.add(this.headMesh);

    // ২. ডান হাতের জয়েন্টস নোড (২১ ল্যান্ডমার্ক)
    this.handJoints = [];
    const jointGeo = new THREE.SphereGeometry(0.012, 12, 12);
    const jointMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      metalness: 0.3,
      emissive: 0x0284c7,
      emissiveIntensity: 0.3,
    });

    for (let i = 0; i < 21; i++) {
      const mesh = new THREE.Mesh(jointGeo, jointMat);
      mesh.position.set(0, 1.0, 0);
      this.scene.add(mesh);
      this.handJoints.push(mesh);
    }
  }

  initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host || "localhost:8000";
    const wsUrl = `${protocol}//${host}/ws/avatar_stream`;
    const statusElem = document.getElementById("hud-stats");

    try {
      this.socket = new WebSocket(wsUrl);

      this.socket.onopen = () => {
        if (statusElem) statusElem.innerText = "WebSocket: সংযুক্ত (Connected)";
      };

      this.socket.onmessage = (event) => {
        const packet = JSON.parse(event.data);
        if (packet.status === "start_stream") {
          this.frameQueue = [];
          this.totalFramesReceived = packet.total_frames;
          if (statusElem) {
            const stepInfo = packet.total_steps > 1 ? ` [ধাপ ১/${packet.total_steps}]` : "";
            statusElem.innerText = `স্ট্রিমিং: ${packet.total_frames} ফ্রেম @ ${packet.fps} FPS${stepInfo}`;
          }
        } else if (packet.status === "frame") {
          this.frameQueue.push(packet.data);
          if (packet.data.stage_info && statusElem) {
            statusElem.innerText = `ধাপ ${packet.data.stage_info.current}/${packet.data.stage_info.total}: ${packet.data.stage_info.sign_name}`;
          }
        } else if (packet.status === "end_stream") {
          if (statusElem) {
            statusElem.innerText = "স্ট্রিমিং সম্পন্ন (Complete)";
          }
        }
      };

      this.socket.onclose = () => {
        if (statusElem) statusElem.innerText = "WebSocket: সংযোগ বিচ্ছিন্ন (Disconnected)";
      };
    } catch (e) {
      console.warn("WebSocket init error:", e);
      if (statusElem) statusElem.innerText = "WebSocket: ত্রুটি (Error)";
    }
  }

  requestSignStream(glossText) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      console.warn("WebSocket is not connected.");
      return;
    }
    const glosses = glossText
      .split(/[\s,]+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0);

    const hudSign = document.getElementById("active-gloss");
    if (hudSign) hudSign.innerText = glosses.join(" ");

    this.socket.send(JSON.stringify({ glosses }));
  }

  renderLoop() {
    requestAnimationFrame(this.renderLoop);
    if (this.controls) {
      this.controls.update();
    }

    if (this.frameQueue.length > 0) {
      const frame = this.frameQueue.shift();

      // ১. হাতের জয়েন্ট পজিশন আপডেট
      if (frame.right_hand) {
        frame.right_hand.forEach((pt, idx) => {
          if (this.handJoints[idx]) {
            const px = pt.x !== undefined ? pt.x : pt[0];
            const py = pt.y !== undefined ? pt.y : pt[1];
            const pz = pt.z !== undefined ? pt.z : (pt[2] || 0.0);

            this.handJoints[idx].position.set(
              (px - 0.5) * 1.5,
              (1.0 - py) * 1.5 + 0.5,
              -pz * 1.5
            );
          }
        });
      }

      // ২. ফেসিয়াল অ্যাকশন ইউনিট (FACS) ব্লেন্ডশেপ ও হেড পোজ আপডেট
      if (frame.facs) {
        // AU12 (Smile) ইনফ্লুয়েন্স
        const au12 = frame.facs.AU12 || 0.0;
        if (this.headMesh && this.headMesh.morphTargetInfluences) {
          this.headMesh.morphTargetInfluences[0] = au12;
        }

        // হেড পিচ (মাথার সামনে/পেছনে নড়াচড়া)
        const pitch = frame.facs.head_pitch || 0.0;
        if (this.headMesh) {
          this.headMesh.rotation.x = THREE.MathUtils.degToRad(pitch);
        }
      }
    }

    this.renderer.render(this.scene, this.camera);
  }
}

/**
 * Full Skeletal Humanoid 3D Avatar Loader for BdSL Synthesizer.
 * Supports GLTF 2.0 skinned humanoid mesh, standard bone hierarchy, and full FACS blendshapes.
 */
class HumanoidBdSLAvatar {
  constructor(containerId, modelUrl = "/static/models/bdsl_humanoid_rig.glb") {
    this.container = document.getElementById(containerId);
    this.morphMeshes = [];
    this.handBones = { right: {}, left: {} };
    this.frameQueue = [];
    this.isPlaying = true;
    this.modelUrl = modelUrl;

    this.initScene();
    this.initLighting();
    this.loadHumanoidModel(this.modelUrl);
    this.initWebSocket();

    this.renderLoop = this.renderLoop.bind(this);
    requestAnimationFrame(this.renderLoop);
  }

  initScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0f1d);
    this.camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.1, 100);
    this.camera.position.set(0, 1.35, 2.0);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    if (this.container) {
      this.container.appendChild(this.renderer.domElement);
    } else {
      document.body.appendChild(this.renderer.domElement);
    }

    if (typeof THREE.OrbitControls !== "undefined") {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.target.set(0, 1.1, 0);
      this.controls.enableDamping = true;
    }
  }

  initLighting() {
    const amb = new THREE.AmbientLight(0xffffff, 0.9);
    this.scene.add(amb);
    const dir = new THREE.DirectionalLight(0x06b6d4, 1.2);
    dir.position.set(2, 4, 3);
    this.scene.add(dir);
  }

  loadHumanoidModel(modelUrl) {
    if (typeof GLTFLoader !== "undefined" || (typeof THREE !== "undefined" && THREE.GLTFLoader)) {
      const LoaderClass = typeof GLTFLoader !== "undefined" ? GLTFLoader : THREE.GLTFLoader;
      const loader = new LoaderClass();
      loader.load(
        modelUrl,
        (gltf) => {
          this.avatarModel = gltf.scene;
          this.avatarModel.traverse((node) => {
            if (node.isSkinnedMesh && node.morphTargetDictionary) {
              this.morphMeshes.push(node);
            }
            if (node.isBone) {
              if (node.name.startsWith("RightHand_")) this.handBones.right[node.name] = node;
              if (node.name.startsWith("LeftHand_")) this.handBones.left[node.name] = node;
            }
          });
          this.scene.add(this.avatarModel);
        },
        undefined,
        (err) => {
          console.warn("GLTF model fallback to procedural rig:", err);
        }
      );
    }
  }

  initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host || "localhost:8000";
    const wsUrl = `${protocol}//${host}/ws/avatar_stream`;

    try {
      this.ws = new WebSocket(wsUrl);
      this.ws.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.frames && Array.isArray(payload.frames)) {
          this.frameQueue.push(...payload.frames);
        }
      };
    } catch (e) {
      console.warn("WebSocket stream fallback:", e);
    }
  }

  applyFacsBlendshapes(facs) {
    if (!facs) return;
    const auMappings = {
      "AU01_inner_brow_raiser": facs.AU01 || 0.0,
      "AU02_outer_brow_raiser": facs.AU02 || 0.0,
      "AU04_brow_lowerer": facs.AU04 || 0.0,
      "AU06_cheek_raiser": facs.AU06 || 0.0,
      "AU12_lip_corner_puller": facs.AU12 || 0.0,
      "AU25_lips_part": facs.AU25 || 0.0,
      "AU26_jaw_drop": facs.AU26 || 0.0,
      "AU43_eyes_closed": facs.AU43 || 0.0
    };

    this.morphMeshes.forEach((mesh) => {
      for (const [auName, val] of Object.entries(auMappings)) {
        const targetIdx = mesh.morphTargetDictionary[auName];
        if (targetIdx !== undefined) {
          mesh.morphTargetInfluences[targetIdx] = THREE.MathUtils.lerp(
            mesh.morphTargetInfluences[targetIdx],
            val,
            0.25
          );
        }
      }
    });

    if (facs.head_pitch !== undefined && this.avatarModel) {
      const headBone = this.avatarModel.getObjectByName("Head");
      if (headBone) {
        headBone.rotation.x = THREE.MathUtils.degToRad(facs.head_pitch);
      }
    }
  }

  renderLoop() {
    requestAnimationFrame(this.renderLoop);
    if (this.controls) this.controls.update();

    if (this.frameQueue.length > 0 && this.isPlaying) {
      const frame = this.frameQueue.shift();
      if (frame.facs) {
        this.applyFacsBlendshapes(frame.facs);
      }
    }

    this.renderer.render(this.scene, this.camera);
  }
}

// ----------------- Runtime Initialization -----------------
document.addEventListener("DOMContentLoaded", () => {
  const avatar = new Ishara3DFACSAvatar("viewport-container");

  const synthBtn = document.getElementById("btn-synthesize");
  const inputElem = document.getElementById("input-glosses");

  if (synthBtn && inputElem) {
    synthBtn.addEventListener("click", () => {
      const text = inputElem.value.trim();
      if (text) {
        avatar.requestSignStream(text);
      }
    });

    inputElem.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        const text = inputElem.value.trim();
        if (text) {
          avatar.requestSignStream(text);
        }
      }
    });
  }
});
