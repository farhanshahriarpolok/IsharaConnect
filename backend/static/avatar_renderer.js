/**
 * IsharaConnect - 3D Human Avatar & Skeletal Rig Renderer
 * Built with Three.js for real-time BdSL gesture synthesis and landmark replay.
 */

(function () {
  let scene, camera, renderer, controls;
  let isPlaying = true;
  let animTime = 0;
  let currentSignIdx = 0;

  const signs = [
    { bn: "ধন্যবাদ", en: "Thank You", slug: "dhonnobad", duration: 2.2 },
    { bn: "স্বাগতম", en: "Welcome", slug: "shagotom", duration: 2.5 },
    { bn: "আমি", en: "I / Me", slug: "ami", duration: 1.8 },
    { bn: "কেমন আছেন", en: "How are you?", slug: "kemon_achen", duration: 2.4 }
  ];

  // Rig Hierarchy
  const avatar = new THREE.Group();
  let head, torso, rightArm, rightForearm, rightHand, leftArm, leftForearm, leftHand;
  const rightFingers = [];
  const leftFingers = [];

  function init() {
    const container = document.getElementById("viewport-container");
    const width = window.innerWidth;
    const height = window.innerHeight;

    // 1. Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x090d16);
    scene.fog = new THREE.FogExp2(0x090d16, 0.12);

    // 2. Camera
    camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(0, 1.4, 2.4);

    // 3. Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    // 4. Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 + 0.1;
    controls.minDistance = 1.0;
    controls.maxDistance = 5.0;
    controls.target.set(0, 1.2, 0);

    // 5. Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(2, 4, 3);
    dirLight.castShadow = true;
    scene.add(dirLight);

    const rimLight = new THREE.PointLight(0x38bdf8, 2.0, 8);
    rimLight.position.set(-2, 2, -1);
    scene.add(rimLight);

    const floorGrid = new THREE.GridHelper(10, 20, 0x334155, 0x1e293b);
    floorGrid.position.y = 0;
    scene.add(floorGrid);

    // 6. Build 3D Avatar
    buildAvatarRig();
    scene.add(avatar);

    // 7. Event Listeners
    window.addEventListener("resize", onWindowResize);
    document.getElementById("btn-play").addEventListener("click", togglePlay);
    document.getElementById("btn-reset").addEventListener("click", resetPose);

    updateHUD();
    animate();
  }

  function buildAvatarRig() {
    const skinMat = new THREE.MeshStandardMaterial({
      color: 0xfbcfe8,
      roughness: 0.4,
      metalness: 0.1
    });

    const clothMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      roughness: 0.6,
      metalness: 0.2
    });

    const jointMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      roughness: 0.2,
      metalness: 0.8
    });

    // Torso / Chest
    const torsoGeo = new THREE.CylinderGeometry(0.24, 0.18, 0.65, 16);
    torso = new THREE.Mesh(torsoGeo, clothMat);
    torso.position.set(0, 1.0, 0);
    avatar.add(torso);

    // Head
    const headGeo = new THREE.SphereGeometry(0.14, 24, 24);
    head = new THREE.Mesh(headGeo, skinMat);
    head.position.set(0, 1.52, 0);
    avatar.add(head);

    // Right Arm Hierarchy
    const rShoulder = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 12), jointMat);
    rShoulder.position.set(0.28, 0.26, 0);
    torso.add(rShoulder);

    rightArm = new THREE.Group();
    rShoulder.add(rightArm);
    const rUpperArm = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.04, 0.32, 12), clothMat);
    rUpperArm.position.set(0, -0.16, 0);
    rightArm.add(rUpperArm);

    const rElbow = new THREE.Mesh(new THREE.SphereGeometry(0.05, 12, 12), jointMat);
    rElbow.position.set(0, -0.32, 0);
    rightArm.add(rElbow);

    rightForearm = new THREE.Group();
    rElbow.add(rightForearm);
    const rLowerArm = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.035, 0.28, 12), skinMat);
    rLowerArm.position.set(0, -0.14, 0);
    rightForearm.add(rLowerArm);

    rightHand = new THREE.Group();
    rightHand.position.set(0, -0.28, 0);
    rightForearm.add(rightHand);
    const rPalm = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.09, 0.025), skinMat);
    rightHand.add(rPalm);

    // Build Right Fingers (5 fingers)
    for (let f = 0; f < 5; f++) {
      const finger = new THREE.Group();
      finger.position.set((f - 2) * 0.016, -0.05, 0);
      const fMesh = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.05, 0.015), skinMat);
      fMesh.position.set(0, -0.025, 0);
      finger.add(fMesh);
      rightHand.add(finger);
      rightFingers.push(finger);
    }

    // Left Arm Hierarchy
    const lShoulder = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 12), jointMat);
    lShoulder.position.set(-0.28, 0.26, 0);
    torso.add(lShoulder);

    leftArm = new THREE.Group();
    lShoulder.add(leftArm);
    const lUpperArm = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.04, 0.32, 12), clothMat);
    lUpperArm.position.set(0, -0.16, 0);
    leftArm.add(lUpperArm);

    const lElbow = new THREE.Mesh(new THREE.SphereGeometry(0.05, 12, 12), jointMat);
    lElbow.position.set(0, -0.32, 0);
    leftArm.add(lElbow);

    leftForearm = new THREE.Group();
    lElbow.add(leftForearm);
    const lLowerArm = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.035, 0.28, 12), skinMat);
    lLowerArm.position.set(0, -0.14, 0);
    leftForearm.add(lLowerArm);

    leftHand = new THREE.Group();
    leftHand.position.set(0, -0.28, 0);
    leftForearm.add(leftHand);
    const lPalm = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.09, 0.025), skinMat);
    leftHand.add(lPalm);

    // Build Left Fingers
    for (let f = 0; f < 5; f++) {
      const finger = new THREE.Group();
      finger.position.set((f - 2) * 0.016, -0.05, 0);
      const fMesh = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.05, 0.015), skinMat);
      fMesh.position.set(0, -0.025, 0);
      finger.add(fMesh);
      leftHand.add(finger);
      leftFingers.push(finger);
    }
  }

  function updateHUD() {
    const current = signs[currentSignIdx];
    const hudSign = document.getElementById("active-gloss");
    if (hudSign) {
      hudSign.innerText = `${current.bn} (${current.en})`;
    }
  }

  function togglePlay() {
    isPlaying = !isPlaying;
    const btn = document.getElementById("btn-play");
    if (btn) {
      btn.innerText = isPlaying ? "Pause" : "Play";
    }
  }

  function resetPose() {
    animTime = 0;
    currentSignIdx = 0;
    updateHUD();
    rightArm.rotation.set(0, 0, 0);
    rightForearm.rotation.set(0, 0, 0);
    rightHand.rotation.set(0, 0, 0);
    leftArm.rotation.set(0, 0, 0);
    leftForearm.rotation.set(0, 0, 0);
    leftHand.rotation.set(0, 0, 0);
  }

  function animateGesture(delta) {
    if (!isPlaying) return;

    animTime += delta;
    const current = signs[currentSignIdx];
    if (animTime > current.duration) {
      animTime = 0;
      currentSignIdx = (currentSignIdx + 1) % signs.length;
      updateHUD();
    }

    const t = animTime / current.duration;
    const ease = Math.sin(t * Math.PI); // Easing curve

    // Idle breathing
    torso.position.y = 1.0 + Math.sin(Date.now() * 0.002) * 0.01;

    if (current.slug === "dhonnobad") {
      // Dhonnobad: Right hand starts at chin and moves forward
      rightArm.rotation.x = -0.6 - ease * 0.5;
      rightArm.rotation.z = -0.2;
      rightForearm.rotation.x = -1.5 + ease * 0.8;
      rightHand.rotation.x = -0.3 + ease * 0.4;
      leftArm.rotation.set(0, 0, 0.1);
      leftForearm.rotation.set(0, 0, 0);
    } else if (current.slug === "shagotom") {
      // Shagotom: Both hands open sweep inwards
      rightArm.rotation.x = -0.5 - ease * 0.3;
      rightArm.rotation.z = -0.4 + ease * 0.3;
      rightForearm.rotation.x = -1.2;
      leftArm.rotation.x = -0.5 - ease * 0.3;
      leftArm.rotation.z = 0.4 - ease * 0.3;
      leftForearm.rotation.x = -1.2;
    } else if (current.slug === "ami") {
      // Ami: Index finger points to chest
      rightArm.rotation.x = -0.7 * ease;
      rightForearm.rotation.x = -1.6 * ease;
      rightHand.rotation.y = 0.5 * ease;
    } else {
      // Conversational
      rightArm.rotation.x = -0.4 - ease * 0.4;
      rightForearm.rotation.x = -1.0 + Math.sin(t * Math.PI * 2) * 0.3;
      leftArm.rotation.x = -0.4 - ease * 0.4;
      leftForearm.rotation.x = -1.0 + Math.sin(t * Math.PI * 2) * 0.3;
    }
  }

  function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }

  let lastTime = performance.now();
  function animate() {
    requestAnimationFrame(animate);

    const now = performance.now();
    const delta = (now - lastTime) / 1000;
    lastTime = now;

    animateGesture(delta);
    controls.update();
    renderer.render(scene, camera);
  }

  // Initialize on DOM load
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
