import {
  FilesetResolver,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs";

const MEDIAPIPE_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";

const video = document.querySelector("#camera");
const canvas = document.querySelector("#overlay");
const ctx = canvas.getContext("2d");
const stage = document.querySelector("#stage");
const emptyState = document.querySelector("#emptyState");
const startButton = document.querySelector("#startButton");
const cameraButton = document.querySelector("#cameraButton");
const resetButton = document.querySelector("#resetButton");
const statusText = document.querySelector("#statusText");
const systemBadge = document.querySelector("#systemBadge");

const ui = {
  trackingQuality: document.querySelector("#trackingQuality"),
  trackingDetail: document.querySelector("#trackingDetail"),
  leftKnee: document.querySelector("#leftKnee"),
  rightKnee: document.querySelector("#rightKnee"),
  symmetry: document.querySelector("#symmetry"),
  torsoLean: document.querySelector("#torsoLean"),
  phase: document.querySelector("#phase"),
  phaseDetail: document.querySelector("#phaseDetail"),
  repCount: document.querySelector("#repCount"),
  fps: document.querySelector("#fps"),
  latency: document.querySelector("#latency"),
  coverage: document.querySelector("#coverage"),
  coverageDetail: document.querySelector("#coverageDetail"),
};

const LANDMARK = {
  leftShoulder: 11,
  rightShoulder: 12,
  leftHip: 23,
  rightHip: 24,
  leftKnee: 25,
  rightKnee: 26,
  leftAnkle: 27,
  rightAnkle: 28,
  leftHeel: 29,
  rightHeel: 30,
  leftFoot: 31,
  rightFoot: 32,
};

const CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [27, 29], [29, 31], [27, 31],
  [24, 26], [26, 28], [28, 30], [30, 32], [28, 32],
];

const KEY_JOINTS = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];

let poseLandmarker = null;
let mediaStream = null;
let animationId = null;
let running = false;
let usingFrontCamera = true;
let lastVideoTime = -1;
let lastProcessedAt = 0;
let lastFpsUpdate = performance.now();
let processedFrames = 0;
let displayedFps = 0;
let smoothedLeftKnee = null;
let smoothedRightKnee = null;
let previousAverageKnee = null;
let movementTrend = 0;
let squatState = "standing";
let repCount = 0;

const PROCESS_INTERVAL_MS = 70;
const VISIBILITY_REQUIRED = 0.55;
const STANDING_ANGLE = 155;
const BOTTOM_ANGLE = 105;

startButton.addEventListener("click", async () => {
  if (running) {
    stopCamera();
    return;
  }
  await startAssessment();
});

cameraButton.addEventListener("click", async () => {
  usingFrontCamera = !usingFrontCamera;
  cameraButton.textContent = usingFrontCamera ? "Use rear camera" : "Use front camera";
  stage.classList.toggle("mirror", usingFrontCamera);
  if (running) await restartCamera();
});

resetButton.addEventListener("click", resetCounter);
window.addEventListener("pagehide", stopCamera);

async function startAssessment() {
  try {
    setBadge("Loading", "loading");
    startButton.disabled = true;
    statusText.textContent = "Loading the on-device pose model…";

    if (!poseLandmarker) {
      poseLandmarker = await createLandmarker();
    }

    statusText.textContent = "Requesting camera permission…";
    await startCamera();

    running = true;
    startButton.disabled = false;
    startButton.textContent = "Stop assessment";
    cameraButton.disabled = false;
    emptyState.hidden = true;
    setBadge("Tracking", "active");
    statusText.textContent = "Tracking locally on this device. Keep your full body visible.";
    predictLoop();
  } catch (error) {
    console.error(error);
    startButton.disabled = false;
    setBadge("Error", "error");
    statusText.textContent = friendlyError(error);
  }
}

async function createLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);

  const commonOptions = {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.55,
    minPosePresenceConfidence: 0.55,
    minTrackingConfidence: 0.55,
    outputSegmentationMasks: false,
  };

  try {
    return await PoseLandmarker.createFromOptions(vision, commonOptions);
  } catch (gpuError) {
    console.warn("GPU setup failed; falling back to CPU.", gpuError);
    return PoseLandmarker.createFromOptions(vision, {
      ...commonOptions,
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: "CPU",
      },
    });
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Camera access is not available. Open this site in Safari over HTTPS.");
  }

  stopTracksOnly();

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: usingFrontCamera ? "user" : { ideal: "environment" },
      width: { ideal: 720 },
      height: { ideal: 1280 },
      frameRate: { ideal: 30, max: 30 },
    },
  });

  video.srcObject = mediaStream;
  await video.play();

  if (!video.videoWidth || !video.videoHeight) {
    await new Promise((resolve) => video.addEventListener("loadedmetadata", resolve, { once: true }));
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  lastVideoTime = -1;
}

async function restartCamera() {
  running = false;
  if (animationId) cancelAnimationFrame(animationId);
  await startCamera();
  running = true;
  predictLoop();
}

function stopCamera() {
  running = false;
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;
  stopTracksOnly();
  video.srcObject = null;
  clearCanvas();
  emptyState.hidden = false;
  startButton.disabled = false;
  startButton.textContent = "Start assessment";
  cameraButton.disabled = true;
  setBadge("Idle", "idle");
  statusText.textContent = "Camera stopped. Your video was processed locally and was not uploaded by this prototype.";
}

function stopTracksOnly() {
  if (!mediaStream) return;
  for (const track of mediaStream.getTracks()) track.stop();
  mediaStream = null;
}

function predictLoop(now = performance.now()) {
  if (!running) return;

  if (
    poseLandmarker &&
    video.readyState >= 2 &&
    video.currentTime !== lastVideoTime &&
    now - lastProcessedAt >= PROCESS_INTERVAL_MS
  ) {
    lastProcessedAt = now;
    lastVideoTime = video.currentTime;

    const inferenceStart = performance.now();
    const result = poseLandmarker.detectForVideo(video, now);
    const inferenceMs = performance.now() - inferenceStart;

    processedFrames += 1;
    if (now - lastFpsUpdate >= 1000) {
      displayedFps = processedFrames * 1000 / (now - lastFpsUpdate);
      processedFrames = 0;
      lastFpsUpdate = now;
    }

    handleResult(result, inferenceMs);
  }

  animationId = requestAnimationFrame(predictLoop);
}

function handleResult(result, inferenceMs) {
  clearCanvas();

  const landmarks = result.landmarks?.[0];
  if (!landmarks) {
    showNoPose(inferenceMs);
    return;
  }

  drawPose(landmarks);

  const keyLandmarks = KEY_JOINTS.map((index) => landmarks[index]);
  const averageVisibility = average(keyLandmarks.map((point) => point.visibility ?? 0));
  const missing = KEY_JOINTS.filter((index) => (landmarks[index].visibility ?? 0) < VISIBILITY_REQUIRED);
  const hasCoverage = missing.length === 0;

  const leftKneeRaw = angleDegrees(
    landmarks[LANDMARK.leftHip],
    landmarks[LANDMARK.leftKnee],
    landmarks[LANDMARK.leftAnkle],
  );
  const rightKneeRaw = angleDegrees(
    landmarks[LANDMARK.rightHip],
    landmarks[LANDMARK.rightKnee],
    landmarks[LANDMARK.rightAnkle],
  );

  smoothedLeftKnee = exponentialSmooth(smoothedLeftKnee, leftKneeRaw, 0.28);
  smoothedRightKnee = exponentialSmooth(smoothedRightKnee, rightKneeRaw, 0.28);

  const averageKnee = (smoothedLeftKnee + smoothedRightKnee) / 2;
  if (previousAverageKnee !== null) {
    movementTrend = exponentialSmooth(movementTrend, averageKnee - previousAverageKnee, 0.35);
  }
  previousAverageKnee = averageKnee;

  const shoulderMidpoint = midpoint(landmarks[LANDMARK.leftShoulder], landmarks[LANDMARK.rightShoulder]);
  const hipMidpoint = midpoint(landmarks[LANDMARK.leftHip], landmarks[LANDMARK.rightHip]);
  const torsoLean = angleFromVertical(shoulderMidpoint, hipMidpoint);
  const symmetry = Math.abs(smoothedLeftKnee - smoothedRightKnee);

  let phase = "Standing";
  let phaseDetail = "Ready for a repetition";

  if (!hasCoverage || averageVisibility < VISIBILITY_REQUIRED) {
    phase = "Reposition";
    phaseDetail = "Keep hips, knees, ankles, and feet in frame";
  } else {
    ({ phase, detail: phaseDetail } = updateSquatCounter(averageKnee, movementTrend));
  }

  const qualityPercent = Math.round(averageVisibility * 100);
  ui.trackingQuality.textContent = `${qualityPercent}%`;
  ui.trackingDetail.textContent = qualityPercent >= 80
    ? "Strong landmark visibility"
    : qualityPercent >= 60
      ? "Usable, but improve lighting or camera position"
      : "Low confidence—reposition before trusting angles";

  ui.leftKnee.textContent = formatDegrees(smoothedLeftKnee);
  ui.rightKnee.textContent = formatDegrees(smoothedRightKnee);
  ui.symmetry.textContent = formatDegrees(symmetry);
  ui.torsoLean.textContent = formatDegrees(torsoLean);
  ui.phase.textContent = phase;
  ui.phaseDetail.textContent = phaseDetail;
  ui.repCount.textContent = String(repCount);
  ui.fps.textContent = `${displayedFps.toFixed(1)} FPS`;
  ui.latency.textContent = `${inferenceMs.toFixed(0)} ms per analyzed frame`;
  ui.coverage.textContent = hasCoverage ? "Complete" : "Partial";
  ui.coverageDetail.textContent = hasCoverage
    ? "All required joints are visible"
    : `${missing.length} required landmark${missing.length === 1 ? " is" : "s are"} low visibility`;
}

function updateSquatCounter(kneeAngle, trend) {
  if (squatState === "standing") {
    if (kneeAngle < BOTTOM_ANGLE) {
      squatState = "bottom";
      return { phase: "Bottom", detail: "Stand back up to complete the rep" };
    }

    if (trend < -0.22 && kneeAngle < STANDING_ANGLE - 5) {
      return { phase: "Descending", detail: "Controlled downward movement" };
    }

    return { phase: "Standing", detail: "Bend knees and hips to begin" };
  }

  if (squatState === "bottom") {
    if (kneeAngle > STANDING_ANGLE) {
      repCount += 1;
      squatState = "standing";
      return { phase: "Rep complete", detail: `Completed repetition ${repCount}` };
    }

    if (trend > 0.22) {
      return { phase: "Ascending", detail: "Return to a tall standing position" };
    }

    return { phase: "Bottom", detail: "Stand back up to complete the rep" };
  }

  return { phase: "Tracking", detail: "Movement detected" };
}

function showNoPose(inferenceMs) {
  ui.trackingQuality.textContent = "0%";
  ui.trackingDetail.textContent = "No full-body pose detected";
  ui.leftKnee.textContent = "—";
  ui.rightKnee.textContent = "—";
  ui.symmetry.textContent = "—";
  ui.torsoLean.textContent = "—";
  ui.phase.textContent = "Reposition";
  ui.phaseDetail.textContent = "Move farther away and improve lighting";
  ui.fps.textContent = `${displayedFps.toFixed(1)} FPS`;
  ui.latency.textContent = `${inferenceMs.toFixed(0)} ms per analyzed frame`;
  ui.coverage.textContent = "None";
  ui.coverageDetail.textContent = "No pose landmarks available";
}

function drawPose(landmarks) {
  ctx.save();
  ctx.lineWidth = Math.max(3, canvas.width / 220);
  ctx.lineCap = "round";
  ctx.strokeStyle = "rgba(124, 156, 255, 0.9)";

  for (const [startIndex, endIndex] of CONNECTIONS) {
    const start = landmarks[startIndex];
    const end = landmarks[endIndex];
    if ((start.visibility ?? 0) < 0.35 || (end.visibility ?? 0) < 0.35) continue;

    ctx.beginPath();
    ctx.moveTo(start.x * canvas.width, start.y * canvas.height);
    ctx.lineTo(end.x * canvas.width, end.y * canvas.height);
    ctx.stroke();
  }

  for (const index of KEY_JOINTS) {
    const point = landmarks[index];
    const visibility = point.visibility ?? 0;
    if (visibility < 0.25) continue;

    ctx.beginPath();
    ctx.arc(
      point.x * canvas.width,
      point.y * canvas.height,
      Math.max(5, canvas.width / 130),
      0,
      Math.PI * 2,
    );
    ctx.fillStyle = visibility >= VISIBILITY_REQUIRED ? "#4fe0a4" : "#ffd166";
    ctx.fill();
    ctx.strokeStyle = "rgba(5, 11, 24, 0.85)";
    ctx.lineWidth = Math.max(2, canvas.width / 400);
    ctx.stroke();
  }

  ctx.restore();
}

function angleDegrees(a, b, c) {
  const ba = { x: a.x - b.x, y: a.y - b.y, z: (a.z ?? 0) - (b.z ?? 0) };
  const bc = { x: c.x - b.x, y: c.y - b.y, z: (c.z ?? 0) - (b.z ?? 0) };
  const dot = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
  const magnitudeA = Math.hypot(ba.x, ba.y, ba.z);
  const magnitudeC = Math.hypot(bc.x, bc.y, bc.z);
  if (!magnitudeA || !magnitudeC) return 0;
  const cosine = clamp(dot / (magnitudeA * magnitudeC), -1, 1);
  return Math.acos(cosine) * 180 / Math.PI;
}

function angleFromVertical(top, bottom) {
  const dx = top.x - bottom.x;
  const dy = top.y - bottom.y;
  return Math.abs(Math.atan2(dx, -dy) * 180 / Math.PI);
}

function midpoint(a, b) {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: ((a.z ?? 0) + (b.z ?? 0)) / 2,
  };
}

function exponentialSmooth(previous, current, alpha) {
  if (previous === null || previous === undefined || Number.isNaN(previous)) return current;
  return previous + alpha * (current - previous);
}

function average(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function formatDegrees(value) {
  return Number.isFinite(value) ? `${Math.round(value)}°` : "—";
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function resetCounter() {
  repCount = 0;
  squatState = "standing";
  previousAverageKnee = null;
  movementTrend = 0;
  ui.repCount.textContent = "0";
  ui.phase.textContent = "Waiting";
  ui.phaseDetail.textContent = "Stand tall to calibrate";
}

function setBadge(text, type) {
  systemBadge.textContent = text;
  systemBadge.className = `badge badge-${type}`;
}

function friendlyError(error) {
  const name = error?.name ?? "";
  if (name === "NotAllowedError") {
    return "Camera permission was denied. In iPhone Settings, allow Safari camera access, then reload this page.";
  }
  if (name === "NotFoundError") {
    return "No usable camera was found on this device.";
  }
  if (!window.isSecureContext) {
    return "Camera access requires HTTPS. Upload this folder to the secure host described in README.md.";
  }
  return `Could not start the assessment: ${error?.message ?? "Unknown error"}`;
}
