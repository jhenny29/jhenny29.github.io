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
const trackingGate = document.querySelector("#trackingGate");
const gateTitle = document.querySelector("#gateTitle");
const gateDetail = document.querySelector("#gateDetail");
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
  stanceWidth: document.querySelector("#stanceWidth"),
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
  [23, 24],
  [23, 25], [25, 27], [27, 29], [29, 31], [27, 31],
  [24, 26], [26, 28], [28, 30], [30, 32], [28, 32],
];

// Only the lower body is required. The user can stand much closer to the phone.
const KEY_JOINTS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32];

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
let previousAnkleMidpoint = null;
let previousFootSpan = null;
let positionMotion = 0;
let readyAnkleMidpoint = null;
let readyFootSpan = null;

let assessmentMode = "waiting"; // waiting, calibrating, countdown, active, paused, completed
let readyStartedAt = null;
let lastTrackablePoseAt = 0;
let repCount = 0;
let squatState = "standing"; // standing, descending, bottom, ascending
let repStartedAt = null;
let descentFrames = 0;
let bottomFrames = 0;
let standingFrames = 0;

const PROCESS_INTERVAL_MS = 70;
const VISIBILITY_REQUIRED = 0.55;
const ARM_VISIBILITY_REQUIRED = 0.68;
const READY_HOLD_MS = 900;
const START_COUNTDOWN_MS = 2000;
const TARGET_REPS = 3;
const LOST_POSE_GRACE_MS = 450;
const STANDING_ANGLE = 155;
const DESCENT_START_ANGLE = 145;
const BOTTOM_ANGLE = 120;
const MIN_REP_MS = 650;
const MAX_REP_MS = 8000;
const MAX_READY_FOOT_MOTION = 0.012;
const MAX_ACTIVE_FOOT_MOTION = 0.032;
const MAX_ACTIVE_POSITION_DRIFT = 0.075;
const MAX_ACTIVE_STANCE_CHANGE = 0.22;

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

    resetTrackingState(true);
    running = true;
    startButton.disabled = false;
    startButton.textContent = "Stop assessment";
    cameraButton.disabled = false;
    emptyState.hidden = true;
    trackingGate.hidden = false;
    setGate("Show hips to feet", "Stand tall in your squat stance and hold still.", "waiting");
    setBadge("Positioning", "loading");
    statusText.textContent = "The test starts automatically when your hips, knees, ankles, and feet are visible and stable.";
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
  resetTrackingState(false);
  trackingGate.hidden = false;
  setGate("Show hips to feet", "Stand tall in your squat stance and hold still to resume.", "waiting");
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
  trackingGate.hidden = true;
  startButton.disabled = false;
  startButton.textContent = "Start assessment";
  cameraButton.disabled = true;
  resetTrackingState(false);
  setBadge("Idle", "idle");
  statusText.textContent = "Camera stopped. Video was processed locally and was not uploaded.";
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

    handleResult(result, inferenceMs, now);
  }

  animationId = requestAnimationFrame(predictLoop);
}

function handleResult(result, inferenceMs, now) {
  clearCanvas();

  const landmarks = result.landmarks?.[0];
  if (!landmarks) {
    handleUntrackablePose(now, "No person detected");
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

  const hipMidpoint = midpoint(landmarks[LANDMARK.leftHip], landmarks[LANDMARK.rightHip]);
  const ankleMidpoint = midpoint(landmarks[LANDMARK.leftAnkle], landmarks[LANDMARK.rightAnkle]);
  const hipSpan = distance2D(landmarks[LANDMARK.leftHip], landmarks[LANDMARK.rightHip]);
  const footSpan = distance2D(landmarks[LANDMARK.leftAnkle], landmarks[LANDMARK.rightAnkle]);

  const rawPositionMotion = previousAnkleMidpoint
    ? distance2D(ankleMidpoint, previousAnkleMidpoint) + Math.abs(footSpan - previousFootSpan) * 0.5
    : 0;
  positionMotion = exponentialSmooth(positionMotion, rawPositionMotion, 0.35);
  previousAnkleMidpoint = ankleMidpoint;
  previousFootSpan = footSpan;

  const stanceWidthRatio = hipSpan > 0.001 ? footSpan / hipSpan : 0;
  const symmetry = Math.abs(smoothedLeftKnee - smoothedRightKnee);
  const frame = evaluateFraming(landmarks, hipMidpoint);

  const baseTrackable = hasCoverage
    && averageVisibility >= ARM_VISIBILITY_REQUIRED
    && frame.inFrame
    && frame.centered
    && frame.bodySizeOk;

  let phaseResult;

  if (assessmentMode === "active") {
    const feetStableEnough = positionMotion <= MAX_ACTIVE_FOOT_MOTION;
    const positionDrift = readyAnkleMidpoint ? distance2D(ankleMidpoint, readyAnkleMidpoint) : 0;
    const stanceChange = readyFootSpan && readyFootSpan > 0.001
      ? Math.abs(footSpan - readyFootSpan) / readyFootSpan
      : 0;
    const stillInStartingSpot = positionDrift <= MAX_ACTIVE_POSITION_DRIFT
      && stanceChange <= MAX_ACTIVE_STANCE_CHANGE;
    const activeTrackable = baseTrackable && feetStableEnough && stillInStartingSpot;

    if (activeTrackable) {
      lastTrackablePoseAt = now;
      phaseResult = updateSquatCounter(averageKnee, movementTrend, now);
      if (assessmentMode === "active") {
        setBadge(`Recording ${repCount}/${TARGET_REPS}`, "active");
        setGate("Assessment active", `Complete ${TARGET_REPS - repCount} more repetition${TARGET_REPS - repCount === 1 ? "" : "s"}.`, "active");
      }
    } else if (now - lastTrackablePoseAt > LOST_POSE_GRACE_MS) {
      let reason = readinessReason({ hasCoverage, averageVisibility, frame, standing: true, stable: true });
      if (!feetStableEnough) reason = "Foot movement detected";
      if (!stillInStartingSpot) reason = "You moved away from the starting stance";
      pauseAssessment(reason);
      phaseResult = { phase: "Paused", detail: "Return to your squat-ready stance to resume" };
    } else {
      phaseResult = { phase: "Tracking", detail: "Brief tracking interruption" };
    }
  } else {
    phaseResult = updateReadiness({
      now,
      baseTrackable,
      hasCoverage,
      averageVisibility,
      frame,
      averageKnee,
      positionMotion,
      ankleMidpoint,
      footSpan,
    });
  }

  const qualityPercent = Math.round(averageVisibility * 100);
  ui.trackingQuality.textContent = `${qualityPercent}%`;
  ui.trackingDetail.textContent = qualityPercent >= 80
    ? "Strong landmark visibility"
    : qualityPercent >= 68
      ? "Usable for arming; improve lighting if unstable"
      : "Low confidence—reposition before trusting angles";

  ui.leftKnee.textContent = formatDegrees(smoothedLeftKnee);
  ui.rightKnee.textContent = formatDegrees(smoothedRightKnee);
  ui.symmetry.textContent = formatDegrees(symmetry);
  ui.stanceWidth.textContent = stanceWidthRatio > 0 ? `${stanceWidthRatio.toFixed(2)}×` : "—";
  ui.phase.textContent = phaseResult.phase;
  ui.phaseDetail.textContent = phaseResult.detail;
  ui.repCount.textContent = String(repCount);
  ui.fps.textContent = `${displayedFps.toFixed(1)} FPS`;
  ui.latency.textContent = `${inferenceMs.toFixed(0)} ms per analyzed frame`;
  ui.coverage.textContent = baseTrackable ? "Ready" : hasCoverage ? "Adjust" : "Partial";
  ui.coverageDetail.textContent = baseTrackable
    ? "Hips through feet are centered and measurable"
    : coverageMessage({ missing, frame, averageVisibility });
}

function updateReadiness({
  now,
  baseTrackable,
  hasCoverage,
  averageVisibility,
  frame,
  averageKnee,
  positionMotion,
  ankleMidpoint,
  footSpan,
}) {
  const standing = averageKnee >= STANDING_ANGLE;
  const stable = positionMotion <= MAX_READY_FOOT_MOTION && Math.abs(movementTrend) < 0.9;

  if (baseTrackable && standing && stable) {
    if (readyStartedAt === null) readyStartedAt = now;
    const elapsed = now - readyStartedAt;

    if (elapsed < READY_HOLD_MS) {
      assessmentMode = "calibrating";
      const remaining = Math.max(0, READY_HOLD_MS - elapsed);
      const seconds = Math.max(1, Math.ceil(remaining / 1000));
      setBadge("Ready stance", "loading");
      setGate("Hold squat-ready position", `Checking stability… ${seconds}`, "calibrating");
      return { phase: "Getting ready", detail: "Keep hips, knees, ankles, and feet visible" };
    }

    if (elapsed < READY_HOLD_MS + START_COUNTDOWN_MS) {
      assessmentMode = "countdown";
      const countdownRemaining = READY_HOLD_MS + START_COUNTDOWN_MS - elapsed;
      const count = Math.max(1, Math.ceil(countdownRemaining / 1000));
      setBadge(`Starting in ${count}`, "loading");
      setGate(`Starting in ${count}`, "Stay tall with your feet planted.", "calibrating");
      return { phase: "Ready", detail: `Assessment begins in ${count}` };
    }

    assessmentMode = "active";
    lastTrackablePoseAt = now;
    readyAnkleMidpoint = { ...ankleMidpoint };
    readyFootSpan = footSpan;
    resetPartialRep();
    setBadge(`Recording 0/${TARGET_REPS}`, "active");
    setGate("Assessment active", `Complete ${TARGET_REPS} controlled squats.`, "active");
    statusText.textContent = `Recording started automatically. The test stops after ${TARGET_REPS} valid repetitions.`;
    return { phase: "Standing", detail: "Begin your first repetition" };
  }

  readyStartedAt = null;
  assessmentMode = assessmentMode === "paused" ? "paused" : "waiting";
  const reason = readinessReason({ hasCoverage, averageVisibility, frame, standing, stable });
  setBadge(assessmentMode === "paused" ? "Paused" : "Positioning", "loading");
  setGate(assessmentMode === "paused" ? "Recording paused" : "Not ready yet", reason, "waiting");
  statusText.textContent = "Recording stays off until a stable squat-ready stance is detected.";
  return { phase: assessmentMode === "paused" ? "Paused" : "Get ready", detail: reason };
}

function updateSquatCounter(kneeAngle, trend, now) {
  if (repStartedAt && now - repStartedAt > MAX_REP_MS) {
    resetPartialRep();
    return { phase: "Standing", detail: "Movement timed out—start again from standing" };
  }

  if (squatState === "standing") {
    if (kneeAngle < DESCENT_START_ANGLE && trend < -0.35) {
      descentFrames += 1;
      if (descentFrames >= 2) {
        squatState = "descending";
        repStartedAt = now;
        descentFrames = 0;
        return { phase: "Descending", detail: "Continue downward under control" };
      }
    } else {
      descentFrames = 0;
    }
    return { phase: "Standing", detail: "Counter armed—bend knees and hips" };
  }

  if (squatState === "descending") {
    if (kneeAngle <= BOTTOM_ANGLE) {
      bottomFrames += 1;
      if (bottomFrames >= 2) {
        squatState = "bottom";
        bottomFrames = 0;
        return { phase: "Bottom reached", detail: "Stand back up to complete the rep" };
      }
    } else {
      bottomFrames = 0;
    }

    if (kneeAngle >= STANDING_ANGLE && trend >= 0) {
      resetPartialRep();
      return { phase: "Standing", detail: "Rep cancelled because minimum depth was not reached" };
    }

    return { phase: "Descending", detail: "Reach the target depth before standing" };
  }

  if (squatState === "bottom") {
    if (kneeAngle > BOTTOM_ANGLE + 7 && trend > 0.25) {
      squatState = "ascending";
      return { phase: "Ascending", detail: "Return to a tall standing position" };
    }
    return { phase: "Bottom reached", detail: "Stand back up to complete the rep" };
  }

  if (squatState === "ascending") {
    if (kneeAngle >= STANDING_ANGLE) {
      standingFrames += 1;
      const duration = repStartedAt ? now - repStartedAt : 0;
      if (standingFrames >= 2 && duration >= MIN_REP_MS) {
        repCount += 1;
        resetPartialRep();
        if (repCount >= TARGET_REPS) {
          completeAssessment();
          return { phase: "Complete", detail: `${TARGET_REPS} of ${TARGET_REPS} repetitions recorded` };
        }
        return { phase: "Rep complete", detail: `Completed repetition ${repCount} of ${TARGET_REPS}` };
      }
    } else {
      standingFrames = 0;
    }

    if (kneeAngle <= BOTTOM_ANGLE + 3 && trend < 0) {
      squatState = "bottom";
      standingFrames = 0;
      return { phase: "Bottom reached", detail: "Stand back up to complete the rep" };
    }

    return { phase: "Ascending", detail: "Finish in a tall standing position" };
  }

  resetPartialRep();
  return { phase: "Standing", detail: "Counter armed" };
}

function completeAssessment() {
  assessmentMode = "completed";
  running = false;
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;
  stopTracksOnly();
  video.pause();
  startButton.disabled = false;
  startButton.textContent = "Start new assessment";
  cameraButton.disabled = true;
  setBadge("Complete", "active");
  setGate("Assessment complete", `${TARGET_REPS} valid repetitions recorded. Results are now frozen.`, "active");
  statusText.textContent = "Assessment stopped automatically after three repetitions. Tap Start new assessment to run it again.";
}

function handleUntrackablePose(now, reason) {
  previousAnkleMidpoint = null;
  previousFootSpan = null;
  positionMotion = 0;
  readyStartedAt = null;

  if (assessmentMode === "active" && now - lastTrackablePoseAt > LOST_POSE_GRACE_MS) {
    pauseAssessment(reason);
  }
}

function pauseAssessment(reason) {
  assessmentMode = "paused";
  readyStartedAt = null;
  resetPartialRep();
  setBadge("Paused", "loading");
  setGate("Recording paused", `${reason}. Stand tall and hold still to resume.`, "waiting");
  statusText.textContent = "No measurements are being added while recording is paused.";
}

function evaluateFraming(landmarks, hipMidpoint) {
  const points = KEY_JOINTS.map((index) => landmarks[index]);
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const lowerBodyHeight = maxY - minY;

  return {
    inFrame: minX >= 0.01 && maxX <= 0.99 && minY >= 0.01 && maxY <= 0.99,
    centered: hipMidpoint.x >= 0.16 && hipMidpoint.x <= 0.84,
    bodySizeOk: lowerBodyHeight >= 0.30 && lowerBodyHeight <= 0.96,
    bodyHeight: lowerBodyHeight,
  };
}

function readinessReason({ hasCoverage, averageVisibility, frame, standing, stable }) {
  if (!hasCoverage) return "Show both hips, knees, ankles, heels, and feet";
  if (averageVisibility < ARM_VISIBILITY_REQUIRED) return "Improve lighting or reduce joint obstruction";
  if (!frame.inFrame) return "Adjust the phone or your position until hips through feet fit";
  if (!frame.bodySizeOk) return frame.bodyHeight < 0.30
    ? "Move slightly closer so the lower body is large enough to measure"
    : "Move slightly farther away so hips through feet fit";
  if (!frame.centered) return "Move toward the center of the camera view";
  if (!standing) return "Stand tall in your squat-ready stance";
  if (!stable) return "Plant your feet and hold still";
  return "Hold your squat-ready stance";
}

function coverageMessage({ missing, frame, averageVisibility }) {
  if (missing.length) return `${missing.length} required landmark${missing.length === 1 ? " is" : "s are"} low visibility`;
  if (averageVisibility < ARM_VISIBILITY_REQUIRED) return "Landmark confidence is too low to arm";
  if (!frame.inFrame) return "Part of the body is outside the image";
  if (!frame.bodySizeOk) return "Adjust distance from the camera";
  if (!frame.centered) return "Move toward the center of the frame";
  return "Hold a stable standing stance";
}

function showNoPose(inferenceMs) {
  ui.trackingQuality.textContent = "0%";
  ui.trackingDetail.textContent = "No lower-body pose detected";
  ui.leftKnee.textContent = "—";
  ui.rightKnee.textContent = "—";
  ui.symmetry.textContent = "—";
  ui.stanceWidth.textContent = "—";
  ui.phase.textContent = assessmentMode === "paused" ? "Paused" : "Reposition";
  ui.phaseDetail.textContent = assessmentMode === "paused"
    ? "Stand tall and hold still to resume"
    : "Move into view and show your hips through both feet";
  ui.fps.textContent = `${displayedFps.toFixed(1)} FPS`;
  ui.latency.textContent = `${inferenceMs.toFixed(0)} ms per analyzed frame`;
  ui.coverage.textContent = "None";
  ui.coverageDetail.textContent = "No usable lower-body landmarks available";
  setGate(
    assessmentMode === "paused" ? "Recording paused" : "Step into frame",
    "Stand tall with hips, knees, ankles, and feet visible.",
    "waiting",
  );
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


function midpoint(a, b) {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: ((a.z ?? 0) + (b.z ?? 0)) / 2,
  };
}

function distance2D(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
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

function resetPartialRep() {
  squatState = "standing";
  repStartedAt = null;
  descentFrames = 0;
  bottomFrames = 0;
  standingFrames = 0;
}

function resetTrackingState(resetReps) {
  if (resetReps) repCount = 0;
  assessmentMode = "waiting";
  readyStartedAt = null;
  lastTrackablePoseAt = 0;
  readyAnkleMidpoint = null;
  readyFootSpan = null;
  smoothedLeftKnee = null;
  smoothedRightKnee = null;
  previousAverageKnee = null;
  movementTrend = 0;
  previousAnkleMidpoint = null;
  previousFootSpan = null;
  positionMotion = 0;
  resetPartialRep();
  ui.repCount.textContent = String(repCount);
}

function resetCounter() {
  repCount = 0;
  resetTrackingState(false);
  ui.repCount.textContent = "0";
  ui.phase.textContent = "Waiting";
  ui.phaseDetail.textContent = "Show hips through feet and hold a squat-ready stance";
  if (running) {
    trackingGate.hidden = false;
    setBadge("Positioning", "loading");
    setGate("Counter reset", "Show hips through feet and hold your squat-ready stance.", "waiting");
    statusText.textContent = "Recording is off until the ready stance is stable.";
  }
}

function setGate(title, detail, type) {
  gateTitle.textContent = title;
  gateDetail.textContent = detail;
  trackingGate.dataset.type = type;
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
