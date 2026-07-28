# Safari MediaPipe Movement Assessment Prototype — v2

This static HTTPS website tests live pose tracking on an iPhone in Safari.

## Version 2 changes

- Fixes the “Camera is off” overlay remaining visible after camera permission.
- Adds a readiness gate before repetitions can be counted.
- Requires a complete, centered, properly sized full-body pose.
- Requires the user to stand tall with planted feet for about 1.2 seconds.
- Automatically pauses repetition recording when no person is detected, body coverage is lost, framing becomes invalid, or the feet move as if the user is walking.
- Cancels incomplete repetitions when recording pauses.
- Uses a four-stage repetition state machine: standing, descending, bottom, ascending.
- Requires the bottom threshold for multiple analyzed frames and the standing threshold for multiple frames before counting a repetition.
- Keeps the existing diagnostics: tracking quality, knee angles, symmetry, torso lean, phase, repetition count, FPS, latency, and coverage.

## How the readiness gate works

The camera and MediaPipe remain active so the site can tell when the user returns, but **repetition recording is off** until all conditions are met:

1. Required shoulders, hips, knees, ankles, heels, and feet are visible.
2. Landmark visibility is high enough.
3. The whole body is inside the image.
4. The person is centered and not too close or too far away.
5. Both knees indicate a tall standing position.
6. The feet remain stable for approximately 1.2 seconds.

After arming, moving the feet, leaving the frame, or losing reliable tracking pauses the counter. To resume, return to a tall, stable ready stance.

## GitHub Pages update

Upload and replace these files in the repository root:

- `index.html`
- `styles.css`
- `app.js`
- `README.md`
- `LICENSE-NOTICE.txt`

After committing the changes, refresh the GitHub Pages site. On iPhone Safari, close the old tab or use a private tab if Safari temporarily serves a cached copy.

## Prototype thresholds

- Standing angle: 155°
- Descent starts below: 145°
- Bottom reached at: 120°
- Ready hold: 1.2 seconds
- Pose-loss grace period: 450 milliseconds

These are engineering prototype values and are not validated clinical standards.

## Privacy

Video frames are processed locally in the browser. This prototype does not upload or save video. MediaPipe JavaScript/WASM and the pose model are downloaded from public hosting when the page loads.
