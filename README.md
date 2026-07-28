# Safari MediaPipe Squat Assessment Prototype — v3

This static HTTPS site runs MediaPipe Pose Landmarker locally in iPhone Safari and records a short three-repetition squat assessment.

## Version 3 changes

- Requires only the lower body: both hips, knees, ankles, heels, and feet.
- No longer requires shoulders or the head to fit in the camera view.
- Allows the user to stand substantially closer to a front-facing phone.
- Waits for a stable, tall squat-ready stance before beginning.
- Shows a short automatic countdown after the stance is accepted.
- Ignores walking into position because recording has not started yet.
- Stores the starting foot position and pauses if the user walks away during the test.
- Stops automatically and freezes results after exactly three valid repetitions.
- Replaces torso lean with stance width because shoulders are no longer required.

## Solo testing flow

1. Tap **Start assessment** while near the phone.
2. Walk into position.
3. Make sure the image shows your waist/hips, knees, ankles, and both feet.
4. Stand tall in your normal squat stance and hold still.
5. Wait for the short countdown.
6. Perform three controlled squats.
7. The test stops automatically at 3/3 and freezes the diagnostics.

## GitHub Pages update

Replace these files in the repository root:

- `index.html`
- `styles.css`
- `app.js`
- `README.md`
- `LICENSE-NOTICE.txt`

Commit the changes and reopen the GitHub Pages site. Safari may cache the older JavaScript, so close the old tab or use a private tab for the first v3 test.

## Prototype thresholds

- Standing knee angle: 155°
- Descent begins below: 145°
- Bottom recognized at: 120°
- Ready stance hold: 0.9 seconds
- Countdown: 2 seconds
- Target: 3 valid repetitions

These are engineering prototype settings and are not validated clinical standards.

## Privacy

Video is processed locally in the browser. The prototype does not upload or save camera footage. MediaPipe JavaScript/WASM and the pose model are downloaded from public hosting when the page loads.
