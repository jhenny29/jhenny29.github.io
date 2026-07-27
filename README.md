# Safari MediaPipe Movement Assessment Prototype

This is a no-build static website. It tests this pipeline on an iPhone:

1. Safari camera permission
2. MediaPipe Pose Landmarker in the browser
3. Skeleton overlay
4. Per-landmark visibility / tracking quality
5. Left and right knee angles
6. Knee symmetry
7. Torso lean
8. Squat phase and repetition count
9. Inference FPS and latency
10. Full-body coverage checks

## Important terminology

The “Tracking quality” value is the average MediaPipe landmark visibility for the shoulders, hips, knees, ankles, heels, and feet. It is a useful prototype diagnostic, but it is not a clinical confidence score and it is not a medical diagnosis.

## Fastest iPhone test: static HTTPS hosting

An iPhone browser will only provide camera access on a secure HTTPS page. Upload the entire folder—without changing the filenames—to a static hosting provider such as Netlify Drop, GitHub Pages, Cloudflare Pages, or another HTTPS host.

### Simple drag-and-drop approach

1. Unzip this project.
2. Open your chosen static-hosting dashboard on the Surface Pro.
3. Upload the entire `pose-safari-prototype` folder.
4. The host will give you an `https://...` address.
5. Open that address in Safari on your iPhone.
6. Tap **Start assessment**.
7. Choose **Allow** when Safari asks for camera permission.

Do not open `index.html` directly from the iPhone Files app. Camera APIs generally require an HTTPS site.

## Testing position

- Prop the iPhone upright.
- Begin about 7–10 feet away and adjust until your full body is visible.
- Keep the room bright.
- Wear clothing that contrasts with the background.
- Use the rear camera for better quality when somebody else can position the phone.
- Use the front camera when testing alone.

## What to expect

- Green points: required landmarks have usable visibility.
- Yellow points: the joint is detected but visibility is weak.
- Tracking quality: average visibility of required landmarks.
- Knee angles: 3D hip–knee–ankle angles derived from MediaPipe landmarks.
- Squat count: requires the average knee angle to pass below 105° and return above 155°.

Those thresholds are prototype values, not validated assessment standards.

## Files

- `index.html` — user interface
- `styles.css` — mobile styling
- `app.js` — camera, MediaPipe, diagnostics, overlay, and squat counter

## Privacy in this prototype

Camera frames are passed directly to MediaPipe in the browser. This project does not contain code that uploads or stores video. It does download the MediaPipe JavaScript/WASM package and pose model from public CDNs when the page loads.

## Troubleshooting

### Camera permission denied

Open iPhone **Settings → Apps → Safari → Camera** and allow camera access, then reload the site.

### No pose detected

Move farther from the phone, make sure ankles and feet are visible, and improve lighting.

### Low FPS or heat

The prototype intentionally analyzes about 14 frames per second. Close other Safari tabs and try the rear camera. The lite pose model is already selected.

### Blank page or model-load error

Confirm the iPhone has internet access. This prototype loads MediaPipe and the model from online sources.

## Next technical milestone

After this works consistently, the next version should record a short session summary with minimum knee angle, average symmetry, rep consistency, and percentage of frames with complete body coverage. Only after that should we move the same measurement logic into an Expo native development build.
