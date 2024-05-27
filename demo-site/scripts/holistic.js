const video4 = document.createElement('video');
const out4 = document.getElementsByClassName('output4')[0];
const controlsElement4 = document.getElementsByClassName('control4')[0];
const canvasCtx4 = out4.getContext('2d');

const fpsControl = new FPS();
const spinner = document.querySelector('.loading');
spinner.ontransitionend = () => {
  spinner.style.display = 'none';
};

function extractKeypoints(results) {
  const flatten = (arr) => arr.reduce((flat, toFlatten) => flat.concat(toFlatten), []);

  const pose = results.poseLandmarks?.landmark ? 
      flatten(results.poseLandmarks.landmark.map(res => [res.x, res.y, res.z, res.visibility])) :
      Array(33 * 4).fill(0);

  const face = results.faceLandmarks?.landmark ? 
      flatten(results.faceLandmarks.landmark.map(res => [res.x, res.y, res.z])) :
      Array(468 * 3).fill(0);

  const lh = results.leftHandLandmarks?.landmark ? 
      flatten(results.leftHandLandmarks.landmark.map(res => [res.x, res.y, res.z])) :
      Array(21 * 3).fill(0);

  const rh = results.rightHandLandmarks?.landmark ? 
      flatten(results.rightHandLandmarks.landmark.map(res => [res.x, res.y, res.z])) :
      Array(21 * 3).fill(0);

  return pose.concat(face, lh, rh);
}


function removeElements(landmarks, elements) {
  for (const element of elements) {
    delete landmarks[element];
  }
}

function removeLandmarks(results) {
  if (results.poseLandmarks) {
    removeElements(
        results.poseLandmarks,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22]);
  }
}

function connect(ctx, connectors) {
  const canvas = ctx.canvas;
  for (const connector of connectors) {
    const from = connector[0];
    const to = connector[1];
    if (from && to) {
      if (from.visibility && to.visibility &&
          (from.visibility < 0.1 || to.visibility < 0.1)) {
        continue;
      }
      ctx.beginPath();
      ctx.moveTo(from.x * canvas.width, from.y * canvas.height);
      ctx.lineTo(to.x * canvas.width, to.y * canvas.height);
      ctx.stroke();
    }
  }
}

function onResultsHolistic(results) {
  document.body.classList.add('loaded');
  removeLandmarks(results);
  fpsControl.tick();

  // Log the entire results object to check its structure
  console.log('Holistic results:', results);

  // Extract and log keypoints
  const keypoints = extractKeypoints(results);
  console.log('Extracted keypoints:', keypoints);

  canvasCtx4.save();
  canvasCtx4.clearRect(0, 0, out4.width, out4.height);
  canvasCtx4.drawImage(results.image, 0, 0, out4.width, out4.height);
  canvasCtx4.lineWidth = 5;
  
  if (results.poseLandmarks) {
    if (results.rightHandLandmarks) {
      canvasCtx4.strokeStyle = '#00FF00';
      connect(canvasCtx4, [
        [results.poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW], results.rightHandLandmarks[0]]
      ]);
    }
    if (results.leftHandLandmarks) {
      canvasCtx4.strokeStyle = '#FF0000';
      connect(canvasCtx4, [
        [results.poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW], results.leftHandLandmarks[0]]
      ]);
    }
  }
  
  drawConnectors(canvasCtx4, results.poseLandmarks, POSE_CONNECTIONS, { color: '#00FF00' });
  drawLandmarks(canvasCtx4, results.poseLandmarks, { color: '#00FF00', fillColor: '#FF0000' });
  drawConnectors(canvasCtx4, results.rightHandLandmarks, HAND_CONNECTIONS, { color: '#00CC00' });
  drawLandmarks(canvasCtx4, results.rightHandLandmarks, {
    color: '#00FF00',
    fillColor: '#FF0000',
    lineWidth: 2,
    radius: (data) => {
      return lerp(data.from.z, -0.15, .1, 10, 1);
    }
  });
  drawConnectors(canvasCtx4, results.leftHandLandmarks, HAND_CONNECTIONS, { color: '#CC0000' });
  drawLandmarks(canvasCtx4, results.leftHandLandmarks, {
    color: '#FF0000',
    fillColor: '#00FF00',
    lineWidth: 2,
    radius: (data) => {
      return lerp(data.from.z, -0.15, .1, 10, 1);
    }
  });

  canvasCtx4.restore();
}



const holistic = new Holistic({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.1/${file}`;
}});
holistic.onResults(onResultsHolistic);

const camera = new Camera(video4, {
  onFrame: async () => {
    await holistic.send({image: video4});
  },
  width: 480,
  height: 480
});
camera.start();

new ControlPanel(controlsElement4, {
      selfieMode: true,
      upperBodyOnly: true,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    })
    .add([
      new StaticText({title: 'MediaPipe Holistic'}),
      fpsControl,
      new Toggle({title: 'Selfie Mode', field: 'selfieMode'}),
      new Toggle({title: 'Upper-body Only', field: 'upperBodyOnly'}),
      new Toggle(
          {title: 'Smooth Landmarks', field: 'smoothLandmarks'}),
      new Slider({
        title: 'Min Detection Confidence',
        field: 'minDetectionConfidence',
        range: [0, 1],
        step: 0.01
      }),
      new Slider({
        title: 'Min Tracking Confidence',
        field: 'minTrackingConfidence',
        range: [0, 1],
        step: 0.01
      }),
    ])
    .on(options => {
      video4.classList.toggle('selfie', options.selfieMode);
      holistic.setOptions(options);
    });
