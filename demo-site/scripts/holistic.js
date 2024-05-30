const video4 = document.createElement('video');
const out4 = document.getElementsByClassName('output4')[0];
const controlsElement4 = document.getElementsByClassName('control4')[0];
const canvasCtx4 = out4.getContext('2d');

const fpsControl = new FPS();
const spinner = document.querySelector('.loading');
spinner.ontransitionend = () => {
  spinner.style.display = 'none';
};

let sequence = [];
let sentence = [];
let predictions = [];
const threshold = 0.5;
const actions = ["action1", "action2", "action3"]; // Example actions, replace with actual actions
let model;

// Load your TensorFlow.js model (assuming you have a model to load)
async function loadModel() {
  model = await tf.loadLayersModel('path_to_your_model/model.json');
}

loadModel();

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

function probViz(res, actions, ctx, colors) { // visualize the probabilities
  ctx.save();
  const barWidth = 30;
  const barSpacing = 10;
  const offset = 10;
  for (let i = 0; i < actions.length; i++) {
    ctx.fillStyle = colors[i];
    const barHeight = res[i] * 100;
    ctx.fillRect(offset + (barWidth + barSpacing) * i, 50 - barHeight, barWidth, barHeight);
    ctx.fillStyle = "#000000";
    ctx.fillText(actions[i], offset + (barWidth + barSpacing) * i, 50 - barHeight - 10);
  }
  ctx.restore();
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

async function onResultsHolistic(results) {
  document.body.classList.add('loaded');
  removeLandmarks(results);
  fpsControl.tick();

  // Extract keypoints
  const keypoints = extractKeypoints(results);
  sequence.push(keypoints);
  sequence = sequence.slice(-30);

  if (sequence.length === 30 && model) {
    // Prediction logic
    const inputTensor = tf.tensor([sequence]);
    const res = model.predict(inputTensor).dataSync();
    const maxRes = Math.max(...res);
    const maxIndex = res.indexOf(maxRes);
    const action = actions[maxIndex];
    predictions.push(maxIndex);

    // Visualization logic
    if (predictions.slice(-10).filter(p => p === maxIndex).length === 10) {
      if (maxRes > threshold) {
        if (sentence.length > 0) {
          if (action !== sentence[sentence.length - 1]) {
            sentence.push(action);
          }
        } else {
          sentence.push(action);
        }
      }
    }

    if (sentence.length > 5) {
      sentence = sentence.slice(-5);
    }

    // Visualization on the canvas
    const colors = ["#ff0000", "#00ff00", "#0000ff"]; // Example colors
    probViz(res, actions, canvasCtx4, colors);

    canvasCtx4.fillStyle = "rgba(245, 117, 16, 1)";
    canvasCtx4.fillRect(0, 0, 640, 40);
    canvasCtx4.fillStyle = "#ffffff";
    canvasCtx4.font = "30px Arial";
    canvasCtx4.fillText(sentence.join(' '), 3, 30);
  }

  // Draw landmarks and connectors
  canvasCtx4.save();
  canvasCtx4.clearRect(0, 0, out4.width, out4.height);
  canvasCtx4.drawImage(results.image, 0, 0, out4.width, out4.height);
  canvasCtx4.lineWidth = 5;

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
holistic.onResults(drawLandmarks);

const camera = new Camera(video4, {
  onFrame: async () => {
    if (model) {
      await holistic.send({ image: video4 });
    }
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
