const video4 = document.createElement('video');
const out4 = document.getElementsByClassName('output4')[0];
const wordCounter = document.querySelector('.word-counter');
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
const threshold = 0.1;
const actions = ["hello", "thanks", "iloveyou"];

// Used to eliminate jumpy action output 
let lastAction = null;
let consistentAction = null;
let actionCounter = 0;
const actionThreshold = 5; // Change this value to the desired threshold

// word index
let index = 0;
wordCounter.textContent = '0/5';

// Load your TensorFlow.js model (assuming you have a model to load)
async function loadModel() {
  try {
    model = await tf.loadLayersModel('./tfjs_files/model.json');
    console.log("Model loaded successfully.");
    console.log("Model summary:", model.summary());
  } catch (error) {
    console.error("Error loading model:", error);
  }
}

loadModel();

function extractKeypoints(results) {
  const flatten = (arr) => arr.reduce((flat, toFlatten) => flat.concat(toFlatten), []);

  // Extract pose, face, left hand, and right hand landmarks
  const pose = results.poseLandmarks ? 
      flatten(results.poseLandmarks.map(res => [res.x, res.y, res.z, res.visibility])) : 
      Array(33 * 4).fill(0);

  const face = results.faceLandmarks ? 
      flatten(results.faceLandmarks.map(res => [res.x, res.y, res.z])) : 
      Array(468 * 3).fill(0);

  const lh = results.leftHandLandmarks ? 
      flatten(results.leftHandLandmarks.map(res => [res.x, res.y, res.z])) : 
      Array(21 * 3).fill(0);

  const rh = results.rightHandLandmarks ? 
      flatten(results.rightHandLandmarks.map(res => [res.x, res.y, res.z])) : 
      Array(21 * 3).fill(0);

  // Concatenate all keypoints into a single array
  const keypoints = pose.concat(face, lh, rh);

  // Ensure the keypoints array has a length of 1662
  if (keypoints.length !== 1662) {
    console.error(`Invalid keypoints length: ${keypoints.length}`);
    return Array(1662).fill(0);
  }

  return keypoints;
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

function highlightWord(word) {
  const wordsContainer = document.querySelector('.wordcontainer .word');
  const wordDivs = wordsContainer.querySelectorAll('div');

  const currentDiv = wordDivs[index];
  const currentWord = currentDiv.textContent;

  if (word === currentWord) {
      const wordSpans = currentDiv.querySelectorAll('span');
      wordSpans.forEach(span => span.style.color = '#21a663');
      index++; // Increase index by 1
      wordCounter.textContent = `${index}/5`;
  }

  // if (index == (wordDivs.length - 1)) {
  //   alert("Test finished");
  // }
}


async function onResultsHolistic(results) {
  document.body.classList.add('loaded');
  fpsControl.tick();

  const keypoints = extractKeypoints(results);
  sequence.push(keypoints);
  sequence = sequence.slice(-30);

  if (sequence.length === 30 && model) {
    const inputTensor = tf.tensor([sequence]);
    const res = model.predict(inputTensor).dataSync();
    const maxRes = Math.max(...res);
    const maxIndex = res.indexOf(maxRes);
    const action = actions[maxIndex];
    predictions.push(maxIndex);

    if (action === lastAction) {
      actionCounter++;
    } else {
      actionCounter = 1; // Reset counter
      lastAction = action;
    }

    if (actionCounter >= actionThreshold) {
      console.log('Consistent action:', action);
      consistentAction = action;
      actionCounter = 0; // Reset counter after output
    }

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

    // canvasCtx4.fillStyle = "rgba(245, 117, 16, 1)";
    // canvasCtx4.fillRect(0, 0, 640, 40);
    // canvasCtx4.fillStyle = "#ffffff";
    // canvasCtx4.font = "30px Arial";
    // canvasCtx4.fillText(sentence.join(' '), 3, 30);

    // Highlight the word in the word container if consistentAction is not null
    if (consistentAction !== null) {
      highlightWord(consistentAction);
    }
  }

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
holistic.onResults(onResultsHolistic);

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
  upperBodyOnly: false,
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

