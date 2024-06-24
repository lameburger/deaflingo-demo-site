const video4 = document.createElement('video');
const out4 = document.getElementsByClassName('output4')[0];
const wordCounter = document.querySelector('.word-counter');
const actionDisplay = document.querySelector('.waiting-text');
const controlsElement4 = document.getElementsByClassName('control4')[0];
const canvasCtx4 = out4.getContext('2d');
const loadingScreen = document.querySelector('.loading-screen');
const mainContent = document.querySelector('.main-content');

let sequence = []; 
let sentence = [];
let predictions = [];
const threshold = 0.2;
const actions = ['burrito', 'hello', 'i love you', 'meet you', 'my', 'name', 'nice', 'thanks', 'time', 'what', 'where'];

// Used to eliminate jumpy action output 
let lastAction = null;
let consistentAction = null;
let actionCounter = 0;
const actionThreshold = 0; // Change this value to the desired threshold

// Load time
let lastFrameTime = 0;
let isModelLoaded = false;

// word index
let index = 0;
wordCounter.textContent = '0/5';

// Load your TensorFlow.js model (assuming you have a model to load)
async function loadModel() {
  try {
    model = await tf.loadLayersModel('https://firebasestorage.googleapis.com/v0/b/deaflingo-7190a.appspot.com/o/model.json?alt=media&token=b515d3c5-b8de-4a56-8fe3-b79415275a4d');
    console.log("Model loaded successfully.");
    console.log("Model summary:", model.summary());
    console.log(tf.getBackend());

    // Hide loading screen
    loadingScreen.style.display = 'none';
    // Show main content
    mainContent.style.display = 'block';
    // Start the camera once the model is loaded
    startCamera();
    tf.setBackend('wasm').then(() => main());
  } catch (error) {
    console.error("Error loading model:", error);
  }
}

loadModel();

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video4.playsInline = true;
    video4.srcObject = stream;
    video4.play();
    camera.start();
  } catch (error) {
    console.error("Error accessing the camera:", error);
    // Check if the error is due to camera access denial
    if (error.name === 'NotAllowedError' || error.name === 'SecurityError') {
      // Display a user-friendly message informing about camera access denial
      alert("Camera access is required for this application. Please grant camera access in your browser settings.");
    } else {
      // For other errors, display a generic error message
      alert("An error occurred while accessing the camera. Please try again later.");
    }
  }
}


function main() {
  console.log("Main function is running.");
  // Add any additional initialization code here
}

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

function highlightWord(word) {
  const wordsContainer = document.querySelector('.wordcontainer .word');
  const wordDivs = wordsContainer.querySelectorAll('div');

  const currentDiv = wordDivs[index];
  const currentWord = currentDiv.textContent;

  if (word === currentWord) {
      const wordSpans = currentDiv.querySelectorAll('span');
      wordSpans.forEach(span => span.style.color = '#21a663');
      index++; // Increase index by 1
      if (index == 5) {
        location.reload();
      } else {
        wordCounter.textContent = `${index}/5`;
      }
  }

}

function skipWord() {
  const wordsContainer = document.querySelector('.wordcontainer .word');
  const wordDivs = wordsContainer.querySelectorAll('div');

  const currentDiv = wordDivs[index];
  const currentWord = currentDiv.textContent;

  console.log("WORD SKIPPED")
  highlightWord(currentWord);
}

async function onResultsHolistic(results) {
  document.body.classList.add('loaded');

  const keypoints = extractKeypoints(results);
  sequence.push(keypoints);
  sequence = sequence.slice(-30);

  if (sequence.length === 30 && model) {
    const inputTensor = tf.tensor([sequence]);
    const res = model.predict(inputTensor).dataSync();
    inputTensor.dispose();  // Dispose of tensor here
    const maxRes = Math.max(...res);
    const maxIndex = res.indexOf(maxRes);
    const action = actions[maxIndex];
    predictions.push(maxIndex);
    predictions = predictions.slice(-30); // Keep only the last 30 predictions

    if (action === lastAction) {
      actionCounter++;
    } else {
      actionCounter = 1; // Reset counter
      lastAction = action;
    }

    if (actionCounter >= actionThreshold) {
      console.log('Consistent action:', action);
      consistentAction = action;
      actionDisplay.textContent = `${consistentAction}`;
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

    if (consistentAction !== null) {
      highlightWord(consistentAction);
    }
  }

  canvasCtx4.save();
  canvasCtx4.clearRect(0, 0, out4.width, out4.height);
  canvasCtx4.drawImage(results.image, 0, 0, out4.width, out4.height);
  canvasCtx4.lineWidth = 5;

  // drawConnectors(canvasCtx4, results.poseLandmarks, POSE_CONNECTIONS, { color: '#00FF00' });
  // drawLandmarks(canvasCtx4, results.poseLandmarks, { color: '#00FF00', fillColor: '#FF0000' });
  drawConnectors(canvasCtx4, results.rightHandLandmarks, HAND_CONNECTIONS, { color: '#FFFFFF' });
  drawLandmarks(canvasCtx4, results.rightHandLandmarks, {
    color: '#00013',
    fillColor: '#00013',
    lineWidth: 1,
    radius: (data) => {
      return lerp(data.from.z, -0.15, .1, 5, 1);
    }
  });
  drawConnectors(canvasCtx4, results.leftHandLandmarks, HAND_CONNECTIONS, { color: '#FFFFFF' });
  drawLandmarks(canvasCtx4, results.leftHandLandmarks, {
    color: '#00013',
    fillColor: '#00013',
    lineWidth: 1,
    radius: (data) => {
      return lerp(data.from.z, -0.15, .1, 5, 1);
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

document.getElementById('monke').addEventListener('click', () => {
  const wordsContainer = document.querySelector('.wordcontainer .word');
  const wordDivs = wordsContainer.querySelectorAll('div');

  const currentDiv = wordDivs[index];
  const currentWord = currentDiv.textContent;
  const flipContainer = document.querySelector('.flip-container');
  flipContainer.classList.add('flipping');
  console.log(index);
  console.log('CURRENT: WORD: ', currentWord);
  actionDisplay.textContent = `${currentWord}`;
  
  // Populate the div with an image
  const divToPopulate = document.querySelector('.back'); // Change this selector to match the div you want to populate
  if (currentWord == 'burrito') {
    divToPopulate.innerHTML = '<img src="./images/burrito.gif" style="margin-top: 25%;">';
  } else if (currentWord == 'hello') {
      divToPopulate.innerHTML = '<img src="./images/hello.gif" style="margin-top: 25%;">';
  } else if (currentWord == 'i love you') {
      divToPopulate.innerHTML = '<img src="./images/iloveyou.gif" style="margin-top: 25%;">';
  } else if (currentWord == 'meet you') {
      divToPopulate.innerHTML = '<img src="./images/meetyou.gif" style="margin-top: 25%;">';
  } else if (currentWord == 'my') {
      divToPopulate.innerHTML = '<img src="./images/my.gif" style="margin-top: 25%;">';
  } else if (currentWord == 'name') {
      divToPopulate.innerHTML = '<img src="./images/name.gif" style="margin-top: 25%;">';
  } else if (currentWord == 'nice') {
      divToPopulate.innerHTML = '<img src="./images/nice.gif" style="margin-top: 25%;">';
  } else if (currentWord == 'thanks') {
      divToPopulate.innerHTML = '<img src="./images/thanks.gif" style="margin-top: 25%;">';
  } else if (currentWord == 'time') {
      divToPopulate.innerHTML = '<img src="./images/time.gif" style="margin-top: 25%;">';
  } else if (currentWord == 'what') {
      divToPopulate.innerHTML = '<img src="./images/what.gif" style="margin-top: 25%;">';
  } else if (currentWord == 'where') {
      divToPopulate.innerHTML = '<img src="./images/where.gif" style="margin-top: 25%;">';
  }
  
  setTimeout(() => {
    flipContainer.classList.remove('flipping');
    // document.querySelector('.back').textContent = `${currentWord}`;
  }, 5000);
});

new ControlPanel(controlsElement4, {
  selfieMode: true,
  upperBodyOnly: false,
  smoothLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
})
.add([])
.on(options => {
  video4.classList.toggle('selfie', options.selfieMode);
  holistic.setOptions(options);
});