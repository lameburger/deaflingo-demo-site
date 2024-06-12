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
const threshold = 0.5;
const actions = ["hello", "thanks", "iloveyou"];

// Used to eliminate jumpy action output 
let lastAction = null;
let consistentAction = null;
let actionCounter = 0;
const actionThreshold = 10; // Change this value to the desired threshold

// Load time
let lastFrameTime = 0;
let isModelLoaded = false;

// word index
let index = 0;
wordCounter.textContent = '0/5';

// Save and load model from IndexedDB
const modelUrl = './tfjs_files/model.json';
const modelKey = 'tfjs-model';

async function saveModelToIndexedDB(model) {
    await model.save(`indexeddb://${modelKey}`);
    console.log("Model saved to IndexedDB.");
}

async function loadModelFromIndexedDB() {
    try {
        const model = await tf.loadLayersModel(`indexeddb://${modelKey}`);
        console.log("Model loaded from IndexedDB.");
        return model;
    } catch (error) {
        console.log("No model found in IndexedDB, loading from URL.");
        return null;
    }
}

async function loadModel() {
    try {
        // Check if model exists in IndexedDB
        let model = await loadModelFromIndexedDB();
        if (!model) {
            // If not, load from URL and save it
            model = await tf.loadLayersModel(modelUrl);
            await saveModelToIndexedDB(model);
        }
        console.log("Model loaded successfully.");
        console.log("Model summary:", model.summary());
        // Hide loading screen
        loadingScreen.style.display = 'none';
        // Show main content
        mainContent.style.display = 'block';
        // Start the camera once the model is loaded
        startCamera();
        tf.setBackend('wasm').then(() => main());
        return model;
    } catch (error) {
        console.error("Error loading model:", error);
        return null;
    }
}

let model = await loadModel();

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
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22]
        );
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
    drawConnectors(canvasCtx4, results.rightHandLandmarks, HAND_CONNECTIONS, { color: '#00CC00' });
    drawLandmarks(canvasCtx4, results.rightHandLandmarks, {
        color: '#00FF00',
        fillColor: '#FF0000',
        lineWidth: 1,
        radius: (data) => {
            return lerp(data.from.z, -0.15, .1, 5, 1);
        }
    });
    drawConnectors(canvasCtx4, results.leftHandLandmarks, HAND_CONNECTIONS, { color: '#CC0000' });
    drawLandmarks(canvasCtx4, results.leftHandLandmarks, {
        color: '#FF0000',
        fillColor: '#00FF00',
        lineWidth: 1,
        radius: (data) => {
            return lerp(data.from.z, -0.15, .1, 5, 1);
        }
    });

    canvasCtx4.restore();

}

const holistic = new Holistic({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.1/${file}`;
    }
});
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
    if (currentWord == 'iloveyou') {
        divToPopulate.innerHTML = '<img src="./images/iloveyou.gif" style="margin-top: 25%;">';
    } else if (currentWord == 'hello') {
        divToPopulate.innerHTML = '<img src="./images/hello.gif" style="margin-top: 25%;">';
    } else {
        divToPopulate.innerHTML = '<img src="./images/thanks.gif" style="margin-top: 25%;">';
    }

    setTimeout(() => {
        flipContainer.classList.remove('flipping');
        // document.querySelector('.back').textContent = `${currentWord}`;
    }, 5000);
});

// Adjust camera settings for mobile devices
if (/Mobi|Android/i.test(navigator.userAgent)) {
    camera.width = 320;
    camera.height = 320;
}

camera.start();

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
