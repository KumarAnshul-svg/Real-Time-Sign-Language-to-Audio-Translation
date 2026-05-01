/**
 * ASL Chat | Intelligent Interpreter
 * script.js (Optimized & Visual Version)
 */

// UI Elements
const videoElement = document.getElementById('input-video');
const canvasElement = document.getElementById('output-canvas');
const canvasCtx = canvasElement.getContext('2d');
const sentenceOutput = document.getElementById('sentence-output');
const gestureHint = document.getElementById('gesture-hint');
const modelStatus = document.getElementById('model-status');
const statusText = document.getElementById('status-text');
const fpsValue = document.getElementById('fps-value');

// State
let labels = [];
let ortSession = null;
let currentSentence = "";
let predictionBuffer = [];
let lastAddedLetter = "";
let lastAddTime = 0;
let isCalibrating = false;
let calibrationData = { x: 0, y: 0 };

// Configuration - OPTIMIZED FOR RESPONSIVENESS
let config = {
    threshold: 0.70,      // Lowered from 0.85 for easier detection
    inferenceDelay: 100,  // Faster inference (10 fps)
    smoothingFrames: 6,   // Fewer frames for faster response
    showLandmarks: true,
    addDelay: 1200
};

/**
 * INITIALIZATION
 */
async function init() {
    try {
        const labelResp = await fetch('/labels');
        labels = await labelResp.json();

        statusText.textContent = "Connecting...";
        ortSession = await ort.InferenceSession.create('models/asl_model.onnx', {
            executionProviders: ['wasm']
        });
        
        modelStatus.classList.add('active');
        statusText.textContent = "Online";

        setupMediaPipe();
        setupUIListeners();
    } catch (e) {
        console.error("Init Error:", e);
        statusText.textContent = "Offline";
    }
}

/**
 * PIPELINE
 */
function setupMediaPipe() {
    const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.6, // Lowered for better tracking
        minTrackingConfidence: 0.6
    });

    hands.onResults(onResults);

    const camera = new Camera(videoElement, {
        onFrame: async () => {
            await hands.send({ image: videoElement });
        },
        width: 1280,
        height: 720
    });
    camera.start();
}

/**
 * INFERENCE
 */
let lastInferenceTime = 0;
let frameCount = 0;
let lastFpsTime = Date.now();

async function onResults(results) {
    const now = Date.now();
    
    // FPS
    frameCount++;
    if (now - lastFpsTime >= 1000) {
        fpsValue.textContent = frameCount;
        frameCount = 0;
        lastFpsTime = now;
    }

    // Canvas
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0];

        if (config.showLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#10a37f', lineWidth: 4 });
            drawLandmarks(canvasCtx, landmarks, { color: '#ffffff', lineWidth: 1, radius: 2 });
        }

        if (now - lastInferenceTime > config.inferenceDelay) {
            const pred = await runInference(landmarks);
            smoothAndProcess(pred);
            lastInferenceTime = now;
            
            // Debug: show raw prediction in the gesture pill if threshold is met
            if (pred.confidence > 0.5) {
                gestureHint.textContent = `Raw: ${pred.label} (${Math.round(pred.confidence*100)}%)`;
            }
        }
    } else {
        predictionBuffer = [];
        gestureHint.textContent = "Position your hand...";
    }

    canvasCtx.restore();
}

async function runInference(landmarks) {
    if (!ortSession) return { label: 'nothing', confidence: 0 };

    // Standard Normalization
    const wrist = landmarks[0];
    const inputData = new Float32Array(42);
    
    for (let i = 0; i < 21; i++) {
        inputData[i * 2] = landmarks[i].x - wrist.x;
        inputData[i * 2 + 1] = landmarks[i].y - wrist.y;
    }

    const inputTensor = new ort.Tensor('float32', inputData, [1, 42]);
    const outputMap = await ortSession.run({ input: inputTensor });
    const output = outputMap.output.data;

    // Softmax
    const maxVal = Math.max(...output);
    const expValues = output.map(v => Math.exp(v - maxVal));
    const sumExp = expValues.reduce((a, b) => a + b);
    const probs = expValues.map(v => v / sumExp);
    const maxProb = Math.max(...probs);
    const predictedIdx = probs.indexOf(maxProb);

    return { label: labels[predictedIdx], confidence: maxProb };
}

function smoothAndProcess(prediction) {
    // If confidence is low, treat as nothing
    const label = prediction.confidence > config.threshold ? prediction.label : 'nothing';
    
    predictionBuffer.push(label);
    if (predictionBuffer.length > config.smoothingFrames) {
        predictionBuffer.shift();
    }

    const counts = {};
    predictionBuffer.forEach(l => counts[l] = (counts[l] || 0) + 1);
    const mostFreq = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
    const count = counts[mostFreq];

    // More lenient stability (60% match)
    if (count >= config.smoothingFrames * 0.6 && mostFreq !== 'nothing') {
        commitLetter(mostFreq);
    }
}

function commitLetter(letter) {
    const now = Date.now();
    
    if (letter === 'space') {
        if (lastAddedLetter !== 'space' || now - lastAddTime > config.addDelay) {
            currentSentence += " ";
            lastAddedLetter = 'space';
            lastAddTime = now;
        }
    } else if (letter === 'del') {
        if (lastAddedLetter !== 'del' || now - lastAddTime > 600) {
            currentSentence = currentSentence.slice(0, -1);
            lastAddedLetter = 'del';
            lastAddTime = now;
        }
    } else if (letter !== 'nothing') {
        if (letter !== lastAddedLetter || now - lastAddTime > config.addDelay) {
            currentSentence += letter;
            lastAddedLetter = letter;
            lastAddTime = now;
        }
    }
    
    sentenceOutput.textContent = currentSentence;
}

/**
 * UI LISTENERS
 */
function setupUIListeners() {
    const helpModal = document.getElementById('help-modal');
    const settingsModal = document.getElementById('settings-modal');

    document.getElementById('help-trigger').onclick = () => helpModal.style.display = 'flex';
    document.getElementById('settings-trigger').onclick = () => settingsModal.style.display = 'flex';
    
    document.querySelectorAll('.close-modal').forEach(btn => {
        btn.onclick = () => {
            helpModal.style.display = 'none';
            settingsModal.style.display = 'none';
        };
    });

    document.getElementById('speak-btn').onclick = () => {
        const utterance = new SpeechSynthesisUtterance(currentSentence);
        window.speechSynthesis.speak(utterance);
    };

    document.getElementById('copy-btn').onclick = () => {
        navigator.clipboard.writeText(currentSentence);
        alert("Copied!");
    };

    document.getElementById('clear-btn').onclick = () => {
        currentSentence = "";
        sentenceOutput.textContent = "";
    };

    document.getElementById('correct-btn').onclick = async () => {
        if (!currentSentence) return;
        try {
            const resp = await fetch('/correct', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: currentSentence })
            });
            const data = await resp.json();
            currentSentence = data.corrected;
            sentenceOutput.textContent = currentSentence;
        } catch (e) {}
    };

    // Setting Adjustments
    document.getElementById('conf-threshold').oninput = (e) => {
        config.threshold = parseFloat(e.target.value);
        document.getElementById('conf-val').textContent = e.target.value;
    };
    
    document.getElementById('smoothing-frames').onchange = (e) => {
        config.smoothingFrames = parseInt(e.target.value);
    };
}

// Start
init();
