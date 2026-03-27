const DROP_AREA = document.getElementById('dropArea');
const FILE_INPUT = document.getElementById('fileInput');
const UPLOAD_BTN = document.getElementById('uploadBtn');
const INFERENCE_BOX = document.getElementById('inferenceBox');
const CANVAS = document.getElementById('outputCanvas');
const CTX = CANVAS.getContext('2d');
const STATUS_TEXT = document.getElementById('statusText');
const PROGRESS_FILL = document.getElementById('progressFill');

const MODEL_PATH = './best.onnx';
let session = null;
const INPUT_DIM = 800; // Expected by YOLOv8 model we exported

// Init Model
async function loadModel() {
    try {
        updateStatus('Allocating WebAssembly Node...', 25);
        // Specify execution provider
        session = await ort.InferenceSession.create(MODEL_PATH, { executionProviders: ['wasm'] });
        updateStatus('ONNX Web Node Online. Waiting for SAR target...', 100);
        setTimeout(() => { PROGRESS_FILL.style.width = '0%'; }, 1000);
        STATUS_TEXT.classList.add('success');
    } catch (e) {
        console.error(e);
        updateStatus('Failed to allocate remote edge model.', 0);
        STATUS_TEXT.classList.add('error');
    }
}

window.onload = loadModel;

function updateStatus(msg, progress) {
    STATUS_TEXT.innerText = msg;
    if(progress !== null) PROGRESS_FILL.style.width = progress + '%';
}

// Event Listeners for Upload
UPLOAD_BTN.onclick = () => FILE_INPUT.click();
DROP_AREA.onclick = () => FILE_INPUT.click();
DROP_AREA.ondragover = (e) => { e.preventDefault(); DROP_AREA.classList.add('dragover'); };
DROP_AREA.ondragleave = () => DROP_AREA.classList.remove('dragover');
DROP_AREA.ondrop = (e) => {
    e.preventDefault();
    DROP_AREA.classList.remove('dragover');
    if(e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
};
FILE_INPUT.onchange = (e) => {
    if(e.target.files.length) handleFile(e.target.files[0]);
};

function handleFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => runInference(img);
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function preprocess(imgData, width, height) {
    const float32Data = new Float32Array(3 * INPUT_DIM * INPUT_DIM);
    // Letterbox padding vs pure stretching. For strict MVP, we'll map correctly into 800x800 tensor.
    // CHW format normalization
    for (let c = 0; c < 3; ++c) {
        for (let i = 0; i < INPUT_DIM * INPUT_DIM; ++i) {
            float32Data[c * INPUT_DIM * INPUT_DIM + i] = imgData.data[i * 4 + c] / 255.0; // RGB mapped to 0...1
        }
    }
    return new ort.Tensor('float32', float32Data, [1, 3, INPUT_DIM, INPUT_DIM]);
}

async function runInference(img) {
    INFERENCE_BOX.style.display = 'flex';
    updateStatus('Target acquired. Running ONNX topology...', 30);
    STATUS_TEXT.classList.remove('success', 'error');
    
    // Draw directly to hidden off-screen logic, then onto visible canvas
    CANVAS.width = img.width;
    CANVAS.height = img.height;
    CTX.drawImage(img, 0, 0, img.width, img.height);
    
    // Create an offscreen canvas specifically for the 800x800 input tensor
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = INPUT_DIM;
    tempCanvas.height = INPUT_DIM;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(img, 0, 0, INPUT_DIM, INPUT_DIM);
    const imgData = tempCtx.getImageData(0, 0, INPUT_DIM, INPUT_DIM);
    
    const tensor = preprocess(imgData, INPUT_DIM, INPUT_DIM);
    
    updateStatus('Inferencing...', 60);
    try {
        const t0 = performance.now();
        const results = await session.run({ images: tensor });
        const t1 = performance.now();
        
        updateStatus(`Decoupling predictions. Latency: ${(t1-t0).toFixed(1)}ms...`, 80);
        
        // Output geometry mapping
        // YOLOv8 exports outputs [1, 5, 13125] meaning [x_center, y_center, w, h, conf]
        const output = results.output0.data;
        const numClasses = 1;
        const numDetections = 13125;
        let boxes = [];
        
        for (let i = 0; i < numDetections; i++) {
            const conf = output[4 * numDetections + i];
            if (conf > 0.1) {
                const xc = output[0 * numDetections + i];
                const yc = output[1 * numDetections + i];
                const w = output[2 * numDetections + i];
                const h = output[3 * numDetections + i];
                
                // Scale back explicitly to original image space
                const x1 = ((xc - w / 2) / INPUT_DIM) * img.width;
                const y1 = ((yc - h / 2) / INPUT_DIM) * img.height;
                const box_w = (w / INPUT_DIM) * img.width;
                const box_h = (h / INPUT_DIM) * img.height;
                
                boxes.push({x1, y1, w: box_w, h: box_h, conf});
            }
        }
        
        // Simple NMS mock for MVP visual overlap clearance
        boxes.sort((a,b) => b.conf - a.conf);
        let finalBoxes = [];
        let marked = new Array(boxes.length).fill(false);
        for(let i=0; i<boxes.length; i++) {
            if(marked[i]) continue;
            finalBoxes.push(boxes[i]);
            for(let j=i+1; j<boxes.length; j++) {
                 // lazy spatial pruning
                 if(Math.abs(boxes[i].x1 - boxes[j].x1) < 50 && Math.abs(boxes[i].y1 - boxes[j].y1) < 50) marked[j] = true;
            }
        }
        
        // RENDER 
        CTX.lineWidth = 3;
        CTX.strokeStyle = '#3fb950';
        CTX.fillStyle = 'rgba(63, 185, 80, 0.2)';
        CTX.font = '16px Outfit, sans-serif';
        
        finalBoxes.forEach(b => {
             CTX.strokeRect(b.x1, b.y1, b.w, b.h);
             CTX.fillRect(b.x1, b.y1, b.w, b.h);
             // Label text background
             CTX.fillStyle = '#3fb950';
             CTX.fillRect(b.x1, b.y1 - 25, 95, 25);
             CTX.fillStyle = '#fff';
             CTX.fillText(`SHIP ${(b.conf*100).toFixed(0)}%`, b.x1 + 4, b.y1 - 7);
             CTX.fillStyle = 'rgba(63, 185, 80, 0.2)'; // reset for next box
        });
        
        updateStatus(`Inference Complete. Ships Detected: ${finalBoxes.length}. Zero-Server Engine Offline.`, 100);
        STATUS_TEXT.classList.add('success');
        
    } catch(err) {
        console.error(err);
        updateStatus('Runtime error during tensor decoupling.', 0);
        STATUS_TEXT.classList.add('error');
    }
}
