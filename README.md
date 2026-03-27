# SAR Ship Detection Pipeline 🚢📡

A lightweight, high-performance edge-ready ship detection pipeline for Synthetic Aperture Radar (SAR) imagery. Built from the ground up, this system leverages YOLOv8-nano enhanced by a generative data engine, complete with an explainable white-box output, an adversarial defense layer against hostile radar interference, and a zero-server edge-deployment architecture.

---

## 🌟 Core Architecture & Pipeline Phases

This repository represents an end-to-end 6-phase scalable pipeline built for critical observation targets. 

### 1. Bulletproof MVP (YOLOv8 Core)
- **What it does:** Dynamically processes massive, raw HRSID data, automatically parsing COCO JSON formats into normalized YOLO labels.
- **Performance:** Trains a lightning-fast `YOLOv8-nano` anchor point, mapping high-confidence targets accurately over pure SAR noise.

### 2. GAN Synthetic Data Engine
- **What it does:** Extracts 280 localized 64x64 pixel "ship signatures" and trains a PyTorch Deep Convolutional GAN (DCGAN) on their raw structures.
- **Performance:** Programmatically synthesizes structurally sound radar anomalies and dynamically seamlessly injects them back into the YOLO training fold.

### 3. Concept-Based Explainability (White-Box)
- **What it does:** Strips away the "Black Box" nature of neural networking by calculating physical, programmatic metrics for every bounding box.
- **Metrics Extracted:** *Radar Cross-Section Intensity* (RCS), *Edge Sharpness/Laplacian Variance*, and *Geometric Aspect Ratio* are structurally printed on the visual bounding box.

### 4. Zero-Server Edge Deployment 
- **What it does:** Bypasses expensive backend GPU routing. The `best.pt` model is flattened into an agnostic `.onnx` standalone graph.
- **Performance:** Utilizing `onnxruntime-web`, the model is executed entirely in a standard browser utilizing raw CPU WebAssembly (WASM) instructions inside a sleek Glassmorphism HTML/JS frontend.

### 5. Adversarial Defense Layer
- **What it does:** Protects the model architecture from artificial noise jamming and adversarial speckle attacks designed to scramble the latent space.
- **Performance:** Implements a localized adaptive spatial defense mechanism (`defense_layer.py`), successfully filtering injected salt-and-pepper hostility and fully restoring model object detection fidelity before it reaches the model tensor.

### 6. Sentinel Scale Inference (Sliding Window) 
- **What it does:** Empowers the lightweight MVP to scale up to massive, megapixel arrays recorded by Sentinel-1 satellites. 
- **Performance:** Automatically slices 800x800 tiling arrays over the landscape, bypassing memory-starvation, mathematically terminating "ghost anchors", and stitching targets accurately using global Non-Maximum Suppression (NMS).

---

## 📁 Repository Structure

* **`data_pipeline.py`** - Core dataset parsing (COCO -> YOLO normalized translation).
* **`train_mvp.py`** - Model execution and training loop.
* **`inference.py`** - The primary white-box testing ground. Overlays physical RCS and Laplacian metrics.
* **`extract_chips.py`**, **`gan_engine.py`**, **`augment_dataset.py`** - The Generative Data pipeline used to create realistic adversarial anomalies.
* **`defense_layer.py`** - Simulates an adversarial hit and restores output mapping via an adaptive spatial filter.
* **`export_onnx.py`** - Model compression script for web architecture.
* **`sentinel_inference.py`** - Massive multi-megapixel image scaling code using a tiled overlap mechanism. 
* **`web/`** - Glassmorphism UI for our *Zero-Server WebAssembly* deployment.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed, then clone the repository and run:

```bash
pip install -r requirements.txt
```

### Try the Zero-Server Interface Locally

Our web app executes the ONNX graph entirely on your local CPU without sending any data over the internet.

1. Open your terminal and navigate to the `web` folder.
2. Boot Python's local server:
   ```bash
   python -m http.server 8000
   ```
3. Open a browser and navigate to `http://localhost:8000`.
4. Drag-and-drop any `.jpg` SAR image sample to watch instant inference.

### Run Sentinel Scene Mapping

To process a massive image using our sliding-window scaling mechanism:
1. Provide a massive target (e.g. `sentinel2_scene.jpg`).
2. Execute the inference:
```bash
python sentinel_inference.py
```
3. Check the root folder for the stitched output: `sentinel2_final_output.jpg`.

---

**Built for Cosmix by The Boogiemen.**
