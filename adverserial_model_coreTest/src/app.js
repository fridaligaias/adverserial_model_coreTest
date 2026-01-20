// app.js
import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import { applyAdversarial } from "./adversarial";

const progressContainer = document.getElementById("progressContainer");
const progressBar = document.getElementById("progressBar");

const selectBtn = document.getElementById("selectBtn");
const submitBtn = document.getElementById("submitBtn");
const gallery = document.getElementById("gallery");
const canvas = document.getElementById("outputCanvas");
const downloadLink = document.getElementById("downloadLink");

let selectedImage = null;
let model = null;

const images = [
  { name: "cat108.jpg", src: new URL("./assets/gallery/cat108.jpg", import.meta.url).href },
  { name: "dog.png",    src: new URL("./assets/gallery/dog.png", import.meta.url).href },
  { name: "woman1.jpg", src: new URL("./assets/gallery/woman1.jpg", import.meta.url).href },
  { name: "woman2.jpg", src: new URL("./assets/gallery/woman2.jpg", import.meta.url).href }
];

// Load MobileNetV2
async function loadModel() {
  console.log("Loading MobileNetV2 model...");
  // We wait for the model to load before enabling interaction
  model = await mobilenet.load({ version: 2, alpha: 1.0 });
  console.log("Model loaded");
  selectBtn.disabled = false; // Enable button only after model loads
  selectBtn.textContent = "Select Image";
}

selectBtn.onclick = () => {
  gallery.hidden = false;
};

images.forEach(image => {
  const img = document.createElement("img");
  img.src = image.src;
  img.alt = image.name;
  img.width = 150;
  img.height = 150;
  img.crossOrigin = "anonymous"; // Important for canvas security!

  img.onclick = () => {
    document.querySelectorAll("#gallery img").forEach(i => i.classList.remove("selected"));
    img.classList.add("selected");
    selectedImage = img;
    submitBtn.disabled = false;
  };

  gallery.appendChild(img);
});

// Step 3: Submit button click
submitBtn.onclick = async () => {
  if (!selectedImage) return;

  submitBtn.textContent = "Processing...";
  submitBtn.disabled = true;

  progressBar.style.width = "0%";
  progressContainer.style.display = "block";
  
  // Wait a tick so the UI updates
  setTimeout(async () => {
    console.log(`Processing: ${selectedImage.alt}`);
    // 1. Create Tensor and RESIZE to 224x224 (Required by MobileNet)
    const originalTensor = tf.browser.fromPixels(selectedImage)
      .resizeBilinear([224, 224]) // <--- The Critical Fix
      .toFloat()
      .div(255)
      .expandDims();
    // Pass the callback function to update the bar

    // 2. Run Adversarial Attack;
    const advTensor = await applyAdversarial(model, originalTensor, (percent) => {
        console.log(`Attack Progress: ${percent}%`);
        progressBar.style.width = `${percent}%`;
        });

    await tf.browser.toPixels(advTensor.squeeze(), canvas);

    // 4. Offer Download
    downloadLink.href = canvas.toDataURL();
    downloadLink.download = `adversarial_${selectedImage.alt}.png`;
    downloadLink.hidden = false;
    downloadLink.textContent = "Download Adversarial Image";
    
    // Hide progress bar when done
    progressContainer.style.display = "none";
    submitBtn.textContent = "Run Attack";
    submitBtn.disabled = false;

    // Clean up memory
    originalTensor.dispose();
    advTensor.dispose();
    
    submitBtn.textContent = "Run Attack";
    submitBtn.disabled = false;
  }, 100);
};

// Start
selectBtn.disabled = true;
selectBtn.textContent = "Loading Model...";
loadModel();