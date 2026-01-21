// app.js
// og image -> model inference -> adversarial attack -> 
// adversarial image -> model inference -> deceptive result + download link


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

// Create a container for results if it doesn't exist
const resultsContainer = document.createElement("div");
resultsContainer.id = "results";
resultsContainer.style.marginTop = "20px";
resultsContainer.style.fontFamily = "monospace";
document.body.appendChild(resultsContainer);

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
  model = await mobilenet.load({ version: 2, alpha: 1.0 });
  console.log("Model loaded");
  selectBtn.disabled = false;
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
  img.crossOrigin = "anonymous"; 

  img.onclick = () => {
    document.querySelectorAll("#gallery img").forEach(i => i.classList.remove("selected"));
    img.classList.add("selected");
    selectedImage = img;
    submitBtn.disabled = false;
    resultsContainer.innerHTML = ""; // Clear previous results
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height); // Clear canvas
  };

  gallery.appendChild(img);
});

// --- MAIN LOGIC HERE ---
submitBtn.onclick = async () => {
  if (!selectedImage) return;

  submitBtn.textContent = "Processing...";
  submitBtn.disabled = true;
  progressBar.style.width = "0%";
  progressContainer.style.display = "block";
  resultsContainer.innerHTML = "Inferencing original image...";
  
  // Use setTimeout to allow UI to render the "Processing" state
  setTimeout(async () => {
    try {
      // 1. Prepare Tensor
      // Tensor for classification (NO normalisation)
      const classifyTensor = tf.browser.fromPixels(selectedImage)
        .resizeBilinear([224, 224])
        .toFloat()
        .expandDims();

      // Tensor for attack (normalised)
      const attackTensor = classifyTensor.div(255);
      // 2. INFERENCE (Original)
      const originalPreds = await model.classify(classifyTensor);
      const before = originalPreds[0];      
      console.log("Original Prediction:", before);

      // 3. RUN ATTACK (BIM)
      // No UI predictions inside this loop, just the progress bar callback
      const advTensor = await applyAdversarial(model, attackTensor, (percent) => {
        progressBar.style.width = `${percent}%`;
      });

      // 4. INFERENCE (Adversarial)
      // Denormalize back to 0-255 range for classification
      const advTensorDenorm = advTensor.mul(255);
      const advPreds = await model.classify(advTensorDenorm);
      const after = advPreds[0];
      console.log("Adversarial Prediction:", after);

      // 5. RENDER RESULT
      await tf.browser.toPixels(advTensor.squeeze(), canvas);

      // 6. CALCULATE METRICS
      
      // Calculate Confidence Drop:
      // We want: (Prob of Original Class in OLD image) - (Prob of Original Class in NEW image)
      // Note: 'after' is the top class of the new image. If the class flipped, 
      // we need to hunt for the original class name in the 'advPreds' array.
      const originalClassInNewPreds = advPreds.find(p => p.className === before.className);
      const newProbOfOriginalClass = originalClassInNewPreds ? originalClassInNewPreds.probability : 0;
      
      const confidenceDrop = Math.max(0, before.probability - newProbOfOriginalClass) * 100;
      const isClassFlip = before.className !== after.className;

      // 7. DISPLAY METRICS
      resultsContainer.innerHTML = `
        <h3>Deception Metrics</h3>
        <p><strong>Original:</strong> ${before.className} (${(before.probability * 100).toFixed(2)}%)</p>
        <p><strong>Adversarial:</strong> ${after.className} (${(after.probability * 100).toFixed(2)}%)</p>
        <hr>
        <p><strong>Class Flip:</strong> ${isClassFlip ? "✅ YES" : "❌ NO"}</p>
        <p><strong>Confidence Drop:</strong> ${confidenceDrop.toFixed(2)}%</p>
      `;

      // 8. DOWNLOAD SETUP
      downloadLink.href = canvas.toDataURL();
      downloadLink.download = `adversarial_${selectedImage.alt}`;
      downloadLink.hidden = false;
      downloadLink.textContent = "Download Adversarial Image";

      // Cleanup
      classifyTensor.dispose();
      attackTensor.dispose();
      advTensor.dispose();
      advTensorDenorm.dispose(); // Don't forget this!

    } catch (err) {
      console.error(err);
      resultsContainer.innerHTML = `<p style="color:red">Error: ${err.message}</p>`;
    } finally {
      progressContainer.style.display = "none";
      submitBtn.textContent = "Run Attack";
      submitBtn.disabled = false;
    }
  }, 100);
};

// Start
selectBtn.disabled = true;
selectBtn.textContent = "Loading Model...";
loadModel(); 