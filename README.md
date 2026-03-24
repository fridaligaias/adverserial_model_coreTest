# 🛡️ Adversarial Image Perturbation Core
### Client-Side Adversarial AI Project | **womyn.network** Code Snippet 

Check out more about the womyn.project project [here](https://sites.google.com/view/womyn/home).

[![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)](https://developer.mozilla.org/)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-Client--Side-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/js)
[![Parcel](https://img.shields.io/badge/Bundler-Parcel-D4A76A?style=for-the-badge&logo=parcel&logoColor=black)](https://parceljs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## 🚀 Quick Access
Explore the core adversarial logic and the frontend implementation layer:

<p align="left">
  <a href="https://github.com/fridaligaias/adverserial_model_coreTest/blob/main/adverserial_model_coreTest/src/adversarial.js">
    <img src="https://img.shields.io/badge/Core_Algorithm-adversarial.js-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="Adversarial Logic" />
  </a>
  <a href="https://github.com/fridaligaias/adverserial_model_coreTest/blob/main/adverserial_model_coreTest/src/app.js">
    <img src="https://img.shields.io/badge/UI_Controller-app.js-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="App Logic" />
  </a>
</p>

---

## 📌 Project Overview
This repository contains the core image perturbation engine for **womyn.network**. It is a client-side privacy tool designed to protect user images from unwanted AI classification and scraping. 

By applying an imperceptible layer of adversarial noise directly in the browser, the module forces Convolutional Neural Networks (like MobileNetV2) to misclassify the image, acting as a digital "cloak" against unauthorized computer vision models.

### Key Technical Achievements:
* **Client-Side Execution:** Leverages TensorFlow.js to process image tensors entirely within the user's browser, ensuring raw photos never touch a remote server.
* **Untargeted Attack Vector:** Implements a gradient-based attack that iteratively decreases the model's confidence in the true image label.
* **Memory Management:** Utilizes strict `tf.tidy()` and `.dispose()` practices to prevent browser memory leaks during intensive tensor operations.
* **Async Rendering:** Implements frame-pausing (`await tf.nextFrame()`) during the iteration loop to prevent UI blocking and "Page Unresponsive" crashes.

---

## Algorithmic Foundation (I-FGSM)
My project applies the **Iterative Fast Gradient Sign Method (I-FGSM)**, also known as the Basic Iterative Method (BIM). Instead of taking one large step, the algorithm applies small, calculated perturbations over multiple steps ($\text{steps} = 5$).

The mathematical update rule for each iteration is:

$$x_{new} = \text{clip}(x - \epsilon \cdot \text{sign}(\nabla_x J))$$

Where:
* $x$ is the current input image tensor.
* $\epsilon$ (Epsilon) is the learning rate or perturbation magnitude ($\epsilon = 0.01$).
* $\nabla_x J$ is the gradient of the loss function with respect to the input image (the direction that increases the score of the original label).
* $\text{sign}$ reduces the gradient to its directional components ($-1, 0, 1$).

### Into the Code
The underlying mathematics are directly translated into vectorized TensorFlow.js tensor operations:

```javascript
// adversarial.js - Iterative Attack Loop
const grad = tf.tidy(() => {
  const getScore = (x) => { 
    const preds = model.predict(x);
    return preds.gather([labelIndex], 1).squeeze(); 
  };
  // Calculate the derivative (gradient) of the score
  const gradientFn = tf.grad(getScore);
  return gradientFn(adv);
});

// x_new = x - epsilon * sign(grad)
const nextAdv = adv.sub(grad.sign().mul(epsilon));

// Clip pixel values to maintain valid RGB bounds [0, 1]
const clipped = nextAdv.clipByValue(0, 1);
```

## 🛠️ System Architecture

* [![adversarial.js](https://img.shields.io/badge/src/adversarial.js-323330?style=for-the-badge&logo=javascript&logoColor=F7DF1E)](./src/adversarial.js)  
  **The mathematical core.** Handles gradient calculation, tensor mutation, and the iterative loop.

* [![app.js](https://img.shields.io/badge/src/app.js-323330?style=for-the-badge&logo=javascript&logoColor=F7DF1E)](adverserial_model_coreTest/src/app.js)  
  **The application layer.** Loads the pre-trained MobileNetV2 model, handles HTML Canvas interactions, resizes images to `[224, 224]`, and manages the download pipeline.

* [![index.html](https://img.shields.io/badge/src/index.html-E34F26?style=for-the-badge&logo=html5&logoColor=white)](./src/index.html) [![style.css](https://img.shields.io/badge/src/style.css-1572B6?style=for-the-badge&logo=css3&logoColor=white)](./src/style.css)  
  **The testing sandbox UI.** Features a gallery selector and progress visualization.

## 📥 Local Development & Testing
This project utilizes **Parcel** as its build tool for zero-configuration module bundling, but feel free to test!

### Install Dependencies:
```bash
npm install
```

### Start the Development Server:
```bash
npx parcel src/index.html
```

### Usage:
1. Open `http://localhost:1234` in your browser.
2. Wait for the MobileNetV2 model to load (takes seconds).
3. Select an image from the test gallery.
4. Click **Apply adversarial layer** to generate and download the cloaked image.

---

## Context
* **Project Ecosystem:** womyn.network
* **Focus:** AI Security, Adversarial Machine Learning, Client-Side Privacy
