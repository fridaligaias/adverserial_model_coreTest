// adversarial.js
import * as tf from "@tensorflow/tfjs";

export async function applyAdversarial(modelWrapper, imgTensor) {
  // 1. Get the raw model from the MobileNet wrapper
  // The wrapper doesn't support gradients, but the internal graph model does.
  const model = modelWrapper.model;

  const epsilon = 0.05; // Strength of noise (0.01 is subtle, 0.05 is visible)
  const steps = 10;     // Number of iterations

  // 2. Identify the "correct" class we want to attack
  // We predict once to find out what the model thinks this is (e.g., "Tabby Cat")
  const initialPreds = model.predict(imgTensor);
  const labelIndex = initialPreds.argMax(1).dataSync()[0]; // Get index of top class

  let adv = imgTensor.clone();

  for (let i = 0; i < steps; i++) {
    // tf.tidy automagically cleans up all tensors created inside this block
    // except the one we return.
    const grad = tf.tidy(() => {
      // 3. Define the scalar function we want to differentiate
      const getScore = (x) => {
        const preds = model.predict(x);
        // We pick the probability score of the ORIGINAL class
        return preds.gather([labelIndex], 1).squeeze(); 
      };

      // 4. Calculate gradient of that score with respect to the input image
      const gradientFn = tf.grad(getScore);
      return gradientFn(adv);
    });

    // 5. Update the image: 
    // We SUBTRACT the gradient to minimize the confidence of the original class.
    // (Moving the image 'away' from looking like a cat)
    const nextAdv = adv.sub(grad.sign().mul(epsilon));
    
    // Clip pixels to stay within valid color range [0, 1]
    const clipped = nextAdv.clipByValue(0, 1);

    // Clean up the old tensor step and save the new one
    adv.dispose(); 
    adv = clipped;
  }

  return adv;
}
