// adversarial.js
import * as tf from "@tensorflow/tfjs";


export async function applyAdversarial(modelWrapper, imgTensor, onProgress) {
  const model = modelWrapper.model;
  const epsilon = 0.01;
  const steps = 5; 

  const initialPreds = model.predict(imgTensor);
  const labelIndex = initialPreds.argMax(1).dataSync()[0];
  initialPreds.dispose();

  let adv = imgTensor.clone();

  for (let i = 0; i < steps; i++) {
    // Pause execution for 1 frame 
    //  stops the "Page Unresponsive" crash
    await tf.nextFrame();

    // Update the progress bar if a callback was provided
    if (onProgress) {
      const percent = Math.round(((i + 1) / steps) * 100);
      onProgress(percent);
    }

    const grad = tf.tidy(() => {
      const getScore = (x) => {
        const preds = model.predict(x);
        return preds.gather([labelIndex], 1).squeeze(); 
      };
      const gradientFn = tf.grad(getScore);
      return gradientFn(adv);
    });

    const nextAdv = adv.sub(grad.sign().mul(epsilon));
    const clipped = nextAdv.clipByValue(0, 1);

    adv.dispose(); 
    adv = clipped;
  }

  return adv;
}