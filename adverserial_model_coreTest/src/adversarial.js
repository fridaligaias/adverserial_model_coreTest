// adversarial.js
import * as tf from "@tensorflow/tfjs";

export async function applyAdversarial(modelWrapper, imgTensor, onProgress) {
  const model = modelWrapper.model;

  // Attack hyperparameters
  const epsilon = 0.05;       // total L∞ perturbation budget
  const steps = 5;            // number of iterations
  const alpha = epsilon / steps; // step size per iteration

  // Get initial predicted label (untargeted attack)
  const initialPreds = model.predict(imgTensor);
  const labelIndex = initialPreds.argMax(1).dataSync()[0];
  initialPreds.dispose();

  // Keep a copy of the original image for epsilon-ball projection
  const original = imgTensor.clone();
  let adv = imgTensor.clone();

  for (let i = 0; i < steps; i++) {
    // Yield to browser to avoid freezing
    await tf.nextFrame();

    if (onProgress) {
      const percent = Math.round(((i + 1) / steps) * 100);
      onProgress(percent);
    }

    // Compute gradient of the score for the original class
    const grad = tf.tidy(() => {
      const gradientFn = tf.grad(x => {
        const preds = model.predict(x);
        return preds.gather([labelIndex], 1).squeeze();
      });
      return gradientFn(adv);
    });

    // BIM / I-FGSM update with epsilon-ball projection
    const nextAdv = tf.tidy(() => {
      // Step in gradient direction
      const stepped = adv.sub(grad.sign().mul(alpha));

      // Project perturbation back into L∞ epsilon-ball
      const perturbation = stepped
        .sub(original)
        .clipByValue(-epsilon, epsilon);

      // Apply perturbation and clip to valid image range
      return original
        .add(perturbation)
        .clipByValue(0, 1);
    });

    adv.dispose();
    grad.dispose();
    adv = nextAdv;
  }

  original.dispose();
  return adv;
}
