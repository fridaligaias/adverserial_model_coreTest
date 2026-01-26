// adversarial.js
import * as tf from "@tensorflow/tfjs";


export async function applyAdversarial(modelWrapper, imgTensor, onProgress) {
  const model = modelWrapper.model;
  const epsilon = 0.01; // The strength of the noise (perturbation)
  const steps = 5; // How many times to apply the noise

  const initialPreds = model.predict(imgTensor);
  const labelIndex = initialPreds.argMax(1).dataSync()[0];
  initialPreds.dispose();
  // It predicts the class of the clean image.
  // save the labelIndex (e.g., "Panda"). 
  //  is to make the model stop believing this is a "Panda" 
  // (untargeted attack)

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
    // math
    // getScore: returns the model's confidence for the original label
    // tf.grad(getScore): TensorFlow automatically calculates the derivative (gradient).
    // The gradient indicates how to change the input image to reduce the model's confidence in the original label.
    // tf.tidy: Automatically cleans up intermediate tensors created inside the block to prevent memory leaks.
    const grad = tf.tidy(() => {
      const getScore = (x) => { 
        const preds = model.predict(x);
        return preds.gather([labelIndex], 1).squeeze(); 
      };
      const gradientFn = tf.grad(getScore);
      return gradientFn(adv);
    });

    // applying the perturbation - iterative Fast Gradient Sign Method (I-FGSM)/BIM
    // $$x_{new} = x - \epsilon \cdot \text{sign}(\nabla_x J)$$
    // where \(x\) is the input image, \(\epsilon\) is the perturbation strength, and \(\nabla_x J\) is the gradient of the loss with respect to the input image.
    // x - adv - current image tensor (input)
    // .sub() - subtraction - since gradient indicates how to increase the score for the original label, we subtract it to decrease the score.
    // $\nabla_x J$ - grad - gradient calculated previously, direction in pixel space that increases the score of the current label.
    // $\text{sign}$ - .sign() - Converts the gradient into just -1, 0, or 1. size of the gradient doesnt matter, only the direction.
    // $\cdot$ - .mul(epsilon) - scales the sign of the gradient by epsilon, controlling the perturbation strength.
    // $\epsilon$ - epsilon	- learning rate (or perturbation magnitude)
    const nextAdv = adv.sub(grad.sign().mul(epsilon));
    // Clip the values to be in [0, 1]
    const clipped = nextAdv.clipByValue(0, 1);

    adv.dispose(); 
    adv = clipped;
  }

  return adv;
}