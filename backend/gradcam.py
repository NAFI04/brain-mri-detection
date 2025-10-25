# gradcam.py
import numpy as np, cv2, tensorflow as tf

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create model that maps image -> activations of last conv layer + output
    grad_model = tf.keras.models.Model([model.inputs],
                                       [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = 0
        loss = predictions[:, pred_index] if predictions.shape[-1] > 1 else predictions[:, 0]

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(img_path, heatmap, output_path, alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(np.uint8(255 * heatmap), (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
    cv2.imwrite(output_path, overlay)
    return output_path
