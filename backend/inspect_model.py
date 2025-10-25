# inspect_model.py  ------------------------ Inspect your model layers safely ------------------------
import tensorflow as tf                          # Import TensorFlow

# Step 1: Load your trained model
model = tf.keras.models.load_model('brain_mobilenet.h5')  # Load the saved model

# Step 2: Print all layer names and shapes safely
print("Listing all layers in the model:")
for i, layer in enumerate(model.layers):
    # Check if layer has 'output_shape' attribute
    if hasattr(layer, 'output_shape'):
        print(i, layer.name, layer.output_shape)     # Print index, name, and output shape
    else:
        print(i, layer.name, "No output_shape")      # Skip layers like InputLayer
