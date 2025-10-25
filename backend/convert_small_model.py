import tensorflow as tf

# Load your existing model
model = tf.keras.models.load_model("brain_mobilenet.h5")

# Save a lighter version (remove training metadata & optimizer)
model.save("brain_mobilenet_small.h5", include_optimizer=False)

print("✅ New smaller model saved as brain_mobilenet_small.h5")
