from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, InputLayer

# Tiny model for testing
model = Sequential([
    InputLayer(input_shape=(64, 64, 3)),
    Flatten(),
    Dense(1, activation='sigmoid')
])

# Save the tiny model
model.save('model.h5')
