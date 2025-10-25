# train_mobilenet.py
import os, json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

DATASET_DIR = r"C:\Users\ADMIN\Downloads\dataset"   # Path to dataset (with 'yes' & 'no' subfolders)
OUTPUT_DIR = r"C:\Users\ADMIN\Desktop\brain_project" # Folder to save model and label map
IMG_SIZE = (224, 224)                                # Image resize target
BATCH = 16                                           # Batch size
EPOCHS_BASE = 8                                      # Initial training epochs
EPOCHS_FINETUNE = 5                                  # Fine-tuning epochs

# 🔸 Data Augmentation + Train/Validation split
datagen = ImageDataGenerator(
    rescale=1./255, validation_split=0.2,             # Split 80/20
    rotation_range=15, width_shift_range=0.1,
    height_shift_range=0.1, zoom_range=0.1,
    horizontal_flip=True, fill_mode='nearest'
)

# 🔸 Load train & validation sets
train_gen = datagen.flow_from_directory(
    DATASET_DIR, target_size=IMG_SIZE,
    batch_size=BATCH, class_mode='binary', subset='training'
)
val_gen = datagen.flow_from_directory(
    DATASET_DIR, target_size=IMG_SIZE,
    batch_size=BATCH, class_mode='binary', subset='validation'
)

# 🔸 Save label map (index → label)
index_to_label = {str(v): k for k, v in train_gen.class_indices.items()}
with open(os.path.join(OUTPUT_DIR, 'label_map.json'), 'w') as f:
    json.dump(index_to_label, f)
print("Saved label_map.json:", index_to_label)

# 🔸 Build model using MobileNetV2 backbone
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE,3))
base.trainable = False   # Freeze base layers initially

# 🔸 Add custom classification layers
x = GlobalAveragePooling2D()(base.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)  # Binary output
model = Model(base.input, outputs)

# 🔸 Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='binary_crossentropy', metrics=['accuracy'])

# 🔸 Define training callbacks (auto-save, stop, adjust LR)
callbacks = [
    ModelCheckpoint(os.path.join(OUTPUT_DIR, 'best_model.h5'), save_best_only=True, monitor='val_loss'),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# 🔸 Train frozen model first
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_BASE, callbacks=callbacks)

# 🔸 Unfreeze top layers for fine-tuning
base.trainable = True
for layer in base.layers[:-50]:   # Keep earlier layers frozen
    layer.trainable = False

# 🔸 Compile and fine-tune
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_FINETUNE, callbacks=callbacks)

# 🔸 Save final model
model.save(os.path.join(OUTPUT_DIR, 'brain_mobilenet.h5'))
print("Model saved successfully ✅")


# 🔸 Print last conv layer (needed for Grad-CAM)
for layer in reversed(model.layers):
    try:
        if len(layer.output.shape) == 4:
            print("Last Conv Layer for Grad-CAM:", layer.name)
            break
    except:
        continue



    
