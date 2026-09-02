import os
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# CROP GUARD - PLANT DISEASE MODEL TRAINING
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = (
    BASE_DIR
    / "PlantVillage-Dataset"
    / "raw"
    / "color"
)

MODEL_DIR = BASE_DIR / "backend" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "cropguard_model.keras"
CLASS_PATH = MODEL_DIR / "class_names.json"


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 12

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# CHECK DATASET
# ============================================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_DIR}\n"
    )

print("\n" + "=" * 60)
print("CROPGUARD ML TRAINING")
print("=" * 60)

print(f"\nDataset: {DATASET_DIR}")


# ============================================================
# FIND CLASSES
# ============================================================

class_names = sorted(
    [
        folder.name
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    ]
)

NUM_CLASSES = len(class_names)

print(f"\nClasses found: {NUM_CLASSES}")

if NUM_CLASSES != 38:
    print(
        f"WARNING: Expected 38 classes, "
        f"but found {NUM_CLASSES}."
    )

for i, name in enumerate(class_names):
    print(f"{i:02d} -> {name}")


# ============================================================
# COLLECT ALL IMAGES
# ============================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG",
}

image_paths = []
labels = []

print("\nScanning dataset...")

for class_index, class_name in enumerate(class_names):

    class_dir = DATASET_DIR / class_name

    files = [
        p for p in class_dir.rglob("*")
        if p.is_file() and p.suffix in VALID_EXTENSIONS
    ]

    print(
        f"{class_name[:55]:55s} : {len(files)} images"
    )

    for file_path in files:
        image_paths.append(str(file_path))
        labels.append(class_index)


image_paths = np.array(image_paths)
labels = np.array(labels)

print("\n" + "-" * 60)
print(f"Total images: {len(image_paths)}")
print("-" * 60)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=SEED,
    stratify=labels,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp,
)

print("\nDataset split:")
print(f"Train      : {len(X_train)}")
print(f"Validation : {len(X_val)}")
print(f"Test       : {len(X_test)}")


# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(CLASS_PATH, "w") as f:
    json.dump(class_names, f, indent=4)

print(f"\nClass mapping saved to:")
print(CLASS_PATH)


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train,
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}

print("\nClass weights calculated.")


# ============================================================
# IMAGE PIPELINE
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(image, tf.float32)

    return image, label


def make_dataset(paths, labels, shuffle=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(paths), 10000),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


train_ds = make_dataset(
    X_train,
    y_train,
    shuffle=True
)

val_ds = make_dataset(
    X_val,
    y_val
)

test_ds = make_dataset(
    X_test,
    y_test
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.12
        ),

        tf.keras.layers.RandomZoom(
            0.15
        ),

        tf.keras.layers.RandomContrast(
            0.10
        ),

        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),
    ],
    name="crop_augmentation"
)


# ============================================================
# BASE MODEL
# ============================================================

print("\nLoading MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),

    include_top=False,

    weights="imagenet"
)

base_model.trainable = False


# ============================================================
# MODEL
# ============================================================

inputs = tf.keras.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dropout(0.35)(x)

x = tf.keras.layers.Dense(
    256,
    activation="relu"
)(x)

x = tf.keras.layers.Dropout(0.25)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = tf.keras.Model(
    inputs,
    outputs,
    name="CropGuard_MobileNetV2"
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(

    ),

    metrics=[
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=3,
            name="top3_accuracy"
        ),
    ],
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_PATH),

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",

        patience=4,

        mode="max",

        restore_best_weights=True,

        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",

        factor=0.3,

        patience=2,

        min_lr=1e-7,

        verbose=1
    ),
]


# ============================================================
# PHASE 1 - TRANSFER LEARNING
# ============================================================

print("\n" + "=" * 60)
print("PHASE 1 - TRANSFER LEARNING")
print("=" * 60)

history1 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=INITIAL_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks
)


# ============================================================
# PHASE 2 - FINE TUNING
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2 - FINE TUNING")
print("=" * 60)

base_model.trainable = True


# Keep early MobileNet layers frozen.
# Fine-tune only the deeper layers.

fine_tune_from = 100

for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False


model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(

    ),

    metrics=[
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=3,
            name="top3_accuracy"
        ),
    ],
)


model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=FINE_TUNE_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

results = model.evaluate(
    test_ds,
    verbose=1
)

for name, value in zip(
    model.metrics_names,
    results
):
    print(
        f"{name}: {value:.4f}"
    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_PATH
)


print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("\nModel:")
print(MODEL_PATH)

print("\nClasses:")
print(CLASS_PATH)

print("\nCropGuard ML model is ready.")
