# =========================================
# 3. Residual CNN 모델 정의
# =========================================
from tensorflow.keras import layers, models

def res_block(x, n_filters, dropout=0.2):
    """간단 Residual 블록: Conv → BN → ReLU 두 번 + skip connection."""
    shortcut = x
    if x.shape[-1] != n_filters:
        shortcut = layers.Conv2D(n_filters, (1,1), padding="same")(shortcut)

    x = layers.Conv2D(n_filters, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(n_filters, (3,3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([shortcut, x])
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)
    return x

def build_model(input_shape=INPUT_SHAPE):
    inp = layers.Input(shape=input_shape)

    x = layers.BatchNormalization()(inp)           # 0-mean, 1-std 안정화
    x = res_block(x, 32)
    x = layers.MaxPool2D((2,2))(x)                # 64×100
    x = res_block(x, 64)
    x = layers.MaxPool2D((2,2))(x)                # 32×50
    x = res_block(x, 128)
    x = layers.MaxPool2D((2,2))(x)                # 16×25
    x = res_block(x, 256, dropout=0.3)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inp, out)

model = build_model()
model.summary()
# =========================================
# 4. 학습 설정 & 콜백
# =========================================
BATCH_SIZE = 32
EPOCHS = 60

train_ds = make_dataset(X_train, y_train, training=True,  batch=BATCH_SIZE)
val_ds   = make_dataset(X_val,   y_val,   training=False, batch=BATCH_SIZE)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

cb = [
    tf.keras.callbacks.ModelCheckpoint("best_model.h5",
                                       monitor="val_accuracy",
                                       save_best_only=True,
                                       verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                         factor=0.5, patience=5,
                                         min_lr=1e-5, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                     patience=10, restore_best_weights=True)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,   # ↖️ 필요 없으면 None
    callbacks=cb,
    verbose=2
)