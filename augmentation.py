# =========================================
# 2. 데이터셋 객체 & SpecAugment
# =========================================
import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE
INPUT_SHAPE = (128, 200, 1)       # (freq, time, channel)

# ── SpecAugment 함수 ──────────────────────────────────
def spec_augment(spec, max_freq_mask=8, max_time_mask=24):
    """spec: [128, 200, 1]  (float32, dB)"""
    spec = tf.squeeze(spec, -1)          # [128, 200]

    # ▸ Frequency mask
    f = tf.random.uniform([], 0, max_freq_mask, dtype=tf.int32)
    f0 = tf.random.uniform([], 0, 128 - f, dtype=tf.int32)
    mask_f = tf.concat([
        tf.ones([f0, 200]),
        tf.zeros([f,  200]),
        tf.ones([128 - f0 - f, 200])
    ], axis=0)

    # ▸ Time mask
    t = tf.random.uniform([], 0, max_time_mask, dtype=tf.int32)
    t0 = tf.random.uniform([], 0, 200 - t, dtype=tf.int32)
    mask_t = tf.concat([
        tf.ones([128, t0]),
        tf.zeros([128, t]),
        tf.ones([128, 200 - t0 - t])
    ], axis=1)

    spec = spec * mask_f * mask_t
    return tf.expand_dims(spec, -1)      # [128, 200, 1]

def add_noise(spec, noise_level=0.1):
    noise = tf.random.normal(shape=tf.shape(spec), mean=0.0, stddev=noise_level, dtype=tf.float32)
    return spec + noise

# ── tf.data 파이프라인 ──────────────────────────────
def make_dataset(X, y=None, training=False, batch=32):
    ds = tf.data.Dataset.from_tensor_slices((X, y)) if y is not None else tf.data.Dataset.from_tensor_slices(X)

    if training:
        ds = ds.shuffle(len(X)).map(lambda spec, lbl:
            (add_noise(spec_augment(spec)), lbl),  # ← 여기에 Noise 추가
            num_parallel_calls=AUTOTUNE
        )

    ds = ds.batch(batch).prefetch(AUTOTUNE)
    return ds
