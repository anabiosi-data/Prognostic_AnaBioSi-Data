"""Shared helpers for BOC external inference (MM-ViT16 + MM-Transformer-Attention).

Requires TensorFlow / Keras 2.15 (the version all models were saved with).
"""
import io
import json
import os
import tempfile
import zipfile

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras import layers, regularizers

CLASS_NAMES = ["response", "stable", "non-response"]


# ─── Custom layers ────────────────────────────────────────────────────────
class TokenSlice(layers.Layer):
    """x[:, index, :] — replaces the ViT16 `cls_pick` Lambda."""

    def __init__(self, index=0, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, x):
        return x[:, self.index, :]

    def get_config(self):
        return {**super().get_config(), "index": self.index}


class CLSToken(layers.Layer):
    """Learnable [CLS] token (ViT16 image-only model)."""

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        self.cls = self.add_weight(name="cls", shape=(1, 1, self.dim),
                                   initializer="random_normal", trainable=True)

    def call(self, x):
        batch = tf.shape(x)[0]
        return tf.concat([tf.tile(self.cls, [batch, 1, 1]), x], axis=1)

    def get_config(self):
        return {**super().get_config(), "dim": self.dim}


class ParametrisedCompatibility(layers.Layer):
    """Attention compatibility layer (MM-Transformer-Attention), copied from training."""

    def __init__(self, kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        C = input_shape[0][-1]
        self.u = self.add_weight(name="u", shape=(C, 1), initializer="uniform",
                                 regularizer=self.kernel_regularizer, trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        local, g = inputs
        g_proj = K.expand_dims(K.expand_dims(g, 1), 1)
        combined = tf.nn.tanh(local + g_proj)
        return K.dot(combined, self.u)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2], 1)

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_regularizer": regularizers.serialize(self.kernel_regularizer)})
        return config


class AttendedPool(layers.Layer):
    """squeeze(batch_dot(expand_dims(a, 1), f), 1) — replaces the TA `attended_*` Lambdas."""

    def call(self, z):
        return K.squeeze(K.batch_dot(K.expand_dims(z[0], 1), z[1]), 1)


class AddConstant(layers.Layer):
    """x + fixed constant. Holds the ViT16 positional embedding, which the
    original model baked into the graph as a constant input of `add_pos`
    (it is NOT in the weight file, only in config.json)."""

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = np.asarray(value, dtype=np.float32)
        self._const = tf.constant(self.value)

    def call(self, x):
        return x + self._const

    def get_config(self):
        return {**super().get_config(), "value": self.value.tolist()}


CUSTOM_OBJECTS = {
    "TokenSlice": TokenSlice,
    "CLSToken": CLSToken,
    "ParametrisedCompatibility": ParametrisedCompatibility,
    "Custom>ParametrisedCompatibility": ParametrisedCompatibility,
    "AttendedPool": AttendedPool,
    "AddConstant": AddConstant,
}


# ─── Loading ──────────────────────────────────────────────────────────────
def _patch_constant_adds(obj):
    """Turn `Add([tensor, _CONSTANT_VALUE])` into `AddConstant(value)(tensor)`."""
    n = 0
    if isinstance(obj, dict):
        if obj.get("class_name") == "Add" and "_CONSTANT_VALUE" in json.dumps(obj.get("inbound_nodes", []))[:200]:
            node = obj["inbound_nodes"][0]
            tensors = [e for e in node if e[0] != "_CONSTANT_VALUE"]
            consts = [e for e in node if e[0] == "_CONSTANT_VALUE"]
            assert len(tensors) == 1 and len(consts) == 1, obj["name"]
            keep = {k: obj["config"][k] for k in ("name", "trainable", "dtype") if k in obj["config"]}
            obj.update(class_name="AddConstant", module=None, registered_name="AddConstant",
                       config={**keep, "value": consts[0][2]}, inbound_nodes=[tensors])
            obj.pop("build_config", None)
            n += 1
        for v in obj.values():
            n += _patch_constant_adds(v)
    elif isinstance(obj, list):
        for v in obj:
            n += _patch_constant_adds(v)
    return n


def _patch_lambdas(obj):
    """Replace serialized Python-bytecode Lambdas with equivalent real layers."""
    n = 0
    if isinstance(obj, dict):
        if obj.get("class_name") == "Lambda":
            name = obj["config"]["name"]
            keep = {k: obj["config"][k] for k in ("name", "trainable", "dtype") if k in obj["config"]}
            if name == "cls_pick":
                obj.update(class_name="TokenSlice", module=None, registered_name="TokenSlice",
                           config={**keep, "index": 0})
                n += 1
            elif name.startswith("attended_"):
                obj.update(class_name="AttendedPool", module=None, registered_name="AttendedPool",
                           config=keep)
                n += 1
            else:
                raise ValueError(f"Unknown Lambda layer '{name}' — add a replacement")
        for v in obj.values():
            n += _patch_lambdas(v)
    elif isinstance(obj, list):
        for v in obj:
            n += _patch_lambdas(v)
    return n


def load_keras_model(path, patch=True):
    """Load a Keras 2.15 `.keras` file.

    The `.keras` file MUST be used (not `.weights.h5`): the ViT16 positional
    embedding is a fixed constant stored only in config.json.
    With patch=True, Lambda layers (marshalled Python 3.11 bytecode) are
    swapped for equivalent custom layers so loading does not depend on the
    Python version.
    """
    if not patch:
        return tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS,
                                          compile=False, safe_mode=False)
    with zipfile.ZipFile(path) as z:
        cfg = json.loads(z.read("config.json"))
        n = _patch_lambdas(cfg) + _patch_constant_adds(cfg)
        tmp = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
        tmp.close()
        with zipfile.ZipFile(tmp.name, "w", compression=zipfile.ZIP_STORED) as out:
            for item in z.namelist():
                data = json.dumps(cfg).encode() if item == "config.json" else z.read(item)
                out.writestr(item, data)
    try:
        model = tf.keras.models.load_model(tmp.name, custom_objects=CUSTOM_OBJECTS,
                                           compile=False, safe_mode=False)
    finally:
        os.unlink(tmp.name)
    return model, n


# ─── kPa normalisers ──────────────────────────────────────────────────────
class RankNormalizer:
    """Copied from MM-ViT16 run5 (needed to unpickle kpa_scaler_5.pkl)."""

    def __init__(self):
        self.train_sorted = None

    def fit(self, X_train):
        self.train_sorted = np.sort(np.asarray(X_train).flatten())
        return self

    def transform(self, X_new, within_cohort=False):
        arr = np.asarray(X_new).flatten()
        if within_cohort:
            ranks = (np.argsort(np.argsort(arr)) + 1).astype(np.float64)
            n = len(arr)
        else:
            ranks = np.searchsorted(self.train_sorted, arr, side="right").astype(np.float64)
            n = len(self.train_sorted)
        return ((ranks - 0.5) / n).reshape(-1, 1).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class ZScore:
    def __init__(self, mean, std):
        self.mean, self.std = float(mean), float(std)

    def transform(self, X):
        return ((np.asarray(X, dtype=np.float64).reshape(-1, 1) - self.mean) / self.std).astype(np.float32)


# ─── Image preprocessing ──────────────────────────────────────────────────
def load_rgb_uint8(path, background="white"):
    """Open an image and return HxWx3 uint8.

    background="white": composite RGBA onto white (matches training images,
    which are opaque with a white surround). background="drop": drop alpha
    (what Keras/PIL `convert('RGB')` does — transparent pixels keep whatever
    RGB is stored underneath).
    """
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        if background == "white":
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def to_training_grid(rgb_uint8):
    """Mimic flow_from_directory(target_size=(300, 400)): PIL resize to 400x300 (W x H)."""
    im = Image.fromarray(rgb_uint8)
    if im.size != (400, 300):
        im = im.resize((400, 300), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)  # 0..255, (300, 400, 3)


def prep_vit(x300):
    """MM-ViT16: tf.image.resize to 304x400, /255, clip."""
    x = tf.image.resize(x300, [304, 400])
    x = tf.clip_by_value(tf.cast(x, tf.float32) / 255.0, 0.0, 1.0)
    return x.numpy()


def prep_ta(x300):
    """MM-Transformer-Attention: 300x400, /255, adjust_contrast(0.5), clip."""
    x = tf.image.resize(x300, [300, 400])
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.image.adjust_contrast(x, 0.5)
    return tf.clip_by_value(x, 0.0, 1.0).numpy()
