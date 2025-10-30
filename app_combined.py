@st.cache_resource
def load_bilstm_model():
    # --- Recreate the custom AttentionLayer so Keras can reload it ---
    from tensorflow.keras import backend as K
    import tensorflow as tf

    class AttentionLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(AttentionLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(
                name="att_weight",
                shape=(input_shape[-1], 1),
                initializer="normal"
            )
            self.b = self.add_weight(
                name="att_bias",
                shape=(input_shape[1], 1),
                initializer="zeros"
            )
            super(AttentionLayer, self).build(input_shape)

        def call(self, x):
            e = K.tanh(K.dot(x, self.W) + self.b)
            a = K.softmax(e, axis=1)
            output = x * a
            return K.sum(output, axis=1)

    # --- Load the model using the custom layer ---
    model = tf.keras.models.load_model(
        "bilstm_model.keras",
        custom_objects={"AttentionLayer": AttentionLayer},
        compile=False
    )
    tokenizer = joblib.load("tokenizer.joblib")
    label_meta = joblib.load("labels_meta.joblib")
    labels = label_meta["classes"]
    return model, tokenizer, labels
