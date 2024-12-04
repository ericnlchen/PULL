from keras import layers

class AttentionLayer(layers.Layer):
    def __init__(self, filters, reduction_ratio=16, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.filters = filters
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        # Initialize layers here instead of in __init__
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(self.filters // self.reduction_ratio, activation='relu')
        self.dense2 = layers.Dense(self.filters, activation='sigmoid')
        self.reshape = layers.Reshape((1, 1, self.filters))
        self.multiply = layers.Multiply()
        # Mark the layer as "built"
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Squeeze
        se = self.global_avg_pool(inputs)
        se = self.dense1(se)
        se = self.dense2(se)
        se = self.reshape(se)
        
        # Excitation
        output = self.multiply([inputs, se])
        return output

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({
            "filters": self.filters,
            "reduction_ratio": self.reduction_ratio
        })
        return config
