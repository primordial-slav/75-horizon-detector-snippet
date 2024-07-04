import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

class UNetRegressionModel:
    def __init__(self):
        # Load the base model (EfficientNetB1) without the top layer
        self.base_model = tf.keras.applications.EfficientNetB1(input_shape=[192, 384, 3], include_top=False)
        
        # Layer names for feature extraction
        self.layer_names = [
            'block1b_add',      # 112x112 x16
            'block2c_add',      # 56x56   x24
            'block3c_add',      # 28x28   x40
            'block5d_add',      # 14x14   x112
            'block7b_add',      # 7x7     x320
        ]
        
        # Extract the outputs of the specified layers
        self.base_model_outputs = [self.base_model.get_layer(name).output for name in self.layer_names]
        
        # Create the feature extraction model
        self.down_stack = tf.keras.Model(inputs=self.base_model.input, outputs=self.base_model_outputs)
        self.down_stack.trainable = True
        
        # Define the upsampling stack
        self.up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),   # 32x32 -> 64x64
        ]

    def unet_model(self, output_channels: int):
        inputs = tf.keras.layers.Input(shape=[192, 384, 3])

        # Downsampling through the model
        skips = self.down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        mask = tf.keras.layers.Conv2DTranspose(
            filters=output_channels, kernel_size=3, strides=2,
            padding='same')(x)  # 64x64 -> 128x128
        
        pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(mask)
        x = tf.keras.layers.Concatenate(name='concat_mask')([x, pool])
        x = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            strides=(1, 1))(x)
        
        x = tf.keras.layers.Flatten()(x)

        # Regression layers
        dense_1 = tf.keras.layers.Dense(256, activation='relu')(x)
        dense_2 = tf.keras.layers.Dense(128, activation='relu')(dense_1)
        x1 = tf.keras.layers.Dense(1, activation='linear')(dense_2)

        dense_1 = tf.keras.layers.Dense(256, activation='relu')(x)
        dense_2 = tf.keras.layers.Dense(128, activation='relu')(dense_1)
        x2 = tf.keras.layers.Dense(1, activation='linear')(dense_2)
        
        x = tf.keras.layers.Concatenate(axis=-1)([x1, x2])
        
        return tf.keras.Model(inputs=inputs, outputs=x)