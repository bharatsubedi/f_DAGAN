import tensorflow as tf
from tensorflow.python.keras import layers

from src.nets import helper


class Generator(tf.keras.Model):
    def __init__(self, weight_init):
        super(Generator, self).__init__()
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

        self.attention = helper.Attention(32)

        self.dense = layers.Dense(7 * 7 * 64, use_bias=True, kernel_initializer=weight_init,
                                  kernel_regularizer=helper.orthogonal_regularizer_fully())

        self.res_up0 = helper.ResBlockUp(32, weight_init, use_bias=False)
        self.res_up1 = helper.ResBlockUp(32, weight_init, use_bias=False)
        self.conv = layers.Conv2DTranspose(3, 3, 1, 'SAME', use_bias=False, kernel_initializer=weight_init,
                                           kernel_regularizer=helper.orthogonal_regularizer_fully())

    def call(self, inputs, training=False, **kwargs):
        x = self.dense(inputs)  # 4*4*4*ch
        x = tf.reshape(x, shape=[-1, 7, 7, 64])  # [-1, 7, 7, 7*ch]

        x = self.res_up0(x, training=training)  # 14*14 14*ch
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x1 = x
        # non-local layer
        x = self.attention(x, training=training)

        x = self.res_up1(x, training=training)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv(x)

        # Output layer
        x = tf.nn.sigmoid(x)

        return x, x1


class Encoder(tf.keras.Model):
    def __init__(self, weight_init):
        super(Encoder, self).__init__()
        self.attention = helper.Attention(32)
        self.res_down0 = helper.ResBlockDown(32, weight_init, False)
        self.res0 = helper.ResBlock(32, weight_init, False)

    def call(self, inputs, training=True, **kwargs):
        x = self.res_down0(inputs)
        x = self.attention(x, training=training)

        x = self.res0(x)

        return x


class DiscriminatorFeature(tf.keras.Model):
    def __init__(self, weight_init):
        super(DiscriminatorFeature, self).__init__()

        self.attention = helper.Attention(64)
        self.res = helper.ResBlock(128, weight_init, False)
        self.dense = layers.Dense(1, use_bias=True, kernel_initializer=weight_init)

        self.res_down0 = helper.ResBlockDown(64, weight_init, False)
        self.res_down1 = helper.ResBlockDown(32, weight_init, False)
        self.res_down2 = helper.ResBlockDown(128, weight_init, False)

    def call(self, inputs, training=True, **kwargs):
        x = self.res_down0(inputs, training=training)

        # non-local layer
        x = self.attention(x, training=training)

        x = self.res_down1(x, training=training)
        x = self.res_down2(x, training=training)

        x = self.res(x, training=training)
        x = tf.nn.relu(x)

        x = tf.math.reduce_sum(x, axis=[1, 2])
        x = self.dense(x)

        return x


class DiscriminatorImage(tf.keras.Model):
    def __init__(self, weight_init):
        super(DiscriminatorImage, self).__init__()
        self.weight_init = weight_init
        self.attention = helper.Attention(64)

        self.res_down0 = helper.ResBlockDown(16, self.weight_init, False)
        self.res_down1 = helper.ResBlockDown(32, self.weight_init, False)
        self.res_down2 = helper.ResBlockDown(32, self.weight_init, False)
        self.res_down3 = helper.ResBlockDown(64, self.weight_init, False)
        self.res_down4 = helper.ResBlockDown(128, self.weight_init, False)
        self.res_down5 = helper.ResBlockDown(128, self.weight_init, False)
        self.res = helper.ResBlock(128, self.weight_init, False)
        self.dense = layers.Dense(units=1, use_bias=True, kernel_initializer=self.weight_init)

    def call(self, inputs, training=True, **kwargs):
        x = self.res_down0(inputs, training=training)
        x = self.res_down1(x, training=training)
        x = self.res_down2(x, training=training)
        x = self.res_down3(x, training=training)

        # non-local layer
        x = self.attention(x, training=training)

        x = self.res_down4(x, training=training)
        x = self.res_down5(x, training=training)

        x = self.res(x)
        x = tf.nn.relu(x)

        x = tf.math.reduce_sum(x, axis=[1, 2])

        x = self.dense(x)

        return x


# v1 loss
# def discriminator_loss(real, fake):
#     real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real))
#     fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake))

#     loss = real_loss + fake_loss

#     return loss
# v2 loss
# def discriminator_loss(real, fake):
#     real_loss = -tf.reduce_mean(real)
#     fake_loss = tf.reduce_mean(fake)
#     loss=real_loss+fake_loss
#     return loss

#v3 loss lsgan
# def discriminator_loss(real, fake):
#     real_loss = tf.reduce_mean(tf.math.squared_difference(real, 1.0))
#     fake_loss = tf.reduce_mean(tf.math.square(fake))

#     loss = real_loss + fake_loss
#     return loss

# #v4 loss dragan
def discriminator_loss(real, fake):
    alpha =0.9
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    loss = real_loss + fake_loss
    return loss

#v1 loss and v2 loss
# def generator_loss(fake):
#     loss = -tf.reduce_mean(fake)
#     return loss
#v3 loss lsgan
# def generator_loss(fake):
#     loss = tf.reduce_mean(tf.math.squared_difference(fake, 1.0))
#     return loss
# #v4 loss dragan
def generator_loss(fake):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))
    return loss