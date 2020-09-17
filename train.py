import os
from os.path import join

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from src.utils import config
from src.utils.dataset import load_data
from src.nets import nn

weight_init = tf.initializers.orthogonal()
g_model = nn.Generator(weight_init)
d_model1 = nn.DiscriminatorFeature(weight_init)
d_model2 = nn.DiscriminatorImage(weight_init)
encoder = nn.Encoder(weight_init)
train_data,epoch = load_data()
d_opt = tf.optimizers.Adam(config.d_lr, config.beta1, config.beta2)
g_opt = tf.optimizers.Adam(config.g_lr, config.beta1, config.beta2)

if not os.path.exists(config.weight):
    os.makedirs(config.weight)

    
#@tf.function
def train_step(_image1, _image2, _noise):
    with tf.GradientTape(persistent=True) as g_tape:
        feature1 = encoder(_image1, training=True)
        #print('encoder feature1::', feature1.shape)
        feature2 = encoder(_image2, training=True)
        #print('encoder feature2::', feature2.shape)
        fake_image, fake_feature = g_model(_noise, training=True)
        #print('generator fake_image::', fake_image.shape)
        #print('generator fake_feature::', fake_feature.shape)
        feature = tf.keras.layers.Average()([feature1,fake_feature])
        #print('generator fake_feature::', feature.shape)
        #print(tf.concat([feature, feature1], -1).shape)
        with tf.GradientTape(persistent=True) as d_tape:
            fake_feature_logits = d_model1(tf.concat([feature, feature1], -1), training=True)
            real_feature_logits = d_model1(tf.concat([feature1, feature2], -1), training=True)
            _d_loss1 = nn.discriminator_loss(real_feature_logits, fake_feature_logits)
            _g_loss1 = nn.generator_loss(fake_feature_logits)

            real_image_logits = d_model2(tf.concat([image1, image2], -1), training=True)
            fake_image_logits = d_model2(tf.concat([image1, fake_image], -1), training=True)
            _d_loss2 = nn.discriminator_loss(real_image_logits, fake_image_logits)
            _g_loss2 = nn.generator_loss(fake_image_logits)

            _d_loss = _d_loss1 + _d_loss2
            _g_loss = _g_loss1 + _g_loss2

        trainable_variables = d_model1.trainable_variables + d_model2.trainable_variables
        d_grads = d_tape.gradient(_d_loss, trainable_variables)
        d_opt.apply_gradients(zip(d_grads, trainable_variables))

    trainable_variables = encoder.trainable_variables + g_model.trainable_variables
    g_grads = g_tape.gradient(_g_loss, trainable_variables)
    g_opt.apply_gradients(zip(g_grads, trainable_variables))

    return _d_loss, _g_loss


for step, (image1, image2) in enumerate(train_data):
    step += 1
    noise = tf.random.truncated_normal([config.batch_size, config.num_cont_noise])
    d_loss, g_loss = train_step(image1, image2, noise)
    print('step::', step, 'd_loss::', d_loss.numpy(), 'g_loss', g_loss.numpy())
    if step % (50000 // config.batch_size) == 0:
        g_model.save_weights(join('weights', 'g_model.tf'))
        d_model2.save_weights(join('weights', 'd_model.tf'))
