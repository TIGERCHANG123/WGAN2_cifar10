from WGAN_Block import *

class generator(tf.keras.Model):
  def __init__(self):
    super(generator, self).__init__()
    self.input_layer = generator_Input(shape=[4, 4, 512])

    self.middle_layer_list = [
      # generator_Middle(filters=512, strides=2),
      generator_Middle(filters=256, strides=2),
      generator_Middle(filters=128, strides=2),
      generator_Middle(filters=64, strides=2)
    ]

    self.output_layer = generator_Output(image_depth=3, strides=2)
  def call(self, x):
    x = self.input_layer(x)
    # x = self.middle_layer1(x)
    for i in range(len(self.middle_layer_list)):
      x = self.middle_layer_list[i](x)
    x = self.output_layer(x)
    return x

class discriminator(tf.keras.Model):
  def __init__(self):
    super(discriminator, self).__init__()
    self.input_layer = discriminator_Input(filters=64, strides=2)

    self.middle_layer_list = [
      discriminator_Middle(filters=128, strides=2),
      discriminator_Middle(filters=256, strides=2)
    ]
    self.output_layer = discriminator_Output(with_activation=False)

  def call(self, x):
    x = self.input_layer(x)
    for i in range(len(self.middle_layer_list)):
      x = self.middle_layer_list[i](x)
    x = self.output_layer(x)
    return x

def get_gan():
  Generator = generator()
  Discriminator = discriminator()
  gen_name = 'WGAN'
  return Generator, Discriminator, gen_name


