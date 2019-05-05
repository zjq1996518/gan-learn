import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import numpy as np
import matplotlib.pyplot as plt


class GAN(object):

    def __init__(self, is_train=True):

        self.BATCH = 128
        self.EPOCHS = 20000
        self.NOISE_DIM = 128
        if is_train:
            self.generator = self.get_generator()
            self.discriminator = self.get_discriminator()
            self.combine = self.get_combine()
            self.data = self.get_data()
        else:
            self.generator = keras.models.load_model('model/gan_model.h5')

    # 获取MNIST 数据集
    def get_data(self):
        (x_train, _), (_, _) = keras.datasets.mnist.load_data(path='mnist.npz')
        x_train = x_train.reshape((-1, 28 * 28))

        # 将数据值范围调整为 -1 -- 1 以符合tanh值域
        x_train = x_train / (255 / 2) - 1
        return x_train

    '''
    定义生成器    
    '''
    def get_generator(self):
        # 定义生成器
        g_input = keras.Input([self.NOISE_DIM])
        g_dense = Dense(256)(g_input)
        g_dense = LeakyReLU()(g_dense)
        g_dense = Dense(512)(g_dense)
        g_dense = LeakyReLU()(g_dense)
        g_out = Dense(1024)(g_dense)
        g_out = LeakyReLU()(g_out)
        g_out = Dense(784, activation='tanh')(g_out)

        # 返回模型与模型输出，用于联合
        return Model(g_input, g_out)

    '''
    定义判别器    
    '''
    def get_discriminator(self):

        d_inputs = keras.Input([784])
        d_dense = Dense(512)(d_inputs)
        d_dense = LeakyReLU()(d_dense)
        d_dense = Dense(256)(d_dense)
        d_dense = LeakyReLU()(d_dense)
        d_out = Dense(1, activation='sigmoid')(d_dense)
        model = Model(d_inputs, d_out)
        model.compile(keras.optimizers.Adam(2e-4, 0.5), loss='binary_crossentropy')

        return model

    '''
    生成器与判别器结合
    '''
    def get_combine(self):
        combine_out = self.discriminator(self.generator.output)

        # 训练DG时冻结判别器权重，只训练生成器权重
        self.discriminator.trainable = False

        model = Model(self.generator.input, combine_out)
        model.compile(keras.optimizers.Adam(2e-4, 0.5), loss='binary_crossentropy')

        return model

    # 训练
    def train(self):
        # 标签
        valid = np.ones([self.BATCH, 1])
        fake = np.zeros([self.BATCH, 1])

        for i in range(self.EPOCHS):
            print(f'第{i}次训练')
            # 生成符合标准正态分布的随机噪声利用生成器生成图片训练判别器
            z = np.random.normal(0, 1, (self.BATCH, self.NOISE_DIM))
            generate_imgs = self.generator.predict(z)

            # 随机从训练集中挑选图片
            random_id = np.random.randint(0, self.data.shape[0], self.BATCH)

            d_loss_valid = self.discriminator.train_on_batch(self.data[random_id], valid)
            d_loss_fake = self.discriminator.train_on_batch(generate_imgs, fake)
            print(f'D损失为{(d_loss_valid + d_loss_fake) / 2}')

            # 训练生成器
            combine_loss = self.combine.train_on_batch(z, valid)
            print(f'DG损失为{combine_loss}')

    def gen_img(self, name):
        z = np.random.normal(0, 1, (32, self.NOISE_DIM))
        result = self.generator.predict(z)
        fig, axs = plt.subplots(4, 8)
        result = result.reshape([-1, 28, 28])
        cnt = 0
        for i in range(4):
            for j in range(8):
                axs[i, j].imshow(result[cnt, :, :], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1

        # fig 就是生成的图片
        fig.savefig(f'./img/{name}.png')
        plt.close()

    def save_generator(self):
        self.generator.save('model/gan_model.h5')


if __name__ == '__main__':
    gan = GAN()
    gan.train()
    gan.gen_img('gan_img')
    gan.save_generator()
