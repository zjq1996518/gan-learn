# coding: utf-8
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import numpy as np
import matplotlib.pyplot as plt

BATCH = 128
EPOCHS = 20000
# 随机噪声特征量
NOISE_DIM = 128

# 获取MNIST 数据集
def getData():
    (x_train, _), (_, _) = keras.datasets.mnist.load_data(path='mnist.npz')
    x_train = x_train.reshape((-1, 28*28))

    # 将数据值范围调整为 -1 -- 1 以符合tanh值域
    x_train = x_train / (255/2) - 1
    return x_train


def generator():
    # 定义生成器
    g_inputs = keras.Input([NOISE_DIM])
    g_dense = Dense(256)(g_inputs)
    g_dense = LeakyReLU()(g_dense)
    g_dense = Dense(512)(g_dense)
    g_dense = LeakyReLU()(g_dense)
    g_out = Dense(1024)(g_dense)
    g_out = LeakyReLU()(g_out)
    g_out = Dense(784, activation='tanh')(g_out)

    # 返回模型与模型输出，用于联合
    return Model(g_inputs, g_out)


# 定义判别器
def discriminator():
    d_inputs = keras.Input([784])
    d_dense = Dense(512)(d_inputs)
    d_dense = LeakyReLU()(d_dense)
    d_dense = Dense(256)(d_dense)
    d_dense = LeakyReLU()(d_dense)
    d_out = Dense(1, activation='sigmoid')(d_dense)
    D = Model(d_inputs, d_out)
    D.compile(keras.optimizers.Adam(2e-4, 0.5), loss='binary_crossentropy')

    return D

def union(D, g_inputs, g_out):

    # 生成器与判别器结合
    dg_out = D(g_out)

    # 训练DG时冻结判别器权重，只训练生成器权重
    D.trainable = False

    DG = Model(g_inputs, dg_out)
    DG.compile(keras.optimizers.Adam(2e-4, 0.5), loss= 'binary_crossentropy')

    return DG


# 训练
def train(data, D, DG):

    # 标签
    true = np.ones([BATCH, 1])
    false = np.zeros([BATCH, 1])

    for i in range(EPOCHS):

        print(f'第{i}次训练')
        # 生成符合标准正态分布的随机噪声利用生成器生成图片训练判别器
        z = np.random.normal(0, 1, (BATCH, NOISE_DIM))
        generate_imgs = G.predict(z)

        # 随机从训练集中挑选图片
        random_id = np.random.randint(0, data.shape[0], BATCH)

        d_loss_real = D.train_on_batch(data[random_id], true)
        d_loss_fake = D.train_on_batch(generate_imgs, false)
        print(f'D损失为{(d_loss_real+d_loss_fake)/2}')

        # 训练生成器
        dg_loss = DG.train_on_batch(z, true)
        print(f'DG损失为{dg_loss}')


def gen_img(generator, name):
    z = np.random.normal(0, 1, (32, NOISE_DIM))
    result = generator.predict(z)
    fig, axs = plt.subplots(4, 8)
    result = result.reshape([-1, 28, 28])
    cnt = 0
    for i in range(4):
        for j in range(8):
            axs[i,j].imshow(result[cnt, :, :], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1

    # fig 就是生成的图片
    fig.savefig(f'{name}.png')
    plt.close()


if __name__ == '__main__':

    data = getData()
    G = generator()
    D = discriminator()
    DG = union(D, G.input, G.output)
    train(data, D, DG)

    D.save('model/D.h5')
    G.save('model/G.h5')
    DG.save('model/DG.h5')

    gen_img('test')