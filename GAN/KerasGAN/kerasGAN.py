import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

imgRows = 28
imgCols = 28
imgChannels = 1

numOfData = 10
noiseSize = 100

def loadImage() :
    dataSet = []
    dataShape = (imgRows, imgCols, imgChannels)

    for i in range(numOfData) :
        image = Image.open("./trainData/{}.png".format(i)).convert("L")
        arr = np.asarray(image)
        arr.reshape(dataShape)
        dataSet.append(arr)

    return np.array(dataSet)

class GAN :
    def __init__(self) :
        self.imgRows = imgRows
        self.imgCols = imgCols
        self.imgChannels = imgChannels
        self.imgShape = (self.imgRows, self.imgCols, self.imgChannels)
        self.noiseSize = noiseSize
        self.optimizer = Adam(0.0002, 0.5)

        self.discriminator = self._createDiscriminator()
        self.generator = self._createGenerator()


        inputLayer = Input(shape=(self.noiseSize,))
        fakeImage = self.generator(inputLayer)
        
        self.discriminator.trainable = False
        validity = self.discriminator(fakeImage)

        self.combined = Model(inputLayer, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)


    def _createGenerator(self) :
        model = Sequential()

        model.add(Dense(256, input_dim=self.noiseSize))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.imgShape), activation='tanh'))
        model.add(Reshape(self.imgShape))
        model.summary()

        noise = Input(shape=(self.noiseSize,))
        image = model(noise)

        return Model(noise, image)

    def _createDiscriminator(self) :
        model = Sequential()

        model.add(Flatten(input_shape=self.imgShape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        image = Input(shape=self.imgShape)
        validity = model(image)

        discriminatorModel = Model(image, validity)
        discriminatorModel.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        return discriminatorModel

    def trainModel(self, dataSet, epochs=10000, batch_size=5, sample_interval=100) :
        dataSet = dataSet/255
        dataSet = np.expand_dims(dataSet, axis=3)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs) :
            
            # Generate real image
            idx = np.random.randint(0, dataSet.shape[0], batch_size)
            imgs = dataSet[idx]
            

            # Generate fake image
            noise = np.random.rand(batch_size, self.noiseSize)
            gen_imgs = self.generator.predict(noise)

            r_loss = self.discriminator.train_on_batch(imgs, valid)
            f_loss = self.discriminator.train_on_batch(gen_imgs, fake)


            noise = np.random.rand(batch_size, self.noiseSize)
            g_loss = self.combined.train_on_batch(noise, valid)

            print("Epochs : {} [ r_loss : {} / f_loss : {} ] [ g_loss : {}]".format(epoch, r_loss, f_loss, g_loss))


            if epoch % sample_interval == 0 :
                self._printSampleImages(epoch)


    def _printSampleImages(self, epoch) :
        r, c = 3, 3
        noise = np.random.rand(r*c, self.noiseSize)
        gen_imgs = self.generator.predict(noise)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r) :
            for j in range(c) :
                axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("result/{}.png".format(epoch))
        plt.close()

if __name__ == "__main__":
    dataSet = loadImage()
    GAN().trainModel(dataSet)
