import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Generator model
def build_generator(latent_dim):
    model = models.Sequential([
        layers.Dense(128, input_dim=latent_dim, activation='relu'),
        layers.Dense(784, activation='sigmoid'),
        layers.Reshape((28, 28))
    ])
    return model

# Discriminator model
def build_discriminator(input_shape):
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Combined GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan = models.Sequential([generator, discriminator])
    return gan

# Training function
def train_gan(generator, discriminator, gan, epochs, batch_size, latent_dim, real_data):
    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_data = generator.predict(noise)

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gan_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, gan_labels)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, D Loss Real: {d_loss_real}, D Loss Fake: {d_loss_fake}, G Loss: {g_loss}")

# Main function
def main():
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.astype('float32') / 255.0

    latent_dim = 100
    generator = build_generator(latent_dim)
    discriminator = build_discriminator(train_images[0].shape)
    gan = build_gan(generator, discriminator)

    gan.compile(loss='binary_crossentropy', optimizer='adam')

    train_gan(generator, discriminator, gan, epochs=10000, batch_size=32, latent_dim=latent_dim, real_data=train_images)

if __name__ == "__main__":
    main()
