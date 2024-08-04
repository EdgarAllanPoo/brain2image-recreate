import os
import pickle

def train_gan(noise_dim, batch_size, epochs):
    image_folder = "./data/images/Char-Font"
    num_classes = 10

    feature_encoding_dim = 100

    g_adam_lr = 0.00003
    g_adam_beta_1 = 0.5

    