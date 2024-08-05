import os
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import DataLoader
import utils.data_input_util as inutil
from models.Thoughtviz_Decoder import Generator, Discriminator
from models.Thoughtviz_Encoder_CNN import Encoder_CNN

class EEG_Decoder():
    
    def __init__(self, num_classes, image_dir, eeg_dir):
        self.num_classes = num_classes
        self.image_dir = image_dir
        self.eeg_dir = eeg_dir

    def train(self, run_id, noise_dim, img_dim, batch_size, num_epochs):
        device = torch.device("cpu")
        # if torch.cuda.is_available() :
        #     device = torch.device("cuda")
        # elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        #     print("Using MPS...")
        #     device = torch.device("mps") 

        # char_fonts_folders = ["./data/images/Char-Font"]

        eeg_feature_dim = 100

        adam_lr = 0.00005
        adam_beta_1 = 0.5

        x_train, y_train, x_test, y_test = inutil.load_char_data(self.image_dir, resize_shape=(28, 28), num_classes=self.num_classes)
        print("Loaded Characters Dataset.")

        # cuma tes doang
        x_train = x_train[:5642]
        y_train = y_train[:5642]

        x_train_tensor = torch.Tensor(x_train)
        y_train_tensor = torch.Tensor(y_train)

        # train_data_transformed = transform(train_data)
        x_train_loader = DataLoader(x_train_tensor, batch_size=batch_size, shuffle=False)
        y_train_loader = DataLoader(y_train_tensor, batch_size=batch_size, shuffle=False)

        gen = Generator(noise_dim)
        gen_optim = optim.Adam(gen.parameters(), lr=adam_lr, betas=(adam_beta_1, 0.999))

        disc = Discriminator(img_dim)
        disc_optim = optim.Adam(disc.parameters(), lr=adam_lr, betas=(adam_beta_1, 0.999))

        criterion = nn.BCELoss()

        # prepare eeg data
        with open(os.path.join(self.eeg_dir, "data.pkl"), 'rb') as file:
            eeg_data = pickle.load(file, encoding='latin1')
            eeg_features = eeg_data['x_test']
            eeg_labels = eeg_data['y_test']
            # eeg_features_test = eeg_data['x_test']
            # eeg_labels_test = eeg_data['y_test']

        eeg_labels = np.array([np.argmax(y) for y in eeg_labels])
        eeg_labels = eeg_labels.reshape(-1, 1)
        # eeg_labels = [inutil.to_categorical(eeg_labels, 10) for eeg_label in eeg_labels]

        eeg_features_tensor = torch.Tensor(eeg_features).to(device)
        eeg_labels_tensor = torch.Tensor(eeg_labels).to(device)

        encoder = Encoder_CNN(eeg_features_tensor.shape[1], 10).to(device)

        _, eeg_features = encoder(eeg_features_tensor)

        eeg_features_train = DataLoader(eeg_features, batch_size=batch_size, shuffle=False)

        gen.train()
        disc.train()

        print(f"Start training for model id {run_id}")
        for epoch in range(num_epochs):

            for batch_idx, (x, y, eeg) in enumerate(zip(x_train_loader, y_train_loader, eeg_features_train)):

                # inputs, labels = x.to(device), y.to(device)

                # gen_optim.zero_grad()
                # outputs, _ = gen(inputs)

                # loss = loss_function(outputs, labels)

                # loss.backward()
                # gen_optim.step()

                noise = torch.randn(eeg.shape[0], noise_dim).to(device)

                real_imgs, real_labels, encoded_eeg = x.to(device), y.to(device), eeg.to(device)

                # Train Discriminator
                generated_imgs = gen.forward(noise, encoded_eeg)
                disc_real = disc.forward(real_imgs).reshape(-1)
                lossD_real = criterion(disc_real, torch.ones_like(disc_real))
                disc_fake = disc.forward(generated_imgs).reshape(-1)
                lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
                lossD = (lossD_real + lossD_fake) / 2
                disc.zero_grad()
                lossD.backward(retain_graph=True)
                disc_optim.step()

                # Train Generator
                output = disc(generated_imgs).reshape(-1)
                lossG = criterion(output, torch.ones_like(output))
                gen.zero_grad()
                lossG.backward(retain_graph=True)
                gen_optim.step()

                if batch_idx % 100 == 0:
                    print(
                        f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(x_train_loader)}] \ "
                        f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
                    )

eeg_decoder = EEG_Decoder(10, ["./data/images/Char-Font"], "./data/eeg/char/")
eeg_decoder.train(run_id=1, noise_dim=100, img_dim=1, batch_size=32, num_epochs=1)
# run_id, noise_dim, img_dim, batch_size, num_epochs