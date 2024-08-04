import os
import pickle
import random

import torch
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import DataLoader
import utils.data_input_util as inutil
from models.Thoughtviz_Decoder import *

class EEG_Decoder():
    
    def __init__(self, num_classes, image_dir, eeg_dir):
        self.num_classes = num_classes
        self.image_dir = image_dir
        self.eeg_dir = eeg_dir

    def train(self, run_id, noise_dim, batch_size, num_epochs):
        device = torch.device("cpu")
        if torch.cuda.is_available() :
            device = torch.device("cuda")
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            print("Using MPS...")
            device = torch.device("mps") #apple silicon

        # char_fonts_folders = ["./data/images/Char-Font"]

        feature_encoding_dim = 100

        adam_lr = 0.00005
        adam_beta_1 = 0.5

        x_train, y_train, x_test, y_test = inutil.load_char_data(self.image_dir, resize_shape=(28, 28), num_classes=self.num_classes)
        print("Loaded Characters Dataset.")

        x_train_tensor = torch.Tensor(x_train)
        y_train_tensor = torch.Tensor(y_train)

        # train_data_transformed = transform(train_data)
        x_train_loader = DataLoader(x_train_tensor, batch_size=batch_size, shuffle=False)
        y_train_loader = DataLoader(y_train_tensor, batch_size=batch_size, shuffle=False)

        gen = Generator(noise_dim)
        loss_function = nn.CrossEntropyLoss()
        gen_optim = optim.Adam(gen.parameters(), lr=adam_lr, betas=adam_beta_1)

        gen.train()

        print(f"Start training for model id {run_id}")
        for epoch in range(num_epochs):

            gen.train()

            for batch_idx, (x, y) in enumerate(zip(x_train_loader, y_train_loader)):

                inputs, labels = x.to(device), y.to(device)

                gen_optim.zero_grad()
                outputs, _ = gen(inputs)

                loss = loss_function(outputs, labels)

                loss.backward()
                gen_optim.step()

                if batch_idx % 100 == 0:
                    print(
                        f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(x_train_loader)}] \ "
                        f"Loss : {loss:.4f}"
                    )
