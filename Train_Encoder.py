import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.Thoughtviz_Encoder_CNN import Encoder_CNN

class EEG_Encoder():

    def __init__(self, num_classes, dataset):
        self.num_classes = num_classes
        self.dataset = dataset
        self.eeg_pkl_file = os.path.join('./data/eeg/', self.dataset, 'data.pkl')

    # def initialize_weights(model):
    #     for m in model.modules():
    #         if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
    #             nn.init.normal_(m.weight.data, 0.0, 0.02)

    def train(self, 
              model_save_dir, 
              run_id, 
              batch_size,
              num_epochs,
              lr):
        
        device = torch.device("cpu")
        if torch.cuda.is_available() :
            device = torch.device("cuda")
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            print("Using MPS...")
            device = torch.device("mps") #apple silicon

        with open(self.eeg_pkl_file, 'rb') as file:
            data = pickle.load(file, encoding='latin1')
            x_train = data['x_train']
            y_train = data['y_train']
            x_test = data['x_test']
            y_test = data['y_test']

        # data = pickle.load(open(self.eeg_pkl_file, 'rb'), encoding='bytes')

        # x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']

        # transform = transforms.Compose(
        # [
        #     transforms.ToTensor(),
        # ]
        # )

        x_train_tensor = torch.Tensor(x_train)
        y_train_tensor = torch.Tensor(y_train)


        # train_data_transformed = transform(train_data)
        x_train_loader = DataLoader(x_train_tensor, batch_size=batch_size, shuffle=False)
        y_train_loader = DataLoader(y_train_tensor, batch_size=batch_size, shuffle=False)

        eeg_encoder = Encoder_CNN(x_train.shape[1], self.num_classes).to(device)
        # self.initialize_weights(eeg_encoder)

        loss_function = nn.MSELoss()
        opt = optim.SGD(eeg_encoder.parameters(), lr=lr, weight_decay=1e-6, momentum=0.9, nesterov=True)

        eeg_encoder.train()

        print(f"Start training for model id {run_id}")
        for epoch in range(num_epochs):

            eeg_encoder.train()

            for batch_idx, (x, y) in enumerate(zip(x_train_loader, y_train_loader)):

                inputs, labels = x.to(device), y.to(device)

                opt.zero_grad()
                outputs = eeg_encoder(inputs)

                loss = loss_function(outputs, labels)

                loss.backward()
                opt.step()

                if batch_idx % 100 == 0:
                    print(
                        f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(x_train_loader)}] \ "
                        f"Loss : {loss:.4f}"
                    )

            eeg_encoder.eval()

            x_test_tensor = torch.Tensor(x_test)
            y_test_tensor = torch.Tensor(y_test)

            x_test_loader = DataLoader(x_test_tensor, batch_size=batch_size, shuffle=False)
            y_test_loader = DataLoader(y_test_tensor, batch_size=batch_size, shuffle=False)

            correct_label = 0
            total_label = 0
            for batch_idx, (x, y) in enumerate(zip(x_test_loader, y_test_loader)):
                inputs, labels = x.to(device), y.to(device)

                outputs = eeg_encoder(inputs)

                probability = func.softmax(outputs, dim=1)

                for i in range(len(probability)):
                    pred = torch.argmax(probability[i])
                    target = torch.argmax(labels[i])
                    if (pred == target):
                        correct_label += 1
                    total_label += 1

            print(
                f"Accuracy : {correct_label/total_label} % "
            )

if __name__ == '__main__':
    batch_size, num_epochs = 128, 200

    char_encoder = EEG_Encoder(10, 'char')
    char_encoder.train('', 1, batch_size, num_epochs, 0.001)