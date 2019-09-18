import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy 

class Trainer(object):
    def __init__(self, model, dataset_train,
                 dataset_test, model_path,
                 learning_rate=0.01, batch_size=100, epochs=400):

        self.model = model
        self.criterion = nn.CrossEntropyLoss(reduction = 'none')
        self.optimizer = Adam(model.parameters(), lr=learning_rate)

        self.loader_train = DataLoader(dataset_train, batch_size=batch_size,
                                       shuffle=True)
        self.loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
        self.model_path = model_path
        self.epochs = epochs

    def run_epoch(self, dataloader, do_optimize = True):
        epoch_loss = 0
        
        if do_optimize:
            self.model.train()
        else:
            self.model.eval()

        for idx, data in enumerate(dataloader):
            X, Y, mask = data
            X, Y, mask = X.cuda(), Y.cuda(), mask.cuda()

            Y = Y.view(Y.shape[0] * Y.shape[1])
            mask = mask.view(mask.shape[0] * mask.shape[1])

            loss = (self.criterion(self.model(X), Y) * mask).mean()
            if do_optimize:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss *= idx
            epoch_loss += loss.item()
            epoch_loss /= (idx + 1)
        return epoch_loss
    
    def train(self):
        best_test_loss = np.inf
        for _ in range(self.epochs):
            self.run_epoch(self.loader_train)
            with torch.no_grad():
                loss = self.run_epoch(self.loader_test, do_optimize=False)
                print(loss)
            if loss < best_test_loss:
                best_test_loss = loss
                torch.save({ 'state_dict': self.model.state_dict() }, self.model_path)