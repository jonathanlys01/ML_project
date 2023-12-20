import torch
import torch.nn as nn 
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin

from tqdm import tqdm

import numpy as np

import time

from sklearn.model_selection import GridSearchCV


class BaseMLP(torch.nn.Module):
    """
    Base Module class for MLPs
    
    Parameters

    input_size: int, size of the input layer
    output_size: int, size of the output layer
    architecture: list of int, sizes of the hidden layers
    p: float, dropout probability
    """
    def __init__(self, input_size, output_size, architecture, p, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if len(architecture)==0:
            self.layer = nn.Linear(input_size, output_size)
            self.hidden_layers = nn.Sequential([])

        else: 
            self.layer = nn.Linear(input_size, architecture[0])

            self.hidden_layers = nn.ModuleList([
                nn.Linear(architecture[i], architecture[i+1]) for i in range(len(architecture)-1)
            ] + [nn.Linear(architecture[-1], output_size)]
            )

        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(architecture[i])for i in range(len(architecture)-1)])

        self.dropout = nn.Dropout(p)

        self.architecture = architecture

    def forward(self, x):
        x = self.layer(x)
        
        for i in range(len(self.architecture)-1):
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.hidden_layers[i](x)
        if len(self.hidden_layers):
            x = self.hidden_layers[-1](x)

        return F.sigmoid(x)
        

    

class TorchMLP(BaseEstimator, ClassifierMixin):
    """
    MLP implemented with PyTorch, to be used with sklearn pipelines

    Parameters

    n_epochs: int, number of epochs
    bs: int, batch size
    opt: str, one of "adam", "sgd"
    lr: float, learning rate
    architecture: list of int, sizes of the hidden layers
    p: float, dropout probability
    """

    def __init__(self, n_epochs = 200, bs = 64, opt = "adam", lr = 1e-3, p = 0.05, architecture = [16, 32, 16], verbose = True):
        self.model = None
        
        
        self.n_epochs = n_epochs
        self.bs = bs
        self.opt = opt
        self.lr = lr
        self.architecture = architecture
        self.p = p
        self.verbose = verbose
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        self.fitted = False
        
        self.list_loss = []

    def fit(self, X, y,):
        # X (n_samples, dim)
        # y (n_samples, 1)


        input_size = X.shape[1]
        if len(y.shape)==1:
            y = y.reshape(-1,1)
        output_size = y.shape[1]

        assert X.shape[0] == y.shape[0]

        n_samples = X.shape[0]
    
        self.model = BaseMLP(input_size=input_size, output_size=output_size, architecture=self.architecture, p=self.p)
        self.model = self.model.to(self.device)

        criterion = torch.nn.BCEWithLogitsLoss()

        if self.opt == "adam":
            opt = torch.optim.Adam(lr=self.lr, params=self.model.parameters())
        elif self.opt == "sgd":
            opt = torch.optim.SGD(lr=self.lr, params=self.model.parameters())
        elif self.opt == "rmsprop":
            opt = torch.optim.RMSprop(lr=self.lr, params=self.model.parameters())
        else:
            raise NotImplementedError(f"{self.opt} not a valid optimizer")
        
        pbar = tqdm(range(self.n_epochs)) if self.verbose else range(self.n_epochs)
        
        for i in pbar:
            batch_indexes = list(range(n_samples))
            np.random.shuffle(batch_indexes)
            batch_indexes = [batch_indexes[i:i+self.bs] for i in range(0, n_samples, self.bs)]

            total_loss = torch.Tensor([0.])
            for indexes in batch_indexes:
                x_tensor = torch.FloatTensor(X[indexes,:])
                labels = torch.FloatTensor(y[indexes,:])

                opt.zero_grad()

                output = self.model(x_tensor)

                loss = criterion(output, labels)

                total_loss += loss.item()

                loss.backward()

                opt.step()
            self.list_loss.append(float(total_loss))
            if self.verbose:pbar.set_description(f"training loss : {float(total_loss):2f}")

        self.fitted = True
        return self
    def predict(self, X):
        assert self.fitted, "model not fitted"
        x_tensor = torch.FloatTensor(X).to(self.device)
        with torch.inference_mode():
            return self.model(x_tensor).detach().cpu().numpy().reshape(-1)
    def score(self, X, y):
        assert self.fitted, "model not fitted"
        return np.mean((np.round(self.predict(X))==y))
    
def train(dataset, model_list, configs, inputs):
    for model in model_list:
        print(model.__class__.__name__)
        start = time.time()
        grid = GridSearchCV(model, configs[model.__class__.__name__], cv=5, verbose=0, n_jobs=-1)
        grid.fit(inputs[dataset]["X_train"], inputs[dataset]["y_train"])
        print(f"Best : {grid.best_params_} with score {grid.best_score_:0.3f} (in {time.time()-start:0.3f}s)")
        print(f"Test score : {grid.score(inputs[dataset]['X_test'], inputs[dataset]['y_test']):0.3f}")
        print()

def main():
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_features=4, n_samples=100, centers = np.array([[-2,-2], [2,2]]))

    config = dict(
    n_epochs = 100,
    bs = 64,
    opt = "adam",
    lr = 1e-3,
    architecture = [16, 32, 16],
    p = 0.2
    )

    myMLP = TorchMLP(**config)

    myMLP.fit(X, y)

    print(myMLP.score(X, y))

if __name__ == "__main__":
    main()