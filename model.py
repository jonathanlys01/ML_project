import torch
import torch.nn as nn 
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin

from tqdm import tqdm

import numpy as np

from sklearn.datasets import make_blobs


class BaseMLP(torch.nn.Module):
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

    def __init__(self, config):
        self.model = None
        """self.device = torch.device("cpu") # no accelerator
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")"""
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        assert all(config.get(key) is not None for key in ["n_epochs", "bs", "opt","lr","architecture","p"])

        self.n_epochs = config.get("n_epochs")
        self.lr = config.get("lr")
        self.opt = config.get("opt")
        self.bs = config.get("bs")
        self.architecture = config.get("architecture")
        self.p = config.get("p")

        self.fitted = False

        self.config = config
        
        self.list_loss = []

    def fit(self, X, y):
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
            opt = torch.optim.SGD(lr=self.lr)
        else:
            raise NotImplementedError(self.opt, params= self.model.parameters())
        
        pbar = tqdm(range(self.n_epochs))
        
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
            pbar.set_description(f"training loss : {float(total_loss):2f}")

        self.fitted = True
        return self
    def predict(self, X):
        assert self.fitted
        x_tensor = torch.FloatTensor(X).to(self.device)
        with torch.inference_mode():
            return self.model(x_tensor).detach().cpu().numpy()

def main():
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_features=4, n_samples=100, centers = np.array([[-2,-2], [2,2]]))

    config = dict(
    n_epochs = 100,
    bs = 64,
    opt = "adam",
    lr = 1e-3,
    architecture = [16, 32, 16],
    p = 0.05
    )

    myMLP = TorchMLP(config=config)

    myMLP.fit(X, y)

    print(print(100*np.mean((np.round(myMLP.predict(X)))==y.reshape(-1,1))))

if __name__ == "__main__":
    main()