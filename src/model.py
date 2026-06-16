import torch as tr
import math
from torch import nn
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, average_precision_score
from tqdm import tqdm

class BaseModel(nn.Module):
    def __init__(self, nclasses, emb_size=352, lr=1e-3, device="cuda", 
    filters=550, kernel_size=9, num_layers=3, first_dilated_layer=2, 
    dilation_rate=3, resnet_bottleneck_factor=.5, dropout_rate=0.4):
        """
        A CNN with residual layers for sequence classification.

        Args:
            nclasses: Number of output classes.
            emb_size: Size of the input embeddings.
            lr: Learning rate for the optimizer.
            device: Device to run the model on ('cuda' or 'cpu').
        All the other parameters define the architecture of the CNN.
        """
        super().__init__()
        
        self.emb_size = emb_size 
        self.train_steps = 0
        self.dev_steps = 0

        self.cnn = [nn.Conv1d(self.emb_size, filters, kernel_size, padding="same")]
        for k in range(num_layers):
            self.cnn.append(ResidualLayer(k, first_dilated_layer, dilation_rate, 
                                          resnet_bottleneck_factor, filters, kernel_size))
        self.cnn.append(nn.AdaptiveMaxPool1d(1))
        self.cnn = nn.Sequential(*self.cnn)

        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(filters, nclasses)

        self.loss = nn.CrossEntropyLoss()
        self.optim = tr.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)

        self.scheduler = tr.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, 
            mode='max', 
            factor=0.5,      # Reduce LR por 0.5
            patience=5       # Después de 5 epochs sin mejora
        )

        self.to(device)
        self.device = device

    def forward(self, emb):
        """emb is the embedded sequence batch with shape [N, EMBSIZE, L]"""
        y = self.cnn(emb.to(self.device))
        y = self.dropout(y) 
        y = self.fc(y.squeeze(2))
        return y

    def fit(self, dataloader):
        """Train the model for one epoch on the provided dataloader"""
        avg_loss = 0

        self.cnn.train(), self.fc.train()
        self.optim.zero_grad() 

        for k,(x, y, *_) in enumerate(tqdm(dataloader)):
            yhat = self(x)
            y = y.to(self.device) # We obtain soft label

            loss = self.loss(yhat, y)
            loss.backward()
            avg_loss += loss.item()
            self.optim.step()
            self.optim.zero_grad()

            self.train_steps+=1

        avg_loss /= len(dataloader)
        
        # Schedule step is called per epoch with average validation loss
        # This will be called from train_test_base.py after pred()

        return avg_loss

    def pred(self, dataloader):
        """Evaluate the model on the provided dataloader"""
        
        # Hard labels are categorical labels represented as integers, while
        # soft labels are probability distributions over classes.
        pred, ref_soft, ref_hard, names = [], [], [], []
        centers, starts, ends, = [], [], []

        test_loss = 0
        self.eval()

        for seq, y_soft, y_hard, center, name, start, end in tqdm(dataloader):
            with tr.no_grad():
                yhat = self(seq)

                # Soft labels are used for loss calculation
                y_soft = y_soft.to(self.device)
                test_loss += self.loss(yhat, y_soft).item()

            names += name
            centers += center.tolist()
            starts += start.tolist()
            ends += end.tolist()

            pred.append(yhat.detach().cpu())
            ref_soft.append(y_soft.cpu())
            ref_hard.append(y_hard.cpu())

        pred = tr.cat(pred)
        probs = tr.softmax(pred, dim=1) # Added softmax to get probabilities for AUC and APS
        ref_soft = tr.cat(ref_soft)
        ref_hard = tr.cat(ref_hard)

        pred_bin = tr.argmax(pred, dim=1)

        self.dev_steps += 1
        test_loss /= len(dataloader)

        # Hard labels are used for metrics
        acc = accuracy_score(ref_hard, pred_bin) 
        f1 = f1_score(ref_hard, pred_bin, zero_division=0)
        auc = roc_auc_score(ref_hard, probs[:, 1])
        aps = average_precision_score(ref_hard, probs[:, 1])

        return (test_loss, 1-acc, auc, f1, aps, pred, ref_soft, ref_hard, 
                names, centers)
    
    def num_parameters(self):
        """Return the number of trainable parameters in the model"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable

class ResidualLayer(nn.Module):
    def __init__(self, layer_index, first_dilated_layer, dilation_rate, 
                 resnet_bottleneck_factor, filters, kernel_size):
        super().__init__()

        shifted_layer_index = layer_index - first_dilated_layer + 1
        dilation_rate = max(1, dilation_rate**shifted_layer_index)

        num_bottleneck_units = math.floor(
            resnet_bottleneck_factor * filters)

        self.layer = nn.Sequential(nn.BatchNorm1d(filters),
        nn.ReLU(),
        nn.Conv1d(filters, num_bottleneck_units, kernel_size, 
                  dilation=dilation_rate, padding="same"), 
        nn.BatchNorm1d(num_bottleneck_units),
        nn.ReLU(),
        nn.Conv1d(num_bottleneck_units, filters, kernel_size=1, padding="same"))
        # The second convolution is purely local linear transformation across
        # feature channels, as is done in
        # tensorflow_models/slim/nets/resnet_v2.bottleneck

    def forward(self, x):
        return x + self.layer(x)