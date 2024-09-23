"""
 Copyright (C) 2023 Università degli Studi di Camerino.
 Authors: Alessandro Antinori, Rosario Capparuccia, Riccardo Coltrinari, Flavio Corradini, Marco Piangerelli, Barbara Re, Marco Scarpetta, Luca Mozzoni, Vincenzo Nucci

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as
 published by the Free Software Foundation, either version 3 of the
 License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
 """

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from rain.core.base import Tags, LibTag, TypeTag, ComputationalNode
from rain.core.parameter import Parameters, KeyValueParameter

def prepare_data(features_df, labels_df, batch_size=32, shuffle=True):
    X = features_df.values
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    y = labels_df.values
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return data_loader

def add_linear():
    return nn.Linear(in_features=128, out_features=64)

def add_conv2d():
    return nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

def add_lstm():
    return nn.LSTM(input_size=64, hidden_size=128, num_layers=2)

LAYER_FUNCTIONS = {
    "linear": add_linear,
    "conv2d": add_conv2d,
    "lstm": add_lstm,
}

def build_model(layer_types):
    layers = []
    for layer_type in layer_types:
        if layer_type in LAYER_FUNCTIONS:
            layer_fn = LAYER_FUNCTIONS[layer_type]
            layers.append(layer_fn())
    return nn.Sequential(*layers)

def train_model(model, data_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

def evaluate_model(model, test_loader, criterion="mse_loss"):
    model.eval()
    total_loss = 0.0
    if criterion == "ce_loss":
        criterion = nn.CrossEntropyLoss()
    else: criterion = nn.MSELoss()

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

class SequentialNN(ComputationalNode):
    """Builds, trains and evaluates a PyTorch Sequential model on the provided features. 
    The average loss is returned. 

    Input
    -----
    train_features : pandas.DataFrame
        The dataset train features.
    train_labels : pandas.DataFrame
        The dataset train labels.
    test_features : pandas.DataFrame
        The dataset test features.
    test_labels : pandas.DataFrame
        The dataset test labels.

    Output
    ------
    loss : float
        The trained Sequential model.

    Parameters
    ----------
    layers : [conv2d, linear, lstm]
        The layers to include in the model.
    num_epochs : int, default=5
        The number of epochs.
    learning_rate : float, default=0.001
        The learning rate.

    Notes
    -----
    Visit `<https://pandas.pydata.org/pandas-docs/version/1.3/reference/api/pandas.read_csv.html>`_ for Pandas read_csv
    documentation.
    """

    def __init__(
            self, 
            node_id: str,
            layers: list,
            num_epochs: int = 5,
            learning_rate: float = 0.001,
        ):
        super(SequentialNN, self).__init__(node_id)
        self.parameters = Parameters(
            layers=KeyValueParameter("layers", list, layers),
            num_epochs=KeyValueParameter("num_epochs", list, num_epochs),
            learning_rate=KeyValueParameter("learning_rate", list, learning_rate),
        )

    _input_vars = {"train_features": pd.DataFrame, "train_labels": pd.DataFrame, "test_features": pd.DataFrame, "test_labels": pd.DataFrame}
    _output_vars = {"loss": float}

    def execute(self):
        layers = self.parameters.layers.value
        num_epochs = self.parameters.num_epochs.value
        learning_rate = self.parameters.learning_rate.value
        
        train_features = self.train_features.values
        train_labels = self.train_labels.values
        data_loader = prepare_data(train_features, train_labels)
        
        test_features = self.test_features.values
        test_labels = self.test_labels.values
        test_loader = prepare_data(test_features, test_labels, shuffle=False)
        
        model = build_model(layers)
        train_model(model, data_loader, num_epochs=num_epochs, learning_rate=learning_rate)
        
        self.loss = evaluate_model(model, test_loader)

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.TORCH, TypeTag.NN)