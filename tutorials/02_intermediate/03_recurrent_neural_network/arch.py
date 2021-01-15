from constants import *


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.name = 'RNN'
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))


        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


"""
https://towardsdatascience.com/understanding-rnn-and-lstm-f7cdf6dfc14e
https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
"""