# Learning LSTM

## Theory

<p align="center">
  <img src="static/g1.png" />
</p>

## Code In PyTorch

**input**: tensor of shape $(L, H_{in})$ for unbatched input, $(L, N, H_{in})$ when `batch_first=False` or $(N, L, H_{in})$ 
when `batch_first=True` containing the features of the input sequence. 
The input can also be a packed variable length sequence.

`h_0`: tensor of shape $(D * num.layers, H_{out})$ for unbatched input or $(D * num.layers, N, H_{out})$ 
containing the initial hidden state for each element in the input sequence. 
Defaults to zeros if `(h_0, c_0)` is not provided.

`c_0`: tensor of shape $(D * num.layers, H_{cell})$  for unbatched input or $(D * num.layers, N, H_{cell})$ 
containing the initial cell state for each element in the input sequence. 
Defaults to zeros if `(h_0, c_0)` is not provided.

Where:

N = batch size

L = sequence length

D = 2 if `bidirectional=True` otherwise 1

$H_{in}$ = `input_size`

$H_{cell}$ = `hidden_size`

$H_{out}$ = `proj_size` if `proj_size > 0` otherwise `hidden_size`


Class example:

```python
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # lstm
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out
```

## Credits

- [blog | lstm](https://cnvrg.io/pytorch-lstm/)
- [pytorch | lstm](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)