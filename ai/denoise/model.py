import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoiseCRNN(nn.Module):
    """
    A lightweight Convolutional Recurrent Neural Network (CRNN) 
    for predicting the Ideal Ratio Mask (IRM) of a noisy spectrogram.
    """
    def __init__(self, input_dim=257, hidden_dim=256, num_layers=2):
        super().__init__()
        
        # Simple 1D convolutions across the frequency domain
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # GRU for temporal context (crucial for streaming sequence data)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, 
                          num_layers=num_layers, batch_first=True, bidirectional=False)
                          
        # Output layer maps back to frequency bins for the mask
        self.fc = nn.Linear(hidden_dim, input_dim)
        
        # Sigmoid to ensure the mask is between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [Batch, Time, Freq]
        
        # PyTorch Conv1d expects [Batch, Channels, Length]
        # Here we treat Freq as Channels and Time as Length
        x = x.transpose(1, 2) # [Batch, Freq, Time]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Prepare for GRU: [Batch, Time, Channels]
        x = x.transpose(1, 2)
        
        # pass through GRU
        gru_out, _ = self.gru(x)
        
        # predict mask
        mask_logits = self.fc(gru_out)
        mask = self.sigmoid(mask_logits)
        
        return mask
