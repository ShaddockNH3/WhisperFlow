import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import argparse

# Add the project root to sys.path so 'ai.denoise...' imports work 
# even if the script is run from the 'ai' or 'denoise' directories.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai.denoise.dataset import THCHS30DenoiseDataset
from ai.denoise.model import DenoiseCRNN

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Resolve the dataset dir path to be relative to project root if it's not absolute
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not os.path.isabs(args.thchs30_dir):
        # Trying relative to current working dir first (in case they passed `data_thchs30` while in `ai/`)
        thchs30_path = os.path.abspath(args.thchs30_dir)
        if not os.path.exists(thchs30_path):
            # Fall back to resolving relative to project root
            thchs30_path = os.path.join(project_root, args.thchs30_dir)
    else:
        thchs30_path = args.thchs30_dir
    
    # Initialize THCHS-30 dataset
    if not os.path.exists(thchs30_path):
        print(f"ERROR: Dataset dir '{thchs30_path}' not found. Please check your path.")
        return
        
    train_dataset = THCHS30DenoiseDataset(thchs30_path, split='train', noise_dir=args.noise_dir)
    print(f"Found {len(train_dataset)} training samples.")
    if len(train_dataset) == 0:
         return
         
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    model = DenoiseCRNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Pre-allocate window tensor on GPU for STFT
    n_fft = 512
    hop_length = 256
    win_length = 512
    window = torch.hann_window(win_length).to(device)
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (clean_audio, noise_audio) in enumerate(train_loader):
            # 1. Move raw audio to GPU immediately
            clean_audio = clean_audio.to(device)
            noise_audio = noise_audio.to(device)
            batch_size = clean_audio.size(0)
            seq_len = clean_audio.size(1)
            
            # 2. Vectorized SNR Mixing on GPU
            # Random SNR between -5dB and 15dB for each item in batch
            snr_db = torch.empty(batch_size, 1, device=device).uniform_(-5, 15)
            
            clean_power = clean_audio.norm(p=2, dim=1, keepdim=True)**2 / seq_len
            noise_power = noise_audio.norm(p=2, dim=1, keepdim=True)**2 / seq_len
            
            # Prevent division by zero
            noise_power = torch.clamp(noise_power, min=1e-10)
            
            target_noise_power = clean_power / (10 ** (snr_db / 10))
            noise_audio = noise_audio * torch.sqrt(target_noise_power / noise_power)
            
            mixed_audio = clean_audio + noise_audio
            
            # Normalize each batch item
            max_val = mixed_audio.abs().max(dim=1, keepdim=True).values
            # avoid div by zero
            max_val = torch.clamp(max_val, min=1e-8)
            mixed_audio = mixed_audio / max_val
            clean_audio = clean_audio / max_val
            noise_audio = noise_audio / max_val
            
            # 3. Batched STFT on GPU (drastically faster)
            mixed_stft = torch.stft(mixed_audio, n_fft=n_fft, hop_length=hop_length, 
                                    win_length=win_length, window=window, return_complex=True)
            clean_stft = torch.stft(clean_audio, n_fft=n_fft, hop_length=hop_length, 
                                    win_length=win_length, window=window, return_complex=True)
            noise_stft = torch.stft(noise_audio, n_fft=n_fft, hop_length=hop_length, 
                                    win_length=win_length, window=window, return_complex=True)
                                    
            mixed_mag = torch.abs(mixed_stft)
            clean_mag = torch.abs(clean_stft)
            noise_mag = torch.abs(noise_stft)
            
            # Ideal Ratio Mask (IRM)
            target_mask = (clean_mag ** 2) / (clean_mag ** 2 + noise_mag ** 2 + 1e-8)
            
            # Features: Log Magnitude spectrogram
            features = torch.log1p(mixed_mag)
            
            # Transpose to match model input [Batch, Time, Freq]
            features = features.transpose(1, 2)
            target_mask = target_mask.transpose(1, 2)
            
            optimizer.zero_grad()
            
            predicted_mask = model(features)
            loss = criterion(predicted_mask, target_mask)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_loss:.4f}")
        
    # Save the trained model weights
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'denoise_crnn.pt')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the WhisperFlow Denoising CRNN Model on THCHS-30")
    parser.add_argument('--thchs30_dir', type=str, default='ai/data_thchs30', help='Path to THCHS-30 root directory')
    parser.add_argument('--noise_dir', type=str, default=None, help='Path to directory containing background noise wavs (Optional, will synthesize if none)')
    parser.add_argument('--save_dir', type=str, default='ai/denoise/weights', help='Directory to save model weights')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    
    args = parser.parse_args()
    train(args)
