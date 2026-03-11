import torch
import torchaudio
import os
import random
import soundfile as sf
import math
from torch.utils.data import Dataset

class THCHS30DenoiseDataset(Dataset):
    """
    Dataset loader for THCHS-30 clean speech corpus.
    Dynamically generates background noise (white/pink) or loads noise files
    and mixes them with the clean speech to create Noisy-Clean training pairs.
    """
    def __init__(self, thchs30_dir: str, split: str = 'train', noise_dir: str = None, seq_len: int = 48000, target_sr: int = 16000):
        super().__init__()
        
        # The true audio files are in `data`, but `train`, `dev`, `test` contain symlinks.
        # So we read the symlinks in the split directory.
        split_dir = os.path.join(thchs30_dir, split)
        
        # Windows doesn't always play nice with ext4 symlinks via WSL depending on the mount,
        # so we will look at all files ending in .wav in the split_dir
        self.clean_files = []
        if os.path.exists(split_dir):
            for file in os.listdir(split_dir):
                if file.endswith('.wav'):
                    # THCHS-30 symlinks in `train` point to `../data/xxx.wav`
                    # Instead of relying on OS symlinks which might be broken across Win/WSL, 
                    # let's just directly construct the true path in `data`.
                    true_path = os.path.abspath(os.path.join(thchs30_dir, 'data', file))
                    if os.path.exists(true_path):
                        self.clean_files.append(true_path)
                    else:
                        # Fallback just in case it's a real file or a resolvable link
                        path = os.path.join(split_dir, file)
                        if os.path.islink(path):
                            path = os.path.realpath(path)
                        self.clean_files.append(path)
        else:
            print(f"Warning: split directory {split_dir} not found.")
            
        # Optional external noise dir
        self.noise_files = []
        if noise_dir and os.path.exists(noise_dir):
            self.noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.wav')]
            
        self.seq_len = seq_len
        self.target_sr = target_sr
        
    def __len__(self):
        return len(self.clean_files)
        
    def _load_and_pad(self, path: str) -> torch.Tensor:
        try:
            # Bypass torchaudio.load completely to avoid torchcodec issues
            data, sr = sf.read(path, dtype='float32')
            waveform = torch.from_numpy(data)
            
            # soundfile returns [frames] for mono, [frames, channels] for multi
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0) # [1, frames]
            else:
                waveform = waveform.transpose(0, 1) # [channels, frames]
                
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = resampler(waveform)
            # Convert to mono
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)
            
            # Trim or pad to seq_len
            if waveform.size(0) > self.seq_len:
                start = random.randint(0, waveform.size(0) - self.seq_len)
                waveform = waveform[start:start + self.seq_len]
            elif waveform.size(0) < self.seq_len:
                pad_len = self.seq_len - waveform.size(0)
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))
                
            return waveform
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(self.seq_len)

    def _get_noise(self) -> torch.Tensor:
        """Loads a real noise file or generates synthetic noise."""
        if len(self.noise_files) > 0 and random.random() > 0.3:
             # 70% chance to use real noise if available
             noise_idx = random.randint(0, len(self.noise_files) - 1)
             return self._load_and_pad(self.noise_files[noise_idx])
        else:
             # 30% chance (or 100% if no noise dir provided) to generate synthetic Pink/White noise
             noise_type = random.choice(['white', 'pink'])
             if noise_type == 'white':
                 return torch.randn(self.seq_len)
             else:
                 # Vectorized Pink Noise generation via Frequency Domain (1/f)
                 # This is orders of magnitude faster than a Python for-loop
                 white = torch.randn(self.seq_len)
                 X_white = torch.fft.rfft(white)
                 
                 # Create 1/f filter (skip DC component to avoid inf)
                 freqs = torch.fft.rfftfreq(self.seq_len)
                 freqs[0] = 1.0 # Avoid division by zero
                 filter_1f = 1.0 / torch.sqrt(freqs)
                 
                 X_pink = X_white * filter_1f
                 pink = torch.fft.irfft(X_pink, n=self.seq_len)
                 
                 # Normalize
                 pink = pink / (torch.max(torch.abs(pink)) + 1e-8)
                 return pink

    def __getitem__(self, idx):
        # 1. Load clean audio
        clean_audio = self._load_and_pad(self.clean_files[idx])
        
        # 2. Get noise audio
        noise_audio = self._get_noise()
        
        # We now simply return the raw audio tensors.
        # Mixing, SNR adjustment, and STFT computation will happen in the training loop
        # directly on the GPU to drastically speed up training!
        return clean_audio, noise_audio
