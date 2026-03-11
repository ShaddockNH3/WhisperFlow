import torch
import torchaudio
import os
import sys
import argparse
import soundfile as sf

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai.denoise.model import DenoiseCRNN

import numpy as np

def denoise_array(model, waveform_np: np.ndarray, sr: int = 16000) -> np.ndarray:
    device = next(model.parameters()).device
    waveform = torch.from_numpy(waveform_np)
    
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.transpose(0, 1)
        
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
        sr = 16000
    
    # Convert to mono
    if waveform.size(0) > 1:
         waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    
    # STFT parameters matching training
    n_fft = 512
    hop_length = 256
    win_length = 512
    window = torch.hann_window(win_length).to(device)
    
    # Compute STFT
    waveform_device = waveform.to(device)
    
    stft = torch.stft(waveform_device, n_fft=n_fft, hop_length=hop_length, 
                      win_length=win_length, window=window, return_complex=True)
                      
    mag = torch.abs(stft)
    phase = torch.angle(stft)
    
    # Prepare features for model: [Batch=1, Time, Freq]
    features = torch.log1p(mag).transpose(0, 1).unsqueeze(0)
    
    # Predict mask
    with torch.no_grad():
        mask = model(features)
        
    # Apply mask
    mask = mask.squeeze(0).transpose(0, 1)
    enhanced_mag = mag * mask
    
    # Reconstruct
    enhanced_stft = enhanced_mag * torch.exp(1j * phase)
    enhanced_waveform = torch.istft(enhanced_stft, n_fft=n_fft, hop_length=hop_length, 
                                    win_length=win_length, window=window)
                                    
    return enhanced_waveform.cpu().numpy()

def denoise_audio(model_path: str, input_wav: str, output_wav: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = DenoiseCRNN().to(device)
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load audio bypassing torchaudio.load to avoid missing codecs
    data, sr = sf.read(input_wav, dtype='float32')
    
    enhanced_waveform_np = denoise_array(model, data, sr)
    
    sf.write(output_wav, enhanced_waveform_np, 16000)
    print(f"Denoised audio saved to {output_wav}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for THCHS-30 Denoising Model")
    parser.add_argument('--model_path', type=str, default='ai/denoise/weights/denoise_crnn.pt', help='Path to trained model weights')
    parser.add_argument('--input', type=str, required=True, help='Path to input noisy audio wav')
    parser.add_argument('--output', type=str, default='output_clean.wav', help='Path to output clean audio wav')
    
    args = parser.parse_args()
    denoise_audio(args.model_path, args.input, args.output)
