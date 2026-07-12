import os
import numpy as np
import torch

class MaskedData:
    """Corrupt input (80/10/10); targets = original where masked, -100 elsewhere."""
    def __init__(self, data_dir, block_size, batch_size, device, device_type,
                 eot_id, mask_id, real_vocab, mask_p=0.15):
        self.data_dir, self.block_size, self.batch_size = data_dir, block_size, batch_size
        self.device, self.device_type = device, device_type
        self.eot_id, self.mask_id, self.real_vocab, self.mask_p = eot_id, mask_id, real_vocab, mask_p
    

    def mask_and_corrupt(self, buf, span_mask=False):
        """buf: (B, L) original tokens. returns (corrupted_input, mask_flag)"""
        
        if span_mask:
            mask_flag = (self._span_mask_flag(buf.shape)) & (buf != self.eot_id)
        else:
            mask_flag = (torch.rand(buf.shape) < self.mask_p) & (buf != self.eot_id)
        r = torch.rand(buf.shape)
        corrupted = buf.clone()
        corrupted[mask_flag & (r < 0.8)] = self.mask_id # 80% mask from the selected tokens
        rand_pos = mask_flag & (r >= 0.8) & (r < 0.9) # 10% random token, get idx first
        corrupted[rand_pos] = torch.randint(0, self.real_vocab, buf.shape)[rand_pos] # draw random tokens from the real vocab and selected idx in rand_pos
        return corrupted, mask_flag
        
    def _span_mask_flag(self, shape, max_span=3):
        B, L = shape
        flag = torch.zeros(B, L, dtype=torch.bool)
        for b in range(B):                          # cheap: runs on CPU in the prefetch
            n, target = 0, int(self.mask_p * L)
            while n < target:
                s = torch.randint(1, max_span+1, (1,)).item()
                i = torch.randint(0, L, (1,)).item()
                n += (~flag[b, i:i+s]).sum().item()
                flag[b, i:i+s] = True
        return flag
        
    def get_masked_batch(self, split, ix=None, batch_size_override=None):
        """ix: (B,) window starts. None -> i.i.d. draw. The caller passes ix when it owns a sampling schedule, so the shared draw counter stays in one place (train.py)."""
        data = np.memmap(os.path.join(self.data_dir, 'train.bin' if split=='train' else 'val.bin'), dtype=np.uint16, mode='r')
        if ix is None:
            B = self.batch_size if batch_size_override is None else batch_size_override
            ix = torch.randint(len(data) - self.block_size - 1, (B, ))
        buf = torch.stack([
                torch.from_numpy(data[i:i+self.block_size+1].astype(np.int64)) for i in ix
            ]) # (B, T+1)
        corrupted, mask_flag = self.mask_and_corrupt(buf)
        x = corrupted[:, :self.block_size] # (B, T) corrupted input
        y = buf[:, 1:].clone() # (B, T) next tokens
        y[~mask_flag[:, 1:]] = -100 # supervise only <mask>
        if self.device_type == 'cuda':
            # Place the CPU tensor into pinned memory / page-locked memory. Ordinary CPU memory may be moved or swapped by the operating system; pinned memory will not be swapped, so GPU DMA copy is faster and more stable.
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
