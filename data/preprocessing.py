
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split


class SkeletonMultiModalViews(nn.Module):
    """
    Constructs three complementary views from a skeleton sequence tensor.

    Args:
        skeleton : Tensor [B, T, V, C]

    Returns:
        (node_features, s_joint, s_motion)
    """

    def forward(
        self, skeleton: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, V, C = skeleton.shape
       
        s_joint = skeleton.reshape(B, T, V * C)                # [B, T, V*C]
        last      = skeleton[:, -1:, :, :]
        skel_pad1 = torch.cat([skeleton, last],        dim=1)  # [B, T+1, V, C]
        skel_pad2 = torch.cat([skeleton, last, last],  dim=1)  # [B, T+2, V, C]

        slow = skel_pad1[:, 1:T+1, :, :] - skeleton           # [B, T, V, C]
        fast = skel_pad2[:, 2:T+2, :, :] - skeleton           # [B, T, V, C]

        s_motion = torch.cat([fast, slow], dim=-1)             # [B, T, V, 2C]
        s_motion = s_motion.reshape(B, T, V * 2 * C)          # [B, T, V*2C]

        node_features = skeleton                               # [B, T, V, C]

        return node_features, s_joint, s_motion


----------------------------------------------

class SlidingWindowSampler:
    """
    Extracts overlapping T-frame subsequences from a skeleton sequence.

    Args:
        window_size : T  — window length in frames (paper: 400).
        stride      : frame step between consecutive windows (paper: 1).
    """

    def __init__(self, window_size: int = 400, stride: int = 1):
        self.window_size = window_size
        self.stride      = stride

    def __call__(
        self, seq: np.ndarray, label: int
    ) -> List[Dict[str, object]]:
        """
        Args:
            seq   : float32 array [T_raw, V, C]  — full skeleton sequence.
            label : int  — fall-risk class of the subject.

        Returns:
            List of dicts with keys 'seq' ([window_size, V, C]) and 'label'.
        """
        T_raw = seq.shape[0]
        W     = self.window_size

        # Pad short sequences
        if T_raw < W:
            reps = math.ceil(W / T_raw)
            seq  = np.tile(seq, (reps, 1, 1))[:W]
            return [{"seq": seq.astype(np.float32), "label": label}]

        windows = []
        for start in range(0, T_raw - W + 1, self.stride):
            windows.append({
                "seq":   seq[start : start + W].astype(np.float32),
                "label": label,
            })
        return windows


class SkeletonAugmentor:

    def __init__(
        self,
        noise_std: float       = 0.01,
        rotate_prob: float     = 0.5,
        joint_drop_prob: float = 0.1,
        coord_mask_prob: float = 0.1,
        seed: Optional[int]    = None,
    ):
        self.noise_std       = noise_std
        self.rotate_prob     = rotate_prob
        self.joint_drop_prob = joint_drop_prob
        self.coord_mask_prob = coord_mask_prob
        self.rng             = np.random.default_rng(seed)


    def _gaussian_noise(self, seq: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(0.0, self.noise_std, seq.shape).astype(np.float32)
        return seq + noise

    def _random_rotation(self, seq: np.ndarray) -> np.ndarray:
        
        if seq.shape[-1] != 3:
            return seq
        if self.rng.random() > self.rotate_prob:
            return seq

        # Sample random rotation matrix via random unit quaternion
        u = self.rng.uniform(0.0, 1.0, size=3).astype(np.float64)
        q0 = np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1])
        q1 = np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1])
        q2 = np.sqrt(u[0])     * np.sin(2 * np.pi * u[2])
        q3 = np.sqrt(u[0])     * np.cos(2 * np.pi * u[2])

        
        R = np.array([
            [1 - 2*(q2**2 + q3**2),   2*(q1*q2 - q0*q3),   2*(q1*q3 + q0*q2)],
            [  2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2),   2*(q2*q3 - q0*q1)],
            [  2*(q1*q3 - q0*q2),   2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)],
        ], dtype=np.float32)   # [3, 3]

        
        return (seq @ R.T).astype(np.float32)

    def _joint_removal(self, seq: np.ndarray) -> np.ndarray:
        """(c) Stochastic per-joint zeroing across all frames."""
        T, V, C = seq.shape
        mask = (self.rng.random(V) > self.joint_drop_prob).astype(np.float32)
        
        return seq * mask[np.newaxis, :, np.newaxis]

    def _coordinate_masking(self, seq: np.ndarray) -> np.ndarray:
        """(d) Stochastic per-axis masking across all joints and frames."""
        C    = seq.shape[-1]
        mask = (self.rng.random(C) > self.coord_mask_prob).astype(np.float32)
        
        return seq * mask[np.newaxis, np.newaxis, :]


    def __call__(self, seq: np.ndarray) -> np.ndarray:
        """
        Applies all four augmentations in sequence to a single skeleton array.

        Args:
            seq : float32 array [T, V, C].

        Returns:
            Augmented array of the same shape.
        """
        seq = self._gaussian_noise(seq)
        seq = self._random_rotation(seq)
        seq = self._joint_removal(seq)
        seq = self._coordinate_masking(seq)
        return seq



class FallRiskDataset(Dataset):
    """
    Base dataset of paired TUG and FTSTS skeleton windows with fall-risk labels.

    Args:
        data_root : directory of .npz files (mutually exclusive with samples).
        samples   : list of dicts {'tug', 'ftsts', 'label'}.
        seq_len   : fixed temporal length T (paper: 400).
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        samples: Optional[List[Dict]] = None,
        seq_len: int = 400,
    ):
        self.seq_len = seq_len

        if samples is not None:
            self.samples = samples
        elif data_root is not None:
            self.samples = self._load_npz(data_root)
        else:
            raise ValueError("Provide either 'data_root' or 'samples'.")

    @staticmethod
    def _load_npz(root: str) -> List[Dict]:
        files = sorted(Path(root).glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found in '{root}'.")
        samples = []
        for p in files:
            d = np.load(p)
            samples.append({
                "tug":   d["tug"].astype(np.float32),
                "ftsts": d["ftsts"].astype(np.float32),
                "label": int(d["label"]),
            })
        return samples

    def _fix_len(self, seq: np.ndarray) -> np.ndarray:
        """Pad (tile) or crop to self.seq_len along the time axis."""
        T = seq.shape[0]
        if T >= self.seq_len:
            return seq[:self.seq_len]
        reps = math.ceil(self.seq_len / T)
        return np.tile(seq, (reps, 1, 1))[:self.seq_len]


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s     = self.samples[idx]
        tug   = torch.from_numpy(self._fix_len(s["tug"]))    # [T, V, C]
        ftsts = torch.from_numpy(self._fix_len(s["ftsts"]))  # [T, V, C]
        label = torch.tensor(s["label"], dtype=torch.long)
        return tug, ftsts, label


class AugmentedDataset(Dataset):
    
    def __init__(self, base_dataset: Dataset, augmentor: SkeletonAugmentor):
        self.base_dataset = base_dataset
        self.augmentor    = augmentor

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tug, ftsts, label = self.base_dataset[idx]

        # Convert to numpy, augment, convert back
        tug_aug   = self.augmentor(tug.numpy())
        ftsts_aug = self.augmentor(ftsts.numpy())

        return (
            torch.from_numpy(tug_aug),
            torch.from_numpy(ftsts_aug),
            label,
        )


def build_windowed_dataset(
    subject_records: List[Dict],
    window_size: int = 400,
    stride: int = 1,
    seq_len: int = 400,
) -> FallRiskDataset:
    
    sampler = SlidingWindowSampler(window_size=window_size, stride=stride)
    all_windows = []

    for record in subject_records:
        tug_raw   = record["tug"]
        ftsts_raw = record["ftsts"]
        label     = record["label"]

        # Align raw lengths by padding the shorter one
        T_tug, T_fts = tug_raw.shape[0], ftsts_raw.shape[0]
        T_max = max(T_tug, T_fts)

        def _pad_to(seq: np.ndarray, length: int) -> np.ndarray:
            if seq.shape[0] >= length:
                return seq
            reps = math.ceil(length / seq.shape[0])
            return np.tile(seq, (reps, 1, 1))[:length]

        tug_raw   = _pad_to(tug_raw,   T_max)
        ftsts_raw = _pad_to(ftsts_raw, T_max)

        # Extract windows using shared indices
        T_raw = tug_raw.shape[0]
        W     = window_size

        if T_raw < W:
            # Pad to window size and yield single sample
            tug_w   = _pad_to(tug_raw,   W)
            ftsts_w = _pad_to(ftsts_raw, W)
            all_windows.append({
                "tug":   tug_w.astype(np.float32),
                "ftsts": ftsts_w.astype(np.float32),
                "label": label,
            })
        else:
            for start in range(0, T_raw - W + 1, stride):
                all_windows.append({
                    "tug":   tug_raw[start : start + W].astype(np.float32),
                    "ftsts": ftsts_raw[start : start + W].astype(np.float32),
                    "label": label,
                })

    return FallRiskDataset(samples=all_windows, seq_len=seq_len)


def build_dataset_splits(
    dataset: FallRiskDataset,
    n_test: int              = 100,
    train_val_ratio: float   = 9.0,
    batch_size: int          = 16,
    augment_train: bool      = True,
    augmentor: Optional[SkeletonAugmentor] = None,
    seed: int                = 42,
    num_workers: int         = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    n_total = len(dataset)
    if n_total <= n_test + 1:
        raise ValueError(
            f"Dataset has only {n_total} samples but {n_test} are needed for "
            f"the test set alone.  Reduce n_test or increase the dataset size."
        )

    generator = torch.Generator().manual_seed(seed)

    n_trainval = n_total - n_test
    trainval_subset, test_subset = random_split(
        dataset,
        [n_trainval, n_test],
        generator=generator,
    )

    val_ratio = 1.0 / (train_val_ratio + 1.0)          # 1/(9+1) = 0.1
    n_val     = max(1, round(n_trainval * val_ratio))
    n_train   = n_trainval - n_val

    train_subset, val_subset = random_split(
        trainval_subset,
        [n_train, n_val],
        generator=generator,
    )

    if augment_train:
        if augmentor is None:
            augmentor = SkeletonAugmentor()
        train_ds = AugmentedDataset(train_subset, augmentor)
    else:
        train_ds = train_subset

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    train_loader = DataLoader(
        train_ds,
        shuffle=True,       # reshuffled at every epoch (paper requirement)
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_subset,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_subset,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader
