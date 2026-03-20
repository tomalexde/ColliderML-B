from common_imports import *
from torch.utils.data import DataLoader, Dataset
import pickle


# =============================================================================
# Shared dataset class
# =============================================================================

class TrackDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# Collate functions
# =============================================================================

def collate_padded(batch):
    """
    For SDPA version. Pads hits to the longest sequence in the batch.

    Returns:
        x_padded : (B, max_hits, 3)   float32 — zeros at pad positions
        mask     : (B, max_hits)      bool    — True = pad (ignored by attention)
        y_tensor : (B,)               long
    """
    x_list = [item[0] for item in batch]
    y_list = [item[1] for item in batch]

    lengths = [x.shape[0] for x in x_list]
    max_len = max(lengths)
    B       = len(x_list)
    feat    = x_list[0].shape[1]   # 3

    x_padded = torch.zeros(B, max_len, feat, dtype=torch.float32)
    mask     = torch.ones( B, max_len,       dtype=torch.bool)   # True = pad

    for i, (x, length) in enumerate(zip(x_list, lengths)):
        x_padded[i, :length] = x
        mask[i, :length]     = False

    y_tensor = torch.tensor(y_list, dtype=torch.long)
    return x_padded, mask, y_tensor


def collate_packed(batch):
    """
    For FlashAttention version. Packs all real hits end-to-end with no padding.

    Returns:
        x_packed   : (total_hits, 3)   float32 — all real hits, no zeros
        cu_seqlens : (B+1,)            int32   — cumulative hit counts
        max_seqlen : int               — longest event in this batch
        y_tensor   : (B,)              long
    """
    x_list = [item[0] for item in batch]
    y_list = [item[1] for item in batch]

    lengths    = [x.shape[0] for x in x_list]
    max_seqlen = max(lengths)

    x_packed   = torch.cat(x_list, dim=0)
    cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    torch.cumsum(torch.tensor(lengths, dtype=torch.int32), dim=0, out=cu_seqlens[1:])

    y_tensor = torch.tensor(y_list, dtype=torch.long)
    return x_packed, cu_seqlens, max_seqlen, y_tensor


# =============================================================================
# DataModules
# =============================================================================

class PaddedDataModule(pl.LightningDataModule):
    """For SDPA version — returns (x_padded, mask, y) batches."""

    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
        super().__init__()
        self.X_train, self.y_train = X_train, y_train
        self.X_val,   self.y_val   = X_val,   y_val
        self.X_test,  self.y_test  = X_test,  y_test
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_ds = TrackDataset(self.X_train, self.y_train)
        self.val_ds   = TrackDataset(self.X_val,   self.y_val)
        self.test_ds  = TrackDataset(self.X_test,  self.y_test)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, collate_fn=collate_padded, num_workers=31)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          collate_fn=collate_padded, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          collate_fn=collate_padded, num_workers=4)


class PackedDataModule(pl.LightningDataModule):
    """For FlashAttention version — returns (x_packed, cu_seqlens, max_seqlen, y) batches."""

    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
        super().__init__()
        self.X_train, self.y_train = X_train, y_train
        self.X_val,   self.y_val   = X_val,   y_val
        self.X_test,  self.y_test  = X_test,  y_test
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_ds = TrackDataset(self.X_train, self.y_train)
        self.val_ds   = TrackDataset(self.X_val,   self.y_val)
        self.test_ds  = TrackDataset(self.X_test,  self.y_test)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, collate_fn=collate_packed, num_workers=31)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          collate_fn=collate_packed, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          collate_fn=collate_packed, num_workers=4)


# =============================================================================
# Builders
# =============================================================================

def _split(X, y):
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=42, stratify=y_tv)
    return X_train, y_train, X_val, y_val, X_test, y_test


def DataToDataModule(batch_size, X1, I1, X2, I2, X3, I3, X4, I4, mode='sdpa'):
    X = X1 + X2 + X3 + X4
    y = np.concatenate([I1, I2, I3, I4])
    return _make_datamodule(batch_size, X, y, mode)


def DataToDataModule_1d(batch_size, X, y, mode='sdpa'):
    return _make_datamodule(batch_size, X, y, mode)


def _make_datamodule(batch_size, X, y, mode):
    X_train, y_train, X_val, y_val, X_test, y_test = _split(X, y)
    cls = PaddedDataModule if mode == 'sdpa' else PackedDataModule
    return cls(X_train, y_train, X_val, y_val, X_test, y_test, batch_size)


def save_datamodule(data, filepath):
    """Save raw (X, y) lists to disk."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved to {filepath}")


def DataLoad(filepath, batch_size, mode='sdpa'):
    """
    Load (X, y) from pickle and return a DataModule.

    mode: 'sdpa'  → PaddedDataModule  (for SDPA transformer)
          'flash' → PackedDataModule  (for FlashAttention transformer)
    """
    with open(filepath, 'rb') as f:
        X, y = pickle.load(f)
    print(f"Loaded from {filepath} — {len(X)} events, mode={mode}")
    return DataToDataModule_1d(batch_size, X, y, mode=mode)