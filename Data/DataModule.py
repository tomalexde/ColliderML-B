from common_imports import *
from torch.utils.data import DataLoader, Dataset
import pickle

def DataToDataModule(batch_size, X1, I1, X2, I2, X3, I3, X4, I4):
    """
    Converts ragged hit lists and labels into a PaddedDataModule.
    """
    X = X1 + X2 + X3 + X4
    y = np.concatenate([I1, I2, I3, I4])

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

    return PackedDataModule(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        batch_size=batch_size
    )
def DataToDataModule_1d(batch_size, X1, I1):
    """
    Converts ragged hit lists and labels into a PaddedDataModule.
    """
    X = X1 
    y = I1

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

    return PackedDataModule(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        batch_size=batch_size
    )

def save_datamodule(data_module, filepath):
    """
    Saves a PackedDataModule to disk using pickle.
    
    Parameters:
    -----------
    data_module : PackedDataModule
        The datamodule to save
    filepath : str
        Path to save the pickle file e.g. '/global/cfs/cdirs/m4958/usr/emil_sd/data_module.pkl'
    """
    with open(filepath, 'wb') as f:
        pickle.dump(data_module, f)
    print(f"DataModule saved to {filepath}")


def load_datamodule(filepath):
    """
    Loads a PackedDataModule from disk using pickle.
    
    Parameters:
    -----------
    filepath : str
        Path to the pickle file
    
    Returns:
    --------
    data_module : PackedDataModule
        The loaded datamodule, ready to pass to trainer.fit()
    """
    with open(filepath, 'rb') as f:
        data_module = pickle.load(f)
    print(f"DataModule loaded from {filepath}")
    return data_module

class TrackDataset(Dataset):
    """Holds a list of variable-length hit tensors and their event labels."""
    def __init__(self, X, y):
        self.X = X
        self.y = y
        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def collate_packed(batch):
    """
    Pack a batch of variable-length hit tensors end-to-end with NO padding.
 
    flash_attn_varlen_func doesn't need padding — it uses cu_seqlens to know
    where each event starts and ends, so attention is automatically blocked
    from crossing event boundaries.
 
    Returns:
        x_packed    : (total_hits, 3)     all real hits concatenated, no zeros
        cu_seqlens  : (B+1,)  int32       cumulative hit counts [0, n0, n0+n1, ...]
        max_seqlen  : int                 longest event in this batch
        y_tensor    : (B,)    long        event labels
    """
    x_list = [item[0] for item in batch]
    y_list = [item[1] for item in batch]
 
 
    lengths    = [x.shape[0] for x in x_list]
    max_seqlen = max(lengths)
 
    # Pack all events end-to-end — no padding, no wasted compute
    x_packed   = torch.cat(x_list, dim=0)                                 # (total_hits, 3)
    cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    torch.cumsum(torch.tensor(lengths, dtype=torch.int32),
                 dim=0, out=cu_seqlens[1:])                                # (B+1,)
 
    y_tensor = torch.tensor(y_list, dtype=torch.long)
    return x_packed, cu_seqlens, max_seqlen, y_tensor


class PackedDataModule(pl.LightningDataModule):
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