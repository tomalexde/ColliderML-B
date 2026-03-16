from common_imports import *
from torch.utils.data import DataLoader, Dataset
import pickle

def DataToDataModule(batch_size, X1, I1, X2, I2, X3, I3, X4, I4):
    X = X1 + X2 + X3 + X4
    y = np.concatenate([I1, I2, I3, I4])

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

    return PaddedDataModule(
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

    return PaddedDataModule(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        batch_size=batch_size
    )

def save_datamodule(data_module, filepath):
    """
    Saves a PaddedDataModule to disk using pickle.
    
    Parameters:
    -----------
    data_module : PaddedDataModule
        The datamodule to save
    filepath : str
        Path to save the pickle file e.g. '/global/cfs/cdirs/m4958/usr/emil_sd/data_module.pkl'
    """
    with open(filepath, 'wb') as f:
        pickle.dump(data_module, f)
    print(f"DataModule saved to {filepath}")


def load_datamodule(filepath):
    """
    Loads a PaddedDataModule from disk using pickle.
    
    Parameters:
    -----------
    filepath : str
        Path to the pickle file
    
    Returns:
    --------
    data_module : PaddedDataModule
        The loaded datamodule, ready to pass to trainer.fit()
    """
    with open(filepath, 'rb') as f:
        data_module = pickle.load(f)
    print(f"DataModule loaded from {filepath}")
    return data_module

class TrackDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def sort_hits_by_radius(x: torch.Tensor) -> torch.Tensor:
    """
    Option 1: Sort hits by cylindrical radius r = sqrt(x^2 + y^2).

    Returns:
        x_padded : (B, max_hits_padded, 3)   float32
        mask     : (B, max_hits_padded)       bool, True = pad
        y_tensor : (B,)                       long
    """
    x_list = [item[0] for item in batch]
    y_list = [item[1] for item in batch]

    lengths    = [x.shape[0] for x in x_list]
    max_len = max(lengths)

    B    = len(x_list)
    feat = x_list[0].shape[1]   # 3

    x_padded = torch.zeros(B, max_len, feat, dtype=torch.float32)
    mask     = torch.ones(B, max_len, dtype=torch.bool)   # True = pad

    for i, (x, length) in enumerate(zip(x_list, lengths)):
        x_padded[i, :length] = x
        mask[i, :length]     = False

    y_tensor = torch.tensor(y_list, dtype=torch.long)
    return x_padded, mask, y_tensor


class PaddedDataModule(pl.LightningDataModule):
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
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_padded,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            collate_fn=collate_padded,
            num_workers=0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            collate_fn=collate_padded,
            num_workers=0,
        )