from common_imports import *
from torch.utils.data import DataLoader, Dataset
import torch.nested

def DataToDataModule(batch_size, X1, I1, X2, I2, X3, I3, X4, I4):
    """
    Converts ragged hit lists and labels into a NestedTrackDataModule.
    
    Parameters:
    -----------
    X_list : list of torch.Tensors (each of shape [n_hits, 3])
    y_list : np.ndarray of labels
    """
    # Combine datasets
    #X = np.vstack([X1, X2, X3, X4])
    X = X1+X2+X3+X4
    y = np.concatenate([I1, I2, I3, I4])

    # Split data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

    return(NestedDataModule(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        batch_size=batch_size
    ))

class NestedDataset(Dataset):
    """Simple wrapper to hold lists of tensors for variable-length events."""
    def __init__(self, event_list):
        self.event_list = event_list

    def __len__(self):
        return len(self.event_list)

    def __getitem__(self, idx):
        return self.event_list[idx]

class NestedDataModule(pl.LightningDataModule):
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
        super().__init__()
        # Important: These are Python LISTS of tensors, not a single stacked tensor.
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_ds = NestedDataset(self.X_train, self.y_train)
        self.val_ds   = NestedDataset(self.X_val, self.y_val)
        self.test_ds  = NestedDataset(self.X_test, self.y_test)

    def collate_nested(self, batch):
        """
        This is where the magic happens. 
        It takes a batch of individual tensors and 'zips' them into a NestedTensor.
        """
        x_list = [item[0] for item in batch]
        y_list = [item[1] for item in batch]

        # 1. Create the NestedTensor (the efficient ragged structure)
        x_nested = torch.nested.as_nested_tensor(x_list)
        
        # 2. Convert labels to a standard tensor
        y_tensor = torch.tensor(y_list, dtype=torch.long)
        
        return x_nested, y_tensor

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=self.collate_nested,
            num_workers=4  # Helps with data loading speed
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            collate_fn=self.collate_nested
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            collate_fn=self.collate_nested
        )