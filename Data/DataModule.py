from common_imports import *
from torch.utils.data import DataLoader, Dataset


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

    Why radius? The ATLAS/CMS tracker is cylindrical — hits from the same
    particle form a radial arc outward from the interaction point. Sorting by
    radius groups spatially proximate hits together in the sequence, so a
    sliding window captures physically meaningful neighbourhoods instead of
    arbitrary assembly order.

    x: (N, 3) tensor of [x, y, z] hit coordinates.
    Returns the same tensor with rows reordered by ascending radius.
    """
    radius = (x[:, 0] ** 2 + x[:, 1] ** 2).sqrt()   # (N,)
    order  = radius.argsort()
    return x[order]


def collate_padded(batch, sort_by_radius: bool = True):
    """
    Collate variable-length hit tensors into a padded dense batch.

    sort_by_radius: if True (default), hits within each event are sorted by
    cylindrical radius before padding. This makes the sliding window in
    MultiHeadAttention physically meaningful — nearby positions in the
    sequence correspond to nearby detector layers.

    Returns:
        x_padded : (B, max_hits_padded, 3)   float32
        mask     : (B, max_hits_padded)       bool, True = pad
        y_tensor : (B,)                       long
    """
    x_list = [item[0] for item in batch]
    y_list = [item[1] for item in batch]

    # --- Option 1: sort each event's hits by radius ----------------------
    if sort_by_radius:
        x_list = [sort_hits_by_radius(x) for x in x_list]

    lengths = [x.shape[0] for x in x_list]
    max_len = max(lengths)   # pad to longest sequence in batch

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
            num_workers=31,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            collate_fn=collate_padded,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            collate_fn=collate_padded,
            num_workers=4,
        )