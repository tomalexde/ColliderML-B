from common_imports import *
ttbar_base_hits_dir = "/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/ttbar/v1/parquet/reco/tracker_hits"
ttbar_base_tracks_dir = "/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/ttbar/v1/parquet/reco/tracks"

dihiggs_base_hits_dir = "/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/dihiggs/v1/parquet/reco/tracker_hits"
dihiggs_base_tracks_dir = "/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/dihiggs/v1/parquet/reco/tracks"

higgs_portal_base_hits_dir = "/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/higgs_portal/v1/parquet/reco/tracker_hits"
higgs_portal_base_tracks_dir = "/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/higgs_portal/v1/parquet/reco/tracks"

ggf_base_hits_dir = "/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/ggf/v1/parquet/reco/tracker_hits"
ggf_base_tracks_dir = "/global/cfs/cdirs/m4958/data/ColliderML/simulation/hard_scatter/ggf/v1/parquet/reco/tracks"

def prepare_data(
    num_events,
    hits_dir,
    tracks_dir,
    event_id,
    purity_scale,
    max_hits
):
    """
    Prepare ML dataset with track-associated hits and optional raw hits.
    
    Track-associated hits take absolute priority. Raw hits are added based on purity_scale.
    max_hits parameter takes priority over purity_scale.
    
    Parameters:
    -----------
    num_events : int or array-like
        If int, use range(num_events). If array, use as event IDs.
    hits_dir : str
        Directory path to hits parquet data
    tracks_dir : str
        Directory path to tracks parquet data
    event_id : int or str
        Single identifier to assign to all output events
    purity_scale : int/float or False
        Multiplier for raw hits per track hit.
        - 0: only track-associated hits
        - N (positive number): N raw hits per track hit (capped by availability and max_hits)
        - False: include all available raw hits up to max_hits
    max_hits : int
        Maximum hits per event. Takes absolute priority over purity_scale.
    
    Returns:
    --------
    X : np.ndarray
        Shape [n_valid_events, max_hits, 3] containing [x, y, z] coordinates
    masks : np.ndarray
        Shape [n_valid_events, max_hits] where 0=real data, 1=padding
    ids : np.ndarray
        Shape [n_valid_events] all filled with event_id parameter
    """
    
    # Convert num_events to array if needed
    if isinstance(num_events, int):
        events = np.arange(num_events, dtype=np.int32)
    else:
        events = np.asarray(list(num_events), dtype=np.int32)
    
    # Load data efficiently
    hits_df = utils.read_events_hits(hits_dir, events)
    tracks_df = utils.read_events_tracks(tracks_dir, events)
    
    # Pre-group for fast lookup
    hits_by_event = {eid: group[['x', 'y', 'z']].values 
                     for eid, group in hits_df.groupby('event_id')}
    
    tracks_by_event = {eid: group['hit_ids'].values 
                       for eid, group in tracks_df.groupby('event_id')}
    
    # Output arrays
    valid_count = 0
    X_list = []
    masks_list = []
    ids_list = []
    
    # Process each event
    for event_id_val in events:
        # Skip if event not in data
        if event_id_val not in hits_by_event:
            continue
        
        all_hits = hits_by_event[event_id_val]
        track_hit_ids_nested = tracks_by_event.get(event_id_val, np.array([], dtype=object))
        
        # Flatten and unique all track hit IDs for this event
        if len(track_hit_ids_nested) > 0:
            track_hit_ids = np.unique(np.concatenate(track_hit_ids_nested))
        else:
            track_hit_ids = np.array([], dtype=np.int32)
        
        # Filter valid indices (within bounds, non-negative)
        valid_mask = (track_hit_ids >= 0) & (track_hit_ids < len(all_hits))
        track_hit_ids = track_hit_ids[valid_mask]
        
        # Extract track-associated hits
        if len(track_hit_ids) > 0:
            track_hits = all_hits[track_hit_ids]
        else:
            track_hits = np.zeros((0, 3), dtype=np.float32)
        
        n_track_hits = len(track_hits)
        
        # Determine total hits to include
        filled_count = n_track_hits
        
        # Add raw hits if purity_scale is set and space available
        if filled_count < max_hits:
            if purity_scale is not False:
                # Calculate how many raw hits to add
                n_raw_available = len(all_hits) - n_track_hits
                n_raw_to_add = int(n_track_hits * purity_scale)
                n_raw_to_add = min(n_raw_to_add, n_raw_available, max_hits - filled_count)
            else:
                # Include all remaining raw hits up to max_hits
                n_raw_available = len(all_hits) - n_track_hits
                n_raw_to_add = min(n_raw_available, max_hits - filled_count)
            
            if n_raw_to_add > 0:
                # Get raw hit indices (exclude track hits)
                track_hit_set = set(track_hit_ids)
                raw_indices = np.array([i for i in range(len(all_hits)) if i not in track_hit_set], dtype=np.int32)
                
                # Randomly sample raw hits (fast)
                raw_indices = np.random.choice(raw_indices, size=n_raw_to_add, replace=False)
                raw_hits = all_hits[raw_indices]
                
                # Combine track and raw hits
                hits_data = np.vstack([track_hits, raw_hits])
                filled_count = len(hits_data)
            else:
                hits_data = track_hits
        else:
            # Only track hits (capped at max_hits)
            hits_data = track_hits[:max_hits]
            filled_count = len(hits_data)
        
        # Create padded array
        X_event = np.zeros((max_hits, 3), dtype=np.float32)
        mask_event = np.ones(max_hits, dtype=np.uint8)
        
        X_event[:filled_count] = hits_data[:max_hits]
        mask_event[:filled_count] = 0
        
        X_list.append(X_event)
        masks_list.append(mask_event)
        ids_list.append(event_id)
        valid_count += 1
    
    # Stack into final arrays
    if valid_count == 0:
        # Return empty arrays if no valid events
        X = np.zeros((0, max_hits, 3), dtype=np.float32)
        masks = np.zeros((0, max_hits), dtype=np.uint8)
        ids = np.array([], dtype=type(event_id))
    else:
        X = np.stack(X_list, axis=0)
        masks = np.stack(masks_list, axis=0)
        ids = np.array(ids_list, dtype=type(event_id))
    
    return X, masks, ids

def calculate_max_hits_from_purity(
    num_events,
    hits_dir,
    tracks_dir,
    purity_scale
):
    """
    Calculate the necessary max_hits value for a given purity_scale.
    
    Determines what max_hits should be set to in order to achieve the desired
    purity_scale across all events without exceeding available data.
    
    Parameters:
    -----------
    num_events : int or array-like
        If int, use range(num_events). If array, use as event IDs.
    hits_dir : str
        Directory path to hits parquet data
    tracks_dir : str
        Directory path to tracks parquet data
    purity_scale : int or float
        Desired multiplier for raw hits per track hit (e.g., 2 = 2 raw hits per track hit)
    
    Returns:
    --------
    max_hits : int
        Recommended max_hits value to achieve the desired purity_scale
    """
    
    # Convert num_events to array if needed
    if isinstance(num_events, int):
        events = np.arange(num_events, dtype=np.int32)
    else:
        events = np.asarray(list(num_events), dtype=np.int32)
    
    # Load data efficiently
    hits_df = utils.read_events_hits(hits_dir, events)
    tracks_df = utils.read_events_tracks(tracks_dir, events)
    
    # Pre-group for fast lookup
    hits_by_event = {eid: group[['x', 'y', 'z']].values 
                     for eid, group in hits_df.groupby('event_id')}
    
    tracks_by_event = {eid: group['hit_ids'].values 
                       for eid, group in tracks_df.groupby('event_id')}
    
    max_hits_needed = []
    
    # Calculate max_hits needed for each event
    for event_id_val in events:
        if event_id_val not in hits_by_event:
            continue
        
        all_hits = hits_by_event[event_id_val]
        track_hit_ids_nested = tracks_by_event.get(event_id_val, np.array([], dtype=object))
        
        # Flatten and unique all track hit IDs
        if len(track_hit_ids_nested) > 0:
            track_hit_ids = np.unique(np.concatenate(track_hit_ids_nested))
        else:
            track_hit_ids = np.array([], dtype=np.int32)
        
        # Filter valid indices
        valid_mask = (track_hit_ids >= 0) & (track_hit_ids < len(all_hits))
        track_hit_ids = track_hit_ids[valid_mask]
        
        n_track_hits = len(track_hit_ids)
        n_raw_available = len(all_hits) - n_track_hits
        
        # Calculate hits needed: track hits + (track hits * purity_scale)
        n_raw_to_add = int(n_track_hits * purity_scale)
        n_raw_to_add = min(n_raw_to_add, n_raw_available)
        
        event_max_hits = n_track_hits + n_raw_to_add
        max_hits_needed.append(event_max_hits)
    
    # Return maximum across all events (ensures all events fit)
    max_hits = max(max_hits_needed) if max_hits_needed else 0
    
    return int(max_hits)
#Udregning af max num of hits
#events = range(10000)
#ttbar = calculate_max_hits_from_purity(events, ttbar_base_hits_dir,ttbar_base_tracks_dir,10000)
#print(ttbar)
#res 10810

class TrackDataModule(pl.LightningDataModule):
    def __init__(self, X_train, masks_train, y_train, 
                 X_val, masks_val, y_val, 
                 X_test, masks_test, y_test, 
                 batch_size=32):
        super().__init__()
        self.X_train = X_train
        self.masks_train = masks_train
        self.y_train = y_train
        
        self.X_val = X_val
        self.masks_val = masks_val
        self.y_val = y_val
        
        self.X_test = X_test
        self.masks_test = masks_test
        self.y_test = y_test
        
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        # Create datasets with masks
        self.train_dataset = TensorDataset(self.X_train, self.masks_train, self.y_train)
        self.val_dataset = TensorDataset(self.X_val, self.masks_val, self.y_val)
        self.test_dataset = TensorDataset(self.X_test, self.masks_test, self.y_test)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

class LightningNeuralNetwork(pl.LightningModule):
    def __init__(self, feature_dim=7, hidden_size=128, num_heads=4, 
                 num_encoder_layers=2, output_size=1, learning_rate=0.0001):
        super().__init__()

        # Save all hyperparameters
        self.save_hyperparameters()
        
        self.model = cool_transformer(feature_dim=feature_dim, 
                                     hidden_size=hidden_size, 
                                     num_heads=num_heads, 
                                     num_encoder_layers=num_encoder_layers, 
                                     output_size=output_size)
        self.learning_rate = learning_rate

        # Cross entropy loss for multiclass classification
        self.loss_function = nn.CrossEntropyLoss()
        
    def forward(self, x, mask=None):
        return self.model(x, mask)
    
    def training_step(self, batch, batch_idx):
        x, mask, y = batch
        y_hat = self.model(x, mask)
        loss = self.loss_function(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask, y = batch
        y_hat = self.model(x, mask)
        loss = self.loss_function(y_hat, y)

        # Log AUC
        y_cpu = y.cpu().detach().numpy()
        y_hat_cpu = y_hat.cpu().detach().numpy()
        auc = roc_auc_score(y_cpu, y_hat_cpu)
        self.log('val_auc', auc)

        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, mask, y = batch
        y_hat = self(x, mask)
        loss = self.loss_function(y_hat, y)

        # Log AUC
        y_cpu = y.cpu().detach().numpy()
        y_hat_cpu = y_hat.cpu().detach().numpy()
        auc = roc_auc_score(y_cpu, y_hat_cpu)
        self.log('test_auc', auc)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

def data_to_DataModule_2(X1, M1, I1, X2, M2, I2):
    
    # Combine datasets
    X = np.vstack([X1, X2])
    masks = np.vstack([M1, M2])
    y = np.concatenate([I1, I2])

    # Split data into train, validation, and test sets
    X_train_val, X_test, masks_train_val, masks_test, y_train_val, y_test = train_test_split(
        X, masks, y, test_size=0.2, random_state=42, stratify=y)

    X_train, X_val, masks_train, masks_val, y_train, y_val = train_test_split(
        X_train_val, masks_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    masks_train_tensor = torch.tensor(masks_train, dtype=torch.bool).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    masks_val_tensor = torch.tensor(masks_val, dtype=torch.bool).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    masks_test_tensor = torch.tensor(masks_test, dtype=torch.bool).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    data_module = TrackDataModule(
        X_train_tensor, masks_train_tensor, y_train_tensor,
        X_val_tensor, masks_val_tensor, y_val_tensor,
        X_test_tensor, masks_test_tensor, y_test_tensor,
        batch_size=32
    )

    return data_module
def data_to_DataModule_4(X1, M1, I1, X2, M2, I2,X3, M3, I3, X4, M4, I4):
    
    # Combine datasets
    X = np.vstack([X1, X2,X3,X4])
    masks = np.vstack([M1, M2,M3,M4])
    y = np.concatenate([I1, I2,I3,I4])

    # Split data into train, validation, and test sets
    X_train_val, X_test, masks_train_val, masks_test, y_train_val, y_test = train_test_split(
        X, masks, y, test_size=0.2, random_state=42, stratify=y)

    X_train, X_val, masks_train, masks_val, y_train, y_val = train_test_split(
        X_train_val, masks_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    masks_train_tensor = torch.tensor(masks_train, dtype=torch.bool).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    masks_val_tensor = torch.tensor(masks_val, dtype=torch.bool).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    masks_test_tensor = torch.tensor(masks_test, dtype=torch.bool).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    data_module = TrackDataModule(
        X_train_tensor, masks_train_tensor, y_train_tensor,
        X_val_tensor, masks_val_tensor, y_val_tensor,
        X_test_tensor, masks_test_tensor, y_test_tensor,
        batch_size=32
    )

    return data_module
# This is what you use

# events = range(10000)  # Use first 40 events for testing
# purity_scale = 10
# maxhits = 17000
# #event_id ttbar er 0, id ggf er 1, id dihiggs er 2, id higsportal er 3

# X_ttbar, masks_ttbar, ids_ttbar = prepare_data(
#     num_events=events,
#     hits_dir=ttbar_base_hits_dir,
#     tracks_dir=ttbar_base_tracks_dir,
#     event_id=0,  # Label all ttbar events as 0
#     purity_scale=purity_scale,
#     max_hits=maxhits
# )
# #speed numbers
# #all at 2000k max hits 2 purity scale
# #300 takes 2.1s
# #1K takes 4.9s
# #10k ttbar takes 33.6s
# # 50K takes 2m 28s

# #1k 4000 max hits takes 8.1s
# # 10k 4000 max hits takes 31.3s

# #print(f"TTBar - X shape: {X_ttbar.shape}, masks shape: {masks_ttbar.shape}, ids shape: {ids_ttbar.shape}")
# #print(f"Sample: {X_ttbar.shape[0]} events, {(masks_ttbar == 0).sum(axis=1)[:5]} hits per event (first 5)")

# X_ggf, masks_ggf, ids_ggf = prepare_data(
#     num_events=events,
#     hits_dir=ggf_base_hits_dir,
#     tracks_dir=ggf_base_tracks_dir,
#     event_id=1,  # Label all ttbar events as 1
#     purity_scale=purity_scale,
#     max_hits=maxhits
# )


# X_dihiggs, masks_dihiggs, ids_dihiggs = prepare_data(
#     num_events=events,
#     hits_dir=dihiggs_base_hits_dir,
#     tracks_dir=dihiggs_base_tracks_dir,
#     event_id=2,  # Label all ttbar events as 2
#     purity_scale=purity_scale,
#     max_hits=maxhits
# )

# X_higgs_portal, masks_higgs_portal, ids_higgs_portal = prepare_data(
#     num_events=events,
#     hits_dir=higgs_portal_base_hits_dir,
#     tracks_dir=higgs_portal_base_tracks_dir,
#     event_id=3,  # Label all ttbar events as 3
#     purity_scale=purity_scale,
#     max_hits=maxhits
# )

# data_module =  data_to_DataModule_4(X_ttbar, masks_ttbar, ids_ttbar,X_ggf, masks_ggf, ids_ggf,X_dihiggs, masks_dihiggs, ids_dihiggs,X_higgs_portal, masks_higgs_portal, ids_higgs_portal)


def prepare_it_all(events, purity_scale, maxhits):
    #event_id ttbar er 0, id ggf er 1, id dihiggs er 2, id higsportal er 3

    X_ttbar, masks_ttbar, ids_ttbar = prepare_data(
        num_events=events,
        hits_dir=ttbar_base_hits_dir,
        tracks_dir=ttbar_base_tracks_dir,
        event_id=0,  # Label all ttbar events as 0
        purity_scale=purity_scale,
        max_hits=maxhits
    )
    #speed numbers
    #all at 2000k max hits 2 purity scale
    #300 takes 2.1s
    #1K takes 4.9s
    #10k ttbar takes 33.6s
    # 50K takes 2m 28s

    #1k 4000 max hits takes 8.1s
    # 10k 4000 max hits takes 31.3s

    #print(f"TTBar - X shape: {X_ttbar.shape}, masks shape: {masks_ttbar.shape}, ids shape: {ids_ttbar.shape}")
    #print(f"Sample: {X_ttbar.shape[0]} events, {(masks_ttbar == 0).sum(axis=1)[:5]} hits per event (first 5)")

    X_ggf, masks_ggf, ids_ggf = prepare_data(
        num_events=events,
        hits_dir=ggf_base_hits_dir,
        tracks_dir=ggf_base_tracks_dir,
        event_id=1,  # Label all ttbar events as 1
        purity_scale=purity_scale,
        max_hits=maxhits
    )


    X_dihiggs, masks_dihiggs, ids_dihiggs = prepare_data(
        num_events=events,
        hits_dir=dihiggs_base_hits_dir,
        tracks_dir=dihiggs_base_tracks_dir,
        event_id=2,  # Label all ttbar events as 2
        purity_scale=purity_scale,
        max_hits=maxhits
    )

    X_higgs_portal, masks_higgs_portal, ids_higgs_portal = prepare_data(
        num_events=events,
        hits_dir=higgs_portal_base_hits_dir,
        tracks_dir=higgs_portal_base_tracks_dir,
        event_id=3,  # Label all ttbar events as 3
        purity_scale=purity_scale,
        max_hits=maxhits
    )

    return data_to_DataModule_4(X_ttbar, masks_ttbar, ids_ttbar,X_ggf, masks_ggf, ids_ggf,X_dihiggs, masks_dihiggs, ids_dihiggs,X_higgs_portal, masks_higgs_portal, ids_higgs_portal)

