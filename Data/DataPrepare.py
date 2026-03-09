from common_imports import *
from ..filepaths import Filepath
from DataModule import DataToDataModule

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
        
        # Create tensor
        hits_data = hits_data[:max_hits] 
        X_list.append(torch.tensor(hits_data, dtype=torch.float32))
        ids_list.append(event_id)
        valid_count += 1
    
    # Stack into final arrays
    if valid_count == 0:
        # Return empty arrays if no valid events
        X = np.zeros((0, max_hits, 3), dtype=np.float32)
        ids = np.array([], dtype=type(event_id))
        X_list.append(torch.tensor(X, dtype=torch.float32))
    else:
        ids = np.array(ids_list, dtype=type(event_id))
    
    return X_list, ids

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


def prepare_it_all(events, purity_scale, maxhits, batch_size):
    '''
    Prepare data for all four datasets
    '''
    filepath = Filepath()
    #event_id ttbar is 0, id ggf is 1, id dihiggs is 2, id higsportal is 3

    X_ttbar, ids_ttbar = prepare_data(
        num_events=events,
        hits_dir=filepath.ttbar_base_hits_dir,
        tracks_dir=filepath.ttbar_base_tracks_dir,
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

    X_ggf, ids_ggf = prepare_data(
        num_events=events,
        hits_dir=filepath.ggf_base_hits_dir,
        tracks_dir=filepath.ggf_base_tracks_dir,
        event_id=1,  # Label all ttbar events as 1
        purity_scale=purity_scale,
        max_hits=maxhits
    )


    X_dihiggs, ids_dihiggs = prepare_data(
        num_events=events,
        hits_dir=filepath.dihiggs_base_hits_dir,
        tracks_dir=filepath.dihiggs_base_tracks_dir,
        event_id=2,  # Label all ttbar events as 2
        purity_scale=purity_scale,
        max_hits=maxhits
    )

    X_higgs_portal, ids_higgs_portal = prepare_data(
        num_events=events,
        hits_dir=filepath.higgs_portal_base_hits_dir,
        tracks_dir=filepath.higgs_portal_base_tracks_dir,
        event_id=3,  # Label all ttbar events as 3
        purity_scale=purity_scale,
        max_hits=maxhits
    )

    return DataToDataModule(batch_size, X_ttbar, ids_ttbar,X_ggf, ids_ggf,X_dihiggs, ids_dihiggs,X_higgs_portal, ids_higgs_portal)

