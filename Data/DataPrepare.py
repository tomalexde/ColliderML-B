from common_imports import *
from filepaths import Filepath
from Data.DataModule import DataToDataModule, DataToDataModule_1d

def prepare_data(
    num_events,
    hits_dir,
    tracks_dir,
    event_id,
    purity_scale,
    max_hits
):
    """
    Prepare ML dataset with track-associated hits and additional particles based on purity_scale.
    
    Track-associated hits are always included. Additional particles are selected based on (# of tracks) * purity_scale.
    max_hits parameter caps the total hits per event.
    
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
    purity_scale : int/float
        Multiplier for additional particles per track (e.g., 2 = 2 additional particles per track)
    max_hits : int
        Maximum hits per event.
    
    Returns:
    --------
    X_list : list of torch.Tensor
        List of tensors, each of shape [max_hits, 3] containing [x, y, z] coordinates
    ids : np.ndarray
        Shape [n_valid_events] all filled with event_id parameter
    """
    
    # Convert num_events to array if needed
    if isinstance(num_events, int):
        events = np.arange(num_events, dtype=np.int32)
    else:
        events = np.asarray(list(num_events), dtype=np.int32)
    
    # Load data efficiently
    hits_df = utils_tracks.read_events_hits(hits_dir, events)
    tracks_df = utils_tracks.read_events_tracks(tracks_dir, events)
    
    # Pre-group for fast lookup
    hits_by_event = {eid: group[['x', 'y', 'z', 'particle_id']].values 
                     for eid, group in hits_df.groupby('event_id')}
    
    tracks_by_event = {eid: group['hit_ids'].values 
                       for eid, group in tracks_df.groupby('event_id')}
    
    # Output lists
    X_list = []
    ids_list = []
    
    # Process each event
    for event_id_val in events:
        # Skip if event not in data
        if event_id_val not in hits_by_event:
            continue
        
        all_hits = hits_by_event[event_id_val]  # shape (n_hits, 4) with [x, y, z, particle_id]
        track_hit_ids_nested = tracks_by_event.get(event_id_val, np.array([], dtype=object))
        
        # Flatten and unique all track hit IDs for this event
        if len(track_hit_ids_nested) > 0:
            track_hit_ids = np.unique(np.concatenate(track_hit_ids_nested))
        else:
            track_hit_ids = np.array([], dtype=np.int32)
        
        # Filter valid indices (within bounds, non-negative)
        valid_mask = (track_hit_ids >= 0) & (track_hit_ids < len(all_hits))
        track_hit_ids = track_hit_ids[valid_mask]
        
        # Get unique particle IDs associated with tracks
        if len(track_hit_ids) > 0:
            track_pids = np.unique(all_hits[track_hit_ids, 3])
        else:
            track_pids = np.array([], dtype=all_hits.dtype[3])
        
        num_tracks = len(track_pids)  # Number of unique particles with tracks
        
        # Save all hits associated with tracks
        if len(track_hit_ids) > 0:
            track_hits = all_hits[track_hit_ids, :3]
        else:
            track_hits = np.zeros((0, 3), dtype=np.float32)
        
        # Determine additional particles: (# of tracks) * purity_scale
        n_additional_particles = int(num_tracks * purity_scale)
        
        # Get all unique particle IDs not associated with tracks
        all_pids = np.unique(all_hits[:, 3])
        non_track_pids = np.setdiff1d(all_pids, track_pids)
        
        # Select additional particles (up to available)
        n_additional_particles = min(n_additional_particles, len(non_track_pids))
        if n_additional_particles > 0:
            additional_pids = np.random.choice(non_track_pids, size=n_additional_particles, replace=False)
        else:
            additional_pids = np.array([], dtype=non_track_pids.dtype)
        
        # Get all hits for additional particles
        if len(additional_pids) > 0:
            additional_mask = np.isin(all_hits[:, 3], additional_pids)
            additional_hits = all_hits[additional_mask, :3]
        else:
            additional_hits = np.zeros((0, 3), dtype=np.float32)
        
        # Combine track hits and additional hits
        hits_data = np.vstack([track_hits, additional_hits])
        
        # Cap at max_hits
        hits_data = hits_data[:max_hits]
        
        X_list.append(torch.tensor(hits_data, dtype=torch.float32))
        ids_list.append(event_id)
    
    # Convert ids to array
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
    hits_df = utils_tracks.read_events_hits(hits_dir, events)
    tracks_df = utils_tracks.read_events_tracks(tracks_dir, events)
    
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

def create_complex_dataset(purity_array, event_list, id_list, max_hits, batch_size):
    """
    Create a combined dataset from multiple physics processes.
    
    Parameters:
    -----------
    purity_array : list
        List of purity_scale values for each dataset entry
    event_list : list of range/array-like
        List of event ranges for each dataset entry
    id_list : list of int
        List of process IDs (0=ttbar, 1=ggf, 2=dihiggs, 3=higgs_portal)
    max_hits : int
        Maximum hits per event
    
    Returns:
    --------
    X_list : list of torch.Tensor
        Combined list of hit tensors, each of shape [n_hits, 3]
    ids : np.ndarray
        Array of process IDs for each event
    """
    filepath = Filepath()
    # Directory mapping: id -> (hits_dir, tracks_dir)
    dir_map = {
        0: (filepath.ttbar_base_hits_dir,         filepath.ttbar_base_tracks_dir),
        1: (filepath.ggf_base_hits_dir,           filepath.ggf_base_tracks_dir),
        2: (filepath.dihiggs_base_hits_dir,       filepath.dihiggs_base_tracks_dir),
        3: (filepath.higgs_portal_base_hits_dir,  filepath.higgs_portal_base_tracks_dir),
    }
    
    process_names = {0: "ttbar", 1: "ggf", 2: "dihiggs", 3: "higgs_portal"}
    
    all_X    = []
    all_ids  = []
    
    for purity, events, event_id in zip(purity_array, event_list, id_list):
        hits_dir, tracks_dir = dir_map[event_id]
        
        print(f"Loading {process_names[event_id]} | events: {list(events)[0]}–{list(events)[-1]} "
              f"| purity_scale: {purity}")
        
        X_list, ids = prepare_data(
            num_events   = events,
            hits_dir     = hits_dir,
            tracks_dir   = tracks_dir,
            event_id     = event_id,
            purity_scale = purity,
            max_hits     = max_hits,
        )
        
        all_X.extend(X_list)
        all_ids.extend(ids.tolist())
    
    all_ids = np.array(all_ids, dtype=np.int32)
    
    print(f"\nDone. Total events loaded: {len(all_X)}")
    for uid in np.unique(all_ids):
        print(f"  {process_names[uid]}: {np.sum(all_ids == uid)} events")
    
    return DataToDataModule_1d(batch_size,all_X, all_ids)

def prepare_tracks_only(num_events, batch_size):
    """
    Prepare track parameter data from all four datasets.

    Reads d0, z0, phi, theta, qop directly from tracks parquet files.

    Parameters:
    -----------
    num_events : int or array-like
        Number of events (or array of event IDs) to load from each dataset.

    Returns:
    --------
    X_list : list of torch.Tensor
        Combined list of track parameter tensors from all four processes,
        each of shape [n_tracks, 5] with columns [d0, z0, phi, theta, qop]
    ids : np.ndarray
        Array of process IDs (0=ttbar, 1=ggf, 2=dihiggs, 3=higgs_portal) for each event
    """
    filepath = Filepath()

    TRACK_PARAMS = ['d0', 'z0', 'phi', 'theta', 'qop']

    dir_map = {
        0: filepath.ttbar_base_tracks_dir,
        1: filepath.ggf_base_tracks_dir,
        2: filepath.dihiggs_base_tracks_dir,
        3: filepath.higgs_portal_base_tracks_dir,
    }

    # Convert num_events to array if needed
    if isinstance(num_events, int):
        events = np.arange(num_events, dtype=np.int32)
    else:
        events = np.asarray(list(num_events), dtype=np.int32)

    all_X   = []
    all_ids = []

    for event_id, tracks_dir in dir_map.items():
        tracks_df = utils_tracks.read_events_tracks(tracks_dir, events)

        for eid, group in tracks_df.groupby('event_id'):
            params = group[TRACK_PARAMS].values  # shape [n_tracks, 5]
            all_X.append(torch.tensor(params, dtype=torch.float32))
            all_ids.append(event_id)

    all_ids = np.array(all_ids, dtype=np.int32)

    return DataToDataModule_1d(batch_size,all_X, all_ids)