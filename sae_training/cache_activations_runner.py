import math
import os

import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm

from sae_training.activations_store import ActivationsStore
from sae_training.config import CacheActivationsRunnerConfig
from sae_training.utils import shuffle_activations_pairwise


def cache_activations_runner(cfg: CacheActivationsRunnerConfig):
    model = HookedTransformer.from_pretrained(cfg.model_name)
    model.to(cfg.device)
    activations_store = ActivationsStore(cfg, model, create_dataloader=False)
    
    # if the activations directory exists and has files in it, raise an exception
    if os.path.exists(activations_store.cfg.cached_activations_path):
        if len(os.listdir(activations_store.cfg.cached_activations_path)) > 0:
            raise Exception(f"Activations directory ({activations_store.cfg.cached_activations_path}) is not empty. Please delete it or specify a different path. Exiting the script to prevent accidental deletion of files.")
    else:
        os.makedirs(activations_store.cfg.cached_activations_path)
    
    print(f"Started caching {cfg.total_training_tokens} activations")
    tokens_per_buffer = cfg.store_batch_size * cfg.context_size * cfg.n_batches_in_buffer
    n_buffers = math.ceil(cfg.total_training_tokens / tokens_per_buffer)
    for i in tqdm(range(n_buffers), desc="Caching activations"):
        buffer = activations_store.get_buffer(cfg.n_batches_in_buffer)
        torch.save(buffer, f"{activations_store.cfg.cached_activations_path}/{i}.pt")
        del buffer
        
        if i % cfg.shuffle_every_n_buffers == 0 and i > 0:
            # Shuffle the buffers on disk
            
            # Do random pairwise shuffling between the last shuffle_every_n_buffers buffers
            for _ in range(cfg.n_shuffles_with_last_section):
                shuffle_activations_pairwise(activations_store.cfg.cached_activations_path,
                                             buffer_idx_range=(i - cfg.shuffle_every_n_buffers, i))
            
            # Do more random pairwise shuffling between all the buffers
            for _ in range(cfg.n_shuffles_in_entire_dir):
                shuffle_activations_pairwise(activations_store.cfg.cached_activations_path,
                                             buffer_idx_range=(0, i))
                
    # More final shuffling (mostly in case we didn't end on an i divisible by shuffle_every_n_buffers)
    for _ in tqdm(range(cfg.n_shuffles_final), desc="Final shuffling"):
        shuffle_activations_pairwise(activations_store.cfg.cached_activations_path,
                                     buffer_idx_range=(0, n_buffers))
