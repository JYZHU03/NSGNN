import logging
import time
from functools import partial
import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loader


from nsgnn.loader.split_generator import (prepare_splits, set_dataset_splits)
from nsgnn.transform.network_measure_stats import compute_Netsci_stats
from nsgnn.transform.task_preprocessing import task_specific_preprocessing
from nsgnn.transform.transforms import (pre_transform_in_memory)
from nsgnn.loader import power_grid_data
import os
from pathlib import Path


def log_loaded_dataset(dataset, format, name):
    logging.info(f"[*] Loaded dataset '{name}' from '{format}':")
    logging.info(f"  {dataset.data}")
    logging.info(f"  undirected: {dataset[0].is_undirected()}")
    logging.info(f"  num graphs: {len(dataset)}")

    total_num_nodes = 0
    if hasattr(dataset.data, 'num_nodes'):
        total_num_nodes = dataset.data.num_nodes
    elif hasattr(dataset.data, 'x'):
        total_num_nodes = dataset.data.x.size(0)
    logging.info(f"  avg num_nodes/graph: "
                 f"{total_num_nodes // len(dataset)}")
    logging.info(f"  num node features: {dataset.num_node_features}")
    logging.info(f"  num edge features: {dataset.num_edge_features}")
    if hasattr(dataset, 'num_tasks'):
        logging.info(f"  num tasks: {dataset.num_tasks}")

    if hasattr(dataset.data, 'y') and dataset.data.y is not None:
        if isinstance(dataset.data.y, list):
            # A special case for ogbg-code2 dataset.
            logging.info(f"  num classes: n/a")
        elif dataset.data.y.numel() == dataset.data.y.size(0) and \
                torch.is_floating_point(dataset.data.y):
            logging.info(f"  num classes: (appears to be a regression task)")
        else:
            logging.info(f"  num classes: {dataset.num_classes}")
    elif hasattr(dataset.data, 'train_edge_label') or hasattr(dataset.data, 'edge_label'):
        # Edge/link prediction task.
        if hasattr(dataset.data, 'train_edge_label'):
            labels = dataset.data.train_edge_label  # Transductive link task
        else:
            labels = dataset.data.edge_label  # Inductive link task
        if labels.numel() == labels.size(0) and \
                torch.is_floating_point(labels):
            logging.info(f"  num edge classes: (probably a regression task)")
        else:
            logging.info(f"  num edge classes: {len(torch.unique(labels))}")

def standardize_features(features):
    mean_val = torch.mean(features, dim=0, keepdim=True)
    std_val = torch.std(features, dim=0, keepdim=True)
    standardized_fea = (features - mean_val) / (std_val + 1e-6)
    return standardized_fea

@register_loader('custom_master_loader')
def load_dataset_master(format, name, dataset_dir):
    """
    Master loader that controls loading of datasets, overshadowing execution
    of any default GraphGym dataset loader.

    Custom transforms and dataset splitting is applied to each loaded dataset.

    Args:
        format: dataset format name that identifies Dataset class
        name: dataset name to select from the class identified by `format`
        dataset_dir: path where to store the processed dataset

    Returns:
        PyG dataset object with applied perturbation transforms and data splits
    """
    if format.startswith('PowerGrid-'):
        result_path = Path("training_run_directory")
        if os.path.isdir(result_path) == False:
            os.mkdir(result_path)

        dataset_name = cfg.dataset.name
        train_dataset = 'datasets/' + cfg.dataset.train_dataset
        test_dataset = 'datasets/' + cfg.dataset.test_dataset
        dataset_dir = 'datasets/' + dataset_name
        dataset = preformat_Powergrid(dataset_dir, train_dataset, test_dataset)

    else:
        raise ValueError(f"Unknown data format: {format}")

    pre_transform_in_memory(dataset, partial(task_specific_preprocessing, cfg=cfg))

    log_loaded_dataset(dataset, format, name)

    # Precompute necessary statistics for network measure encodings.
    pe_enabled_list = []
    for key, pecfg in cfg.items():
        if key.startswith('Netsci_') and pecfg.enable:
            pe_name = key.split('_', 1)[1]
            pe_enabled_list.append(pe_name)
            if hasattr(pecfg, 'kernel'):
                if pecfg.kernel.times_func:
                    pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
                logging.info(f"Parsed {pe_name} PE kernel times / steps: "
                             f"{pecfg.kernel.times}")
    if pe_enabled_list:
        start = time.perf_counter()
        logging.info(f"Precomputing Encoding statistics: "
                     f"{pe_enabled_list} for all graphs...")
        # Estimate directedness based on 10 graphs to save time.
        is_undirected = all(d.is_undirected() for d in dataset[:10])
        logging.info(f"  ...estimated to be undirected: {is_undirected}")
        pre_transform_in_memory(dataset,
                                partial(compute_Netsci_stats,
                                        pe_types=pe_enabled_list,
                                        is_undirected=is_undirected,
                                        cfg=cfg),
                                show_progress=True
                                )
        elapsed = time.perf_counter() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                  + f'{elapsed:.2f}'[-3:]
        logging.info(f"Done! Took {timestr}")


    if hasattr(dataset.data, 'Netsci_feature'):
        dataset.data.Netsci_feature = standardize_features(dataset.data.Netsci_feature) #normalize_feature

    if hasattr(dataset.data, 'node_measures'):
        dataset.data.node_measures = standardize_features(dataset.data.node_measures)

    if hasattr(dataset.data, 'pestat_RWSE'):
        dataset.data.pestat_RWSE = standardize_features(dataset.data.pestat_RWSE)

    if hasattr(dataset.data, 'Graphlets_feature'):
        dataset.data.Graphlets_feature = standardize_features(dataset.data.Graphlets_feature)

    # Set standard dataset train/val/test splits
    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')

    # Verify or generate dataset train/val/test splits
    prepare_splits(dataset)

    return dataset

def preformat_Powergrid(dataset_dir, train_dataset, test_dataset):
    """Load and preformat Power Grid datasets.

    Returns:
        PyG dataset object
    """

    dataset = join_dataset_splits(
        [power_grid_data.Powergrid(root=dataset_dir, split=split, train_dataset=train_dataset, test_dataset=test_dataset)
         for split in ['train', 'valid', 'test']]
    )
    return dataset

def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object.

    Args:
        datasets: list of 3 PyG datasets to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = [datasets[0].get(i) for i in range(n1)] + \
                [datasets[1].get(i) for i in range(n2)] + \
                [datasets[2].get(i) for i in range(n3)]

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)),
                  list(range(n1, n1 + n2)),
                  list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs

    return datasets[0]
