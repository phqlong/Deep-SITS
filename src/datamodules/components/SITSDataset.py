import random
import numpy as np
import xarray as xr
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


def replace_nanmean(arr):
    """
    Replace nan value in dim 1 to its remain mean value 
    arr: (d0, d1, d2, d3) : (bands, time, lat, long)
    """
    means = np.nanmean(arr, axis=(2, 3)).astype(np.float)

    # Create a mask for NaN values in each band at each time step
    mask = np.isnan(arr)

    # Convert the `(i, j)` indices in `pos` to a flat index
    flat_indices = np.ravel_multi_index(np.array(np.nonzero(mask)[:2]), means.shape)

    # Extract the corresponding values from `means`
    arr[mask] = means.ravel()[flat_indices]
    return arr


def read_xds(crop, path):
    # Open Dataset from disk
    group = f"{crop.Latitude}_{crop.Longitude}_{crop.Season}"
    xds = xr.open_dataset(path, group=group)
    return xds


def preprocess_xds(xds, bands=["vv", "vh", "vv_by_vh", "vv_add_vh", "DOP", "RVI"]):
    # Check type bands
    assert type(bands) == list
    assert type(bands[0]) == str

    # Select bands
    xds = xds[bands]

    # Transpose from (bands, time, lat, long) to (time, bands, lat, long) (No need)
    # data = cleaned_xds.to_array().transpose('time', 'variable', 'latitude', 'longitude').to_numpy()
    data = xds.to_array().to_numpy()
    
    # Replace nan values
    # data = np.nan_to_num(data, nan=0.0)
    data = replace_nanmean(data)
    
    # Replace all invalid value to 0 
    mask = data == -32768.0
    data[mask] = np.median(data, axis=(0,1,2,3))

    data = torch.tensor(data, dtype=torch.float32)
    return data


class SITSDataset(Dataset):
    def __init__(self, crop_yield_data, path1, path2, train=True, target_variable='RiceYield', bands=["vv", "vh", "vv_by_vh", "vv_add_vh", "DOP", "RVI"]):
        self.train = train
        self.bands = bands
        self.data, self.targets = self._load_data(crop_yield_data, path1, path2, target_variable, bands)

    def __getitem__(self, index):
        """
        Return sample (bands, time, lat, long), target
        """
        sample = self.data[index]
        target = self.targets[index]
        return sample, target

    def __len__(self):
        return len(self.targets)

    def _load_data(self, crop_yield_data, path1, path2, target_variable, bands):
        print("Load Dataset...")
        data_list = []
        target_list = []
        for idx, crop in tqdm(crop_yield_data.iterrows(), total=len(crop_yield_data)):
            # Load data from path 1
            xds1 = read_xds(crop, path1)
            data1 = preprocess_xds(xds1, bands)
            data_list.append(data1)
            target_list.append(crop[target_variable]/1000)

            if self.train:
                # Load data from path 2
                xds2 = read_xds(crop, path2)
                data2 = preprocess_xds(xds2, bands)
                data_list.append(data2)
                target_list.append(crop[target_variable]/1000)

        return data_list, target_list
