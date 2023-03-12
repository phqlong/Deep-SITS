import random
import omegaconf
import numpy as np
import xarray as xr
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


class SITSDataset(Dataset):
    def __init__(self, 
                 crop_yield_data, 
                 s1_path1, 
                 s1_path2, 
                 s2_path,
                 target_variable='RiceYield', 
                 s1_bands=['vv', 'vh', 'vv_by_vh', 'vv_add_vh', 'DOP', 'RVI'],
                 s2_bands=['NDVI', 'EVI', 'SAVI', 'NDWI', 'MSI', 'CARI']):
        
        self.s1_path1 = s1_path1
        self.s1_path2 = s1_path2
        self.s2_path = s2_path
        self.target_variable = target_variable
        self.s1_bands = omegaconf.OmegaConf.to_container(s1_bands)
        self.s2_bands = omegaconf.OmegaConf.to_container(s2_bands)

        self.data, self.targets = list(), list()
        self._load_data(crop_yield_data)

    def __getitem__(self, index):
        """
        Return sample (bands, time, lat, long), target
        """
        sample = self.data[index]
        target = self.targets[index]
        return sample, target

    def __len__(self):
        return len(self.targets)

    def _load_data(self, crop_yield_data):
        print("Load Dataset...")
        for idx, crop in tqdm(crop_yield_data.iterrows(), total=len(crop_yield_data)):
            # Load data from Sentinel 1 path 1 & path 2
            s1_data1, s1_time1, s1_mask1 = self._load_xds(crop, self.s1_path1, self.s1_bands, src='s1')
            s1_data2, s1_time2, s1_mask2 = self._load_xds(crop, self.s1_path2, self.s1_bands, src='s1')

            # Load data from Sentinel 2
            s2_data, s2_time, s2_mask = self._load_xds(crop, self.s2_path, self.s2_bands, src='s2')

            # Append data
            data = {'s1': {'data': s1_data1, 
                           'time': s1_time1, 
                           'mask': s1_mask1}, 
                    's2': {'data': s2_data, 
                           'time': s2_time, 
                           'mask': s2_mask}}
            self._append_data(crop, data)

            data = {'s1': {'data': s1_data2, 
                           'time': s1_time2, 
                           'mask': s1_mask2}, 
                    's2': {'data': s2_data, 
                           'time': s2_time, 
                           'mask': s2_mask}}
            self._append_data(crop, data)
        return

    def _append_data(self, crop, data):
        # Append data & target
        self.data.append(data)
        self.targets.append(crop[self.target_variable]/1000)
        return
    
    def _load_xds(self, crop, path, bands, src):
        xds = self._read_xds(crop, path)
        data = self._process_xds(xds, bands, src)
        return data
    
    def _read_xds(self, crop, path):
        # Open Dataset from disk
        group = f"{crop.Latitude}_{crop.Longitude}_{crop.Season}"
        xds = xr.open_dataset(path, group=group)
        return xds

    def _process_xds(self, xds, bands, src):
        if src=='s1':
            # Replace all invalid value to nan 
            xds = xds.where(xds!=-32768.0, np.nan)
            # Time Masking
            mask = np.ones((len(xds.time) + 1), dtype=bool)
        elif src=='s2':            
            # Filter timeseries with 23 timesteps before harvest date
            xds = xds.where(xds.time >= xds.coords['time'][-23], drop=True)
            # Cloud filtering
            xds, mask = self._cloud_filter(xds)

        # Select bands
        xds = xds[bands]

        # Convert xarray dataset to numpy array
        data = xds.to_array().to_numpy()

        # Replace nan values to its remain median value in the same bands & timestamp
        data = self._replace_nan(data)

        # Convert time to day of year
        time = xds.time.dt.dayofyear.to_numpy()

        # Convert numpy array to torch tensor
        data = torch.tensor(data, dtype=torch.float32)
        time = torch.tensor(time, dtype=torch.int32)
        mask = torch.tensor(mask, dtype=torch.bool)
        return data, time, mask

    def _replace_nan(self, arr):
        """
        Replace nan value to its remain median value in the same bands & timestamp
        arr: (d0, d1, d2, d3) : (bands, time, lat, long)
        """
        medians = np.nanmedian(arr, axis=(2, 3)).astype(np.float)

        # Create a mask for NaN values in each band at each time step
        mask = np.isnan(arr)

        # Convert the `(i, j)` indices in `pos` to a flat index
        flat_indices = np.ravel_multi_index(np.array(np.nonzero(mask)[:2]), medians.shape)

        # Extract the corresponding values from `means`
        arr[mask] = medians.ravel()[flat_indices]
        return arr

    def _cloud_filter(self, xds):
        """
        Cloud filtering:
        1. Mask out all pixels OK b SCL values
        2. Replace all not-OK values to nan. If ALL values in a pixel are nan, then drop it.
        Return new xarray dataset (B, T_new, H, W)
        """
        # Create a mask for no data, saturated data, clouds, cloud shadows, and water
        cloud_mask = \
            (xds.SCL != 0) & \
            (xds.SCL != 1) & \
            (xds.SCL != 3) & \
            (xds.SCL != 6) & \
            (xds.SCL != 8) & \
            (xds.SCL != 9) & \
            (xds.SCL != 10) & \
            (xds.SCL != 11)
        
        # Apply cloud mask ... NO Clouds, NO Cloud Shadows and NO Water pixels
        # All masked pixels are converted to "No Data" and stored as 16-bit integers
        filtered_time = xds.where(cloud_mask, drop=True).time
        filtered_xds = xds.where(cloud_mask, drop=False)

        # Replace all filtered time value to 0
        filtered_xds = filtered_xds.where(filtered_xds.time.isin(filtered_time), other=0)

        # Time Masking
        mask = xds.time.isin(filtered_time).to_numpy()
        
        # Add first valu True fo CLS
        mask = np.concatenate([np.array([True]), mask], axis=0)
        return filtered_xds, mask