import h5py
import numpy as np
import torch
import torch.utils.data as data


class H5PyData(data.Dataset):
    def __init__(self, data_path, clip_len, is_train=True, transform=None):
        """
        Args:
            args.data_path: path to the pickled data file with the
                following format:
                {
                    'train_data': [B, H, W, 3] np.uint8,
                    'train_idx': [B], np.int64 (start indexes for each video)
                    'test_data': [B', H, W, 3] np.uint8,
                    'test_idx': [B'], np.int64
                }
            args.sequence_length: length of extracted video sequences
        """
        self.split = 'train' if is_train else 'test'
        self.transform = transform
        
        self.data_path = data_path
        self.sequence_length = clip_len
        self.data = h5py.File(self.data_path, 'r')
        self._images = self.data[f'{self.split}_data']
        self._idxs = self.data[f'{self.split}_idx']

        ep_ends = np.concatenate((self._idx[1:], [len(self._images)]))
        self.ep_lens = ep_ends - self._idx
        self.seq_per_ep = self.ep_lens - self.sequence_length + 1
        assert np.all(self.seq_per_ep > 0)
        self.size = self.seq_per_ep.sum()

    def __getstate__(self):
        state = self.__dict__
        state['data'].close()
        state['data'] = None
        state['_images'] = None
        
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        self.data = h5py.File(self.data_path, 'r')
        self._images = self.data[f'{self.split}_data']

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        i = 0
        while idx > self.seq_per_ep[i] - 1:
            idx -= self.seq_per_ep[i]
            i += 1
        start = np.sum(self.ep_lens[:i]) + idx
        end = start + self.sequence_length
        
        video = torch.tensor(self._images[start:end])
        assert video.shape[0] == self.sequence_length

        if self.transform is not None:
            video = self.transform(video)
        
        return video, torch.tensor(0), torch.tensor(0)