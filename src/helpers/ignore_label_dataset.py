import torch.utils.data


class IgnoreLabelDataset(torch.utils.data.Dataset):
    """
    Wraps a dataset and returns only data (instead of tupels containing data and labels)
    """

    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)
