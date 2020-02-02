import torch.utils.data
from data.FontEffectsdataset import FontEffectsDataset

def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset

class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name opt['datasetname']
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        self.dataset = FontEffectsDataset(opt)
        print("dataset [%s] was created with %d images" % (opt['datasetname'],len(self.dataset)))   
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt['batch_size'],
            shuffle=True,
            num_workers=int(opt['num_threads']))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt['max_dataset_size'])

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt['batch_size'] >= self.opt['max_dataset_size']:
                break
            yield data