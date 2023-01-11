import numpy as np

## Data generator
def get_batch_data(X, Y, batch_size):
    size = X.shape[0]
    i = 0
    while i < size:
        inext = i + batch_size
        if inext > size: inext = size
        yield X[i:inext], Y[i:inext]
        i = inext

## A function for the Pytorch dataloader that will return data as NumPy arrays
def numpy_collate_fn(batch):
    inputs, labels = zip(*batch)
    ## NB. Make sure each element if numpy tensor first
    ## Make sure channel is moved from first axis to last, if not already
    inputs = np.moveaxis(np.stack(inputs), 1, -1)
    labels = np.moveaxis(np.stack(labels), 1, -1)
    return inputs, labels
