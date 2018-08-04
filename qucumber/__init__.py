__version__ = "0.2.0"


def _warn_on_missing_gpu(gpu):
    import warnings
    import torch

    if gpu and not torch.cuda.is_available():
        warnings.warn("Could not find GPU: will continue with CPU.",
                      ResourceWarning)

def set_random_seed(seed, cpu=True, gpu=False, quiet=False):
    import warnings
    import torch

    if gpu and torch.cuda.is_available():
        if not quiet:
            warnings.warn("GPU random seeds are not completely deterministic. "
                          "Proceed with caution.")
        torch.cuda.manual_seed(seed)

    if cpu:
        torch.manual_seed(seed)
