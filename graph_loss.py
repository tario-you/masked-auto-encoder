import matplotlib.pyplot as plt
import re

# Extracted loss data
log_data = """
(mae_env) tarioyou ~/masked-auto-encoder % python "/Users/tarioyou/masked-auto-encoder/inference.py"
Loaded pretrained model weights.
Reconstructed image saved to data/img_reconstructed.png
(mae_env) tarioyou ~/masked-auto-encoder % python "/Users/tarioyou/masked-auto-encoder/mae.py"
Loaded CIFAR-10 training data: (50000, 3, 32, 32)
Checkpoint loaded. Resuming from epoch 1
Epochs:   0%|                                                                                  | 0/9 [00:00<?, ?it/s/Users/tarioyou/masked-auto-encoder/mae.py:188: UserWarning: MPS: no support for int64 min/max ops, casting it to int32 (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/operations/Sort.mm:39.)
  ids_restore = torch.argsort(ids_shuffle, dim=1)
Training:   0%|                                                                 | 0/391 [01:09<?, ?it/s, loss=0.0401]
Epoch 2/10 - Average Loss: 0.0449                                               | 0/391 [00:49<?, ?it/s, loss=0.0401]
Checkpoint saved at epoch 2
Training:   0%|                                                                 | 0/391 [01:00<?, ?it/s, loss=0.0392]
Epoch 3/10 - Average Loss: 0.0397                                               | 0/391 [00:40<?, ?it/s, loss=0.0392]
Checkpoint saved at epoch 3
Epochs:  22%|████████████████▍                                                         | 2/9 [02:22<08:14, 70.69s/it^Zraining:   0%|                                                                 | 0/391 [00:20<?, ?it/s, loss=0.0367]
zsh: suspended  python "/Users/tarioyou/masked-auto-encoder/mae.py"
(mae_env) tarioyou ~/masked-auto-encoder % python "/Users/tarioyou/masked-auto-encoder/mae.py"
Loaded CIFAR-10 training data: (50000, 3, 32, 32)
Checkpoint loaded. Resuming from epoch 3
Epochs:   0%|                                                                                  | 0/7 [00:00<?, ?it/s/Users/tarioyou/masked-auto-encoder/mae.py:188: UserWarning: MPS: no support for int64 min/max ops, casting it to int32 (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/operations/Sort.mm:39.)
  ids_restore = torch.argsort(ids_shuffle, dim=1)
Training: 100%|███████████████████████████████████████████████████████| 391/391 [01:08<00:00,  5.70it/s, loss=0.0348]
Epoch 4/10 - Average Loss: 0.0379█████████████████████████████████████| 391/391 [00:48<00:00,  6.26it/s, loss=0.0348]
Checkpoint saved at epoch 4
Training: 100%|███████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.44it/s, loss=0.0330]
Epoch 5/10 - Average Loss: 0.0354█████████████████████████████████████| 391/391 [01:00<00:00, 10.86it/s, loss=0.0330]
Checkpoint saved at epoch 5
Training: 100%|███████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.43it/s, loss=0.0348]
Epoch 6/10 - Average Loss: 0.0346█████████████████████████████████████| 391/391 [00:53<00:00, 10.95it/s, loss=0.0348]
Checkpoint saved at epoch 6
Epochs:  43%|███████████████████████████████▋                                          | 3/7 [03:30<04:36, 69.03s/it]^CTraceback (most recent call last):
  File "/Users/tarioyou/Library/Python/3.10/lib/python/site-packages/numpy/core/__init__.py", line 24, in <module>
    from . import multiarray
  File "/Users/tarioyou/Library/Python/3.10/lib/python/site-packages/numpy/core/multiarray.py", line 10, in <module>
    from . import overrides
  File "/Users/tarioyou/Library/Python/3.10/lib/python/site-packages/numpy/core/overrides.py", line 8, in <module>
    from numpy.core._multiarray_umath import (
ImportError: PyCapsule_Import could not import module "datetime"

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/tarioyou/Library/Python/3.10/lib/python/site-packages/numpy/__init__.py", line 130, in <module>
    from numpy.__config__ import show as show_config
  File "/Users/tarioyou/Library/Python/3.10/lib/python/site-packages/numpy/__config__.py", line 4, in <module>
    from numpy.core._multiarray_umath import (
Epochs:  43%|███████████████████████████████▋                                          | 3/7 [03:33<04:44, 71.02s/it]
Traceback (most recent call last):
  File "/Users/tarioyou/masked-auto-encoder/mae.py", line 357, in <module>
    avg_loss = train_mae(model, train_loader, optimizer, device)
  File "/Users/tarioyou/masked-auto-encoder/mae.py", line 262, in train_mae
    progress_bar = tqdm(enumerate(dataloader),
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 441, in __iter__
    return self._get_iterator()
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 388, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1042, in __init__
    w.start()
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/multiprocessing/context.py", line 224, in _Popen
ary/Python/3.10/lib/python/site-packages/numpy/core/__init__.py", line 50, in <module>
    raise ImportError(msg)
ImportError: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python3.10 from "/usr/local/bin/python"
  * The NumPy version is: "1.26.4"

and make sure that they are the versions you expect.
Please carefully study the documentation linked above for further help.

Original error was: PyCapsule_Import could not import module "datetime"


The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<string>", line 1, in <module>
    return _default_context.get_context().Process._Popen(process_obj)
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/multiprocessing/context.py", line 284, in _Popen
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/multiprocessing/spawn.py", line 116, in spawn_main
    exitcode = _main(fd, parent_sentinel)
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/multiprocessing/spawn.py", line 125, in _main
    return Popen(process_obj)
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    prepare(preparation_data)
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/multiprocessing/s    super().__init__(process_obj)
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/multiprocessing/popen_fork.py", line 19, in __init__
pawn.py", line 236, in prepare
    _fixup_main_from_path(data['init_main_from_path'])
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/multiprocessing/spawn.py", line 287, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/runpy.py", line 269, in run_path
    return _run_module_code(code, init_globals, run_name,
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/runpy.py", line 96, in _run_module_code
    _run_code(code, mod_globals, init_globals,
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/Users/tarioyou/masked-auto-encoder/mae.py", line 3, in <module>
    import numpy as np
  File "/Users/tarioyou/L    self._launch(process_obj)
  File "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/multiprocessing/popen_spawn_posix.py", line 62, in _launch
    f.write(fp.getbuffer())
KeyboardInterrupt
ibrary/Python/3.10/lib/python/site-packages/numpy/__init__.py", line 135, in <module>
    raise ImportError(msg) from e
ImportError: Error importing numpy: you should not try to import numpy from
        its source directory; please exit the numpy source tree, and relaunch
        your python interpreter from there.

(mae_env) tarioyou ~/masked-auto-encoder % python "/Users/tarioyou/masked-auto-encoder/mae.py"
Loaded CIFAR-10 training data: (50000, 3, 32, 32)
Checkpoint loaded. Resuming from epoch 6
Epochs:   0%|                                                                                  | 0/4 [00:00<?, ?it/s^Zraining:  29%|███████████████▉                                       | 113/391 [00:43<01:31,  3.02it/s, loss=0.0360]
zsh: suspended  python "/Users/tarioyou/masked-auto-encoder/mae.py"
(mae_env) tarioyou ~/masked-auto-encoder % python "/Users/tarioyou/masked-auto-encoder/mae.py"
Loaded CIFAR-10 training data: (50000, 3, 32, 32)
Checkpoint loaded. Resuming from epoch 6
Epochs:   0%|                                                                                  | 0/4 [00:00<?, ?it/s/Users/tarioyou/masked-auto-encoder/mae.py:188: UserWarning: MPS: no support for int64 min/max ops, casting it to int32 (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/operations/Sort.mm:39.)
  ids_restore = torch.argsort(ids_shuffle, dim=1)
Training: 100%|███████████████████████████████████████████████████████| 391/391 [01:07<00:00,  5.76it/s, loss=0.0362]
Epoch 7/10 - Average Loss: 0.0340█████████████████████████████████████| 391/391 [00:47<00:00,  6.98it/s, loss=0.0362]
Checkpoint saved at epoch 7
Training: 100%|███████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.41it/s, loss=0.0328]
Epoch 8/10 - Average Loss: 0.0330█████████████████████████████████████| 391/391 [00:51<00:00, 11.17it/s, loss=0.0328]
Checkpoint saved at epoch 8
Training: 100%|███████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.45it/s, loss=0.0325]
Epoch 9/10 - Average Loss: 0.0321████████████████████████████████████▊| 390/391 [00:40<00:00,  9.40it/s, loss=0.0325]
Checkpoint saved at epoch 9
Training: 100%|███████████████████████████████████████████████████████| 391/391 [01:01<00:00,  6.35it/s, loss=0.0267]
Epoch 10/10 - Average Loss: 0.0310████████████████████████████████████| 391/391 [00:57<00:00, 11.01it/s, loss=0.0267]
Checkpoint saved at epoch 10
Epochs: 100%|██████████████████████████████████████████████████████████████████████████| 4/4 [04:36<00:00, 69.25s/it]
(mae_env) tarioyou ~/masked-auto-encoder % python "/Users/tarioyou/masked-auto-encoder/inference.py"
Loaded pretrained model weights.
Reconstructed image saved to data/img_reconstructed.png
(mae_env) tarioyou ~/masked-auto-encoder % python "/Users/tarioyou/masked-auto-encoder/inference.py"
Loaded pretrained model weights.
Reconstructed image saved to data/img_reconstructed.png
(mae_env) tarioyou ~/masked-auto-encoder % python "/Users/tarioyou/masked-auto-encoder/mae.py"
Loaded CIFAR-10 training data: (50000, 3, 32, 32)
Checkpoint loaded. Resuming from epoch 10
Epochs:   0%|                                                                       | 0/90 [00:00<?, ?it/s/Users/tarioyou/masked-auto-encoder/mae.py:188: UserWarning: MPS: no support for int64 min/max ops, casting it to int32 (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/operations/Sort.mm:39.)
  ids_restore = torch.argsort(ids_shuffle, dim=1)
Training: 100%|█████████████████████████████████████████████| 391/391 [01:07<00:00,  5.78it/s, loss=0.0299]
Epoch 11/100 - Average Loss: 0.0303█████████████████████████| 391/391 [01:05<00:00, 10.38it/s, loss=0.0299]
Checkpoint saved at epoch 11
Training: 100%|█████████████████████████████████████████████| 391/391 [01:03<00:00,  6.17it/s, loss=0.0283]
Epoch 12/100 - Average Loss: 0.0299████████████████████████▉| 390/391 [00:43<00:00,  9.15it/s, loss=0.0283]
Checkpoint saved at epoch 12
Training: 100%|█████████████████████████████████████████████| 391/391 [01:02<00:00,  6.24it/s, loss=0.0292]
Epoch 13/100 - Average Loss: 0.0295█████████████████████████| 391/391 [01:01<00:00, 11.55it/s, loss=0.0292]
Checkpoint saved at epoch 13
Training: 100%|█████████████████████████████████████████████| 391/391 [01:00<00:00,  6.47it/s, loss=0.0245]
Epoch 14/100 - Average Loss: 0.0290█████████████████████████| 391/391 [00:52<00:00, 11.08it/s, loss=0.0245]
Checkpoint saved at epoch 14
Training: 100%|█████████████████████████████████████████████| 391/391 [01:03<00:00,  6.13it/s, loss=0.0300]
Epoch 15/100 - Average Loss: 0.0283█████████████████████████| 391/391 [00:56<00:00, 10.22it/s, loss=0.0300]
Checkpoint saved at epoch 15
Training: 100%|█████████████████████████████████████████████| 391/391 [01:03<00:00,  6.15it/s, loss=0.0290]
Epoch 16/100 - Average Loss: 0.0277█████████████████████████| 391/391 [00:55<00:00, 10.09it/s, loss=0.0290]
Checkpoint saved at epoch 16
Training: 100%|█████████████████████████████████████████████| 391/391 [01:16<00:00,  5.08it/s, loss=0.0266]
Epoch 17/100 - Average Loss: 0.0269█████████████████████████| 391/391 [01:15<00:00, 10.91it/s, loss=0.0266]
Checkpoint saved at epoch 17
Training: 100%|█████████████████████████████████████████████| 391/391 [01:02<00:00,  6.21it/s, loss=0.0259]
Epoch 18/100 - Average Loss: 0.0261█████████████████████████| 391/391 [01:02<00:00, 11.32it/s, loss=0.0259]
Checkpoint saved at epoch 18
Training: 100%|█████████████████████████████████████████████| 391/391 [01:00<00:00,  6.46it/s, loss=0.0240]
Epoch 19/100 - Average Loss: 0.0253█████████████████████████| 391/391 [00:53<00:00, 11.14it/s, loss=0.0240]
Checkpoint saved at epoch 19
Training: 100%|█████████████████████████████████████████████| 391/391 [01:00<00:00,  6.47it/s, loss=0.0263]
Epoch 20/100 - Average Loss: 0.0246█████████████████████████| 391/391 [00:57<00:00, 11.21it/s, loss=0.0263]
Checkpoint saved at epoch 20
Training: 100%|█████████████████████████████████████████████| 391/391 [01:00<00:00,  6.43it/s, loss=0.0232]
Epoch 21/100 - Average Loss: 0.0240█████████████████████████| 391/391 [01:00<00:00, 10.63it/s, loss=0.0232]
Checkpoint saved at epoch 21
Training: 100%|█████████████████████████████████████████████| 391/391 [01:00<00:00,  6.43it/s, loss=0.0238]
Epoch 22/100 - Average Loss: 0.0236█████████████████████████| 391/391 [00:53<00:00, 11.16it/s, loss=0.0238]
Checkpoint saved at epoch 22
Training: 100%|█████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0246]
Epoch 23/100 - Average Loss: 0.0233█████████████████████████| 391/391 [00:56<00:00, 11.37it/s, loss=0.0246]
Checkpoint saved at epoch 23
Training: 100%|█████████████████████████████████████████████| 391/391 [01:00<00:00,  6.49it/s, loss=0.0248]
Epoch 24/100 - Average Loss: 0.0230█████████████████████████| 391/391 [00:51<00:00, 11.35it/s, loss=0.0248]
Checkpoint saved at epoch 24
Training: 100%|█████████████████████████████████████████████| 391/391 [01:00<00:00,  6.42it/s, loss=0.0238]
Epoch 25/100 - Average Loss: 0.0228█████████████████████████| 391/391 [00:54<00:00, 11.23it/s, loss=0.0238]
Checkpoint saved at epoch 25
Training: 100%|█████████████████████████████████████████████| 391/391 [01:01<00:00,  6.35it/s, loss=0.0230]
Epoch 26/100 - Average Loss: 0.0226█████████████████████████| 391/391 [00:57<00:00, 10.15it/s, loss=0.0230]
Checkpoint saved at epoch 26
Training: 100%|█████████████████████████████████████████████| 391/391 [01:02<00:00,  6.30it/s, loss=0.0193]
Epoch 27/100 - Average Loss: 0.0224█████████████████████████| 391/391 [00:59<00:00, 10.94it/s, loss=0.0193]
Checkpoint saved at epoch 27
Training: 100%|█████████████████████████████████████████████| 391/391 [01:00<00:00,  6.47it/s, loss=0.0219]
Epoch 28/100 - Average Loss: 0.0223█████████████████████████| 391/391 [00:51<00:00, 11.30it/s, loss=0.0219]
Checkpoint saved at epoch 28
Training: 100%|█████████████████████████████████████████████| 391/391 [00:59<00:00,  6.54it/s, loss=0.0227]
Epoch 29/100 - Average Loss: 0.0221█████████████████████████| 391/391 [00:54<00:00, 11.44it/s, loss=0.0227]
Checkpoint saved at epoch 29
Training: 100%|█████████████████████████████████████████████| 391/391 [01:00<00:00,  6.51it/s, loss=0.0214]
Epoch 30/100 - Average Loss: 0.0220█████████████████████████| 391/391 [00:58<00:00, 11.26it/s, loss=0.0214]
Checkpoint saved at epoch 30
Training: 100%|█████████████████████████████████████████████| 391/391 [01:00<00:00,  6.51it/s, loss=0.0243]
Epoch 31/100 - Average Loss: 0.0219█████████████████████████| 391/391 [00:52<00:00, 11.04it/s, loss=0.0243]
Checkpoint saved at epoch 31
Training: 100%|█████████████████████████████████████████████| 391/391 [01:07<00:00,  5.83it/s, loss=0.0211]
Epoch 32/100 - Average Loss: 0.0218█████████████████████████| 391/391 [01:06<00:00, 11.18it/s, loss=0.0211]
Checkpoint saved at epoch 32
Training: 100%|█████████████████████████████████████████████| 391/391 [01:01<00:00,  6.39it/s, loss=0.0218]
Epoch 33/100 - Average Loss: 0.0217█████████████████████████| 391/391 [00:53<00:00, 10.79it/s, loss=0.0218]
Checkpoint saved at epoch 33
Training: 100%|█████████████████████████████████████████████| 391/391 [01:01<00:00,  6.37it/s, loss=0.0201]
Epoch 34/100 - Average Loss: 0.0216█████████████████████████| 391/391 [00:41<00:00,  9.07it/s, loss=0.0201]
Checkpoint saved at epoch 34
Training: 100%|█████████████████████████████████████████████| 391/391 [01:04<00:00,  6.11it/s, loss=0.0220]
Epoch 35/100 - Average Loss: 0.0214█████████████████████████| 391/391 [00:57<00:00, 11.56it/s, loss=0.0220]
Checkpoint saved at epoch 35
Training: 100%|█████████████████████████████████████████████| 391/391 [01:00<00:00,  6.45it/s, loss=0.0200]
Epoch 36/100 - Average Loss: 0.0214█████████████████████████| 391/391 [00:57<00:00, 10.82it/s, loss=0.0200]
Checkpoint saved at epoch 36
Training: 100%|█████████████████████████████████████████████| 391/391 [01:01<00:00,  6.36it/s, loss=0.0219]
Epoch 37/100 - Average Loss: 0.0213█████████████████████████| 391/391 [01:00<00:00, 11.55it/s, loss=0.0219]
Checkpoint saved at epoch 37
Training: 100%|█████████████████████████████████████████████| 391/391 [00:59<00:00,  6.53it/s, loss=0.0206]
Epoch 38/100 - Average Loss: 0.0212█████████████████████████| 391/391 [00:52<00:00, 11.29it/s, loss=0.0206]
Checkpoint saved at epoch 38
Training: 100%|█████████████████████████████████████████████| 391/391 [01:00<00:00,  6.47it/s, loss=0.0230]
Epoch 39/100 - Average Loss: 0.0211█████████████████████████| 391/391 [00:56<00:00, 11.16it/s, loss=0.0230]
Checkpoint saved at epoch 39
Training: 100%|█████████████████████████████████████████████| 391/391 [01:01<00:00,  6.38it/s, loss=0.0205]
Epoch 40/100 - Average Loss: 0.0210█████████████████████████| 391/391 [00:58<00:00, 10.49it/s, loss=0.0205]
Checkpoint saved at epoch 40
Epochs:  33%|████████████████████                                        | 30/90 [34:26<1:07:48, 67.82s/it^Zraining:  65%|█████████████████████████████▎               | 255/391 [00:29<00:12, 10.47it/s, loss=0.0207]
zsh: suspended  python "/Users/tarioyou/masked-auto-encoder/mae.py"
ered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/operations/Sort.mm:39.)
  ids_restore = torch.argsort(ids_shuffle, dim=1)
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:07<00:00,  5.83it/s, loss=0.0201]
Epoch 41/300 - Average Loss: 0.0209███████████████████████████████████████████| 391/391 [00:47<00:00,  6.58it/s, loss=0.0201]
Checkpoint saved at epoch 41
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0231]
Epoch 42/300 - Average Loss: 0.0209███████████████████████████████████████████| 391/391 [00:52<00:00, 11.66it/s, loss=0.0231]
Checkpoint saved at epoch 42
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0183]
Epoch 43/300 - Average Loss: 0.0208███████████████████████████████████████████| 391/391 [00:57<00:00, 11.68it/s, loss=0.0183]
Checkpoint saved at epoch 43
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.54it/s, loss=0.0205]
Epoch 44/300 - Average Loss: 0.0206███████████████████████████████████████████| 391/391 [00:52<00:00, 11.01it/s, loss=0.0205]
Checkpoint saved at epoch 44
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.51it/s, loss=0.0216]
Epoch 45/300 - Average Loss: 0.0207███████████████████████████████████████████| 391/391 [00:57<00:00, 11.33it/s, loss=0.0216]
Checkpoint saved at epoch 45
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0199]
Epoch 46/300 - Average Loss: 0.0205███████████████████████████████████████████| 391/391 [00:51<00:00, 11.69it/s, loss=0.0199]
Checkpoint saved at epoch 46
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.63it/s, loss=0.0198]
Epoch 47/300 - Average Loss: 0.0205███████████████████████████████████████████| 391/391 [00:55<00:00, 11.72it/s, loss=0.0198]
Checkpoint saved at epoch 47
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0202]
Epoch 48/300 - Average Loss: 0.0204███████████████████████████████████████████| 391/391 [00:51<00:00, 10.36it/s, loss=0.0202]
Checkpoint saved at epoch 48
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.59it/s, loss=0.0209]
Epoch 49/300 - Average Loss: 0.0203███████████████████████████████████████████| 391/391 [00:56<00:00, 10.32it/s, loss=0.0209]
Checkpoint saved at epoch 49
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:01<00:00,  6.40it/s, loss=0.0208]
Epoch 50/300 - Average Loss: 0.0203███████████████████████████████████████████| 391/391 [01:00<00:00, 10.21it/s, loss=0.0208]
Checkpoint saved at epoch 50
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0219]
Epoch 51/300 - Average Loss: 0.0202███████████████████████████████████████████| 391/391 [00:53<00:00, 11.61it/s, loss=0.0219]
Checkpoint saved at epoch 51
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.63it/s, loss=0.0181]
Epoch 52/300 - Average Loss: 0.0201███████████████████████████████████████████| 391/391 [00:58<00:00, 11.50it/s, loss=0.0181]
Checkpoint saved at epoch 52
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.58it/s, loss=0.0181]
Epoch 53/300 - Average Loss: 0.0201███████████████████████████████████████████| 391/391 [00:53<00:00, 11.49it/s, loss=0.0181]
Checkpoint saved at epoch 53
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0211]
Epoch 54/300 - Average Loss: 0.0200███████████████████████████████████████████| 391/391 [00:58<00:00, 11.31it/s, loss=0.0211]
Checkpoint saved at epoch 54
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.54it/s, loss=0.0195]
Epoch 55/300 - Average Loss: 0.0200██████████████████████████████████████████▊| 390/391 [00:39<00:00,  9.74it/s, loss=0.0195]
Checkpoint saved at epoch 55
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.59it/s, loss=0.0202]
Epoch 56/300 - Average Loss: 0.0199███████████████████████████████████████████| 391/391 [00:57<00:00, 11.35it/s, loss=0.0202]
Checkpoint saved at epoch 56
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.58it/s, loss=0.0191]
Epoch 57/300 - Average Loss: 0.0199███████████████████████████████████████████| 391/391 [00:52<00:00, 11.58it/s, loss=0.0191]
Checkpoint saved at epoch 57
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0197]
Epoch 58/300 - Average Loss: 0.0198███████████████████████████████████████████| 391/391 [00:56<00:00, 11.30it/s, loss=0.0197]
Checkpoint saved at epoch 58
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.58it/s, loss=0.0182]
Epoch 59/300 - Average Loss: 0.0197███████████████████████████████████████████| 391/391 [00:52<00:00, 11.64it/s, loss=0.0182]
Checkpoint saved at epoch 59
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0196]
Epoch 60/300 - Average Loss: 0.0198███████████████████████████████████████████| 391/391 [00:56<00:00, 11.33it/s, loss=0.0196]
Checkpoint saved at epoch 60
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.41it/s, loss=0.0226]
Epoch 61/300 - Average Loss: 0.0197███████████████████████████████████████████| 391/391 [00:51<00:00, 11.24it/s, loss=0.0226]
Checkpoint saved at epoch 61
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0213]
Epoch 62/300 - Average Loss: 0.0196███████████████████████████████████████████| 391/391 [00:54<00:00, 11.65it/s, loss=0.0213]
Checkpoint saved at epoch 62
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0181]
Epoch 63/300 - Average Loss: 0.0195███████████████████████████████████████████| 391/391 [00:49<00:00, 11.57it/s, loss=0.0181]
Checkpoint saved at epoch 63
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0190]
Epoch 64/300 - Average Loss: 0.0196███████████████████████████████████████████| 391/391 [00:54<00:00, 11.62it/s, loss=0.0190]
Checkpoint saved at epoch 64
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0213]
Epoch 65/300 - Average Loss: 0.0195███████████████████████████████████████████| 391/391 [00:49<00:00, 11.53it/s, loss=0.0213]
Checkpoint saved at epoch 65
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.44it/s, loss=0.0190]
Epoch 66/300 - Average Loss: 0.0194███████████████████████████████████████████| 391/391 [00:54<00:00, 10.90it/s, loss=0.0190]
Checkpoint saved at epoch 66
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0204]
Epoch 67/300 - Average Loss: 0.0193███████████████████████████████████████████| 391/391 [00:57<00:00, 11.36it/s, loss=0.0204]
Checkpoint saved at epoch 67
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0202]
Epoch 68/300 - Average Loss: 0.0193███████████████████████████████████████████| 391/391 [00:52<00:00, 11.55it/s, loss=0.0202]
Checkpoint saved at epoch 68
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0207]
Epoch 69/300 - Average Loss: 0.0193███████████████████████████████████████████| 391/391 [00:57<00:00, 11.32it/s, loss=0.0207]
Checkpoint saved at epoch 69
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0213]
Epoch 70/300 - Average Loss: 0.0193███████████████████████████████████████████| 391/391 [00:52<00:00, 11.66it/s, loss=0.0213]
Checkpoint saved at epoch 70
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0177]
Epoch 71/300 - Average Loss: 0.0193███████████████████████████████████████████| 391/391 [00:57<00:00, 10.92it/s, loss=0.0177]
Checkpoint saved at epoch 71
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.44it/s, loss=0.0188]
Epoch 72/300 - Average Loss: 0.0192███████████████████████████████████████████| 391/391 [00:52<00:00, 11.04it/s, loss=0.0188]
Checkpoint saved at epoch 72
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0178]
Epoch 73/300 - Average Loss: 0.0192███████████████████████████████████████████| 391/391 [00:55<00:00, 11.76it/s, loss=0.0178]
Checkpoint saved at epoch 73
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.64it/s, loss=0.0189]
Epoch 74/300 - Average Loss: 0.0192███████████████████████████████████████████| 391/391 [00:51<00:00, 11.65it/s, loss=0.0189]
Checkpoint saved at epoch 74
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0194]
Epoch 75/300 - Average Loss: 0.0191███████████████████████████████████████████| 391/391 [00:56<00:00, 11.31it/s, loss=0.0194]
Checkpoint saved at epoch 75
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0205]
Epoch 76/300 - Average Loss: 0.0191███████████████████████████████████████████| 391/391 [00:51<00:00, 11.63it/s, loss=0.0205]
Checkpoint saved at epoch 76
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.45it/s, loss=0.0169]
Epoch 77/300 - Average Loss: 0.0191███████████████████████████████████████████| 391/391 [00:56<00:00, 11.25it/s, loss=0.0169]
Checkpoint saved at epoch 77
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0205]
Epoch 78/300 - Average Loss: 0.0191███████████████████████████████████████████| 391/391 [00:49<00:00, 11.69it/s, loss=0.0205]
Checkpoint saved at epoch 78
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.57it/s, loss=0.0191]
Epoch 79/300 - Average Loss: 0.0190███████████████████████████████████████████| 391/391 [00:54<00:00, 11.55it/s, loss=0.0191]
Checkpoint saved at epoch 79
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.64it/s, loss=0.0200]
Epoch 80/300 - Average Loss: 0.0190███████████████████████████████████████████| 391/391 [00:49<00:00, 11.60it/s, loss=0.0200]
Checkpoint saved at epoch 80
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0194]
Epoch 81/300 - Average Loss: 0.0190███████████████████████████████████████████| 391/391 [00:54<00:00, 11.15it/s, loss=0.0194]
Checkpoint saved at epoch 81
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.59it/s, loss=0.0198]
Epoch 82/300 - Average Loss: 0.0189███████████████████████████████████████████| 391/391 [00:58<00:00, 11.63it/s, loss=0.0198]
Checkpoint saved at epoch 82
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:01<00:00,  6.39it/s, loss=0.0181]
Epoch 83/300 - Average Loss: 0.0189███████████████████████████████████████████| 391/391 [00:53<00:00, 11.67it/s, loss=0.0181]
Checkpoint saved at epoch 83
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0207]
Epoch 84/300 - Average Loss: 0.0188███████████████████████████████████████████| 391/391 [00:56<00:00, 11.58it/s, loss=0.0207]
Checkpoint saved at epoch 84
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0169]
Epoch 85/300 - Average Loss: 0.0189███████████████████████████████████████████| 391/391 [00:51<00:00, 11.45it/s, loss=0.0169]
Checkpoint saved at epoch 85
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0198]
Epoch 86/300 - Average Loss: 0.0188███████████████████████████████████████████| 391/391 [00:55<00:00, 11.60it/s, loss=0.0198]
Checkpoint saved at epoch 86
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0180]
Epoch 87/300 - Average Loss: 0.0188███████████████████████████████████████████| 391/391 [00:50<00:00, 10.34it/s, loss=0.0180]
Checkpoint saved at epoch 87
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.51it/s, loss=0.0192]
Epoch 88/300 - Average Loss: 0.0188███████████████████████████████████████████| 391/391 [00:55<00:00, 10.88it/s, loss=0.0192]
Checkpoint saved at epoch 88
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.63it/s, loss=0.0178]
Epoch 89/300 - Average Loss: 0.0188███████████████████████████████████████████| 391/391 [00:58<00:00, 11.58it/s, loss=0.0178]
Checkpoint saved at epoch 89
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.59it/s, loss=0.0174]
Epoch 90/300 - Average Loss: 0.0187███████████████████████████████████████████| 391/391 [00:53<00:00, 11.01it/s, loss=0.0174]
Checkpoint saved at epoch 90
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.63it/s, loss=0.0190]
Epoch 91/300 - Average Loss: 0.0187███████████████████████████████████████████| 391/391 [00:58<00:00, 11.71it/s, loss=0.0190]
Checkpoint saved at epoch 91
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0184]
Epoch 92/300 - Average Loss: 0.0187███████████████████████████████████████████| 391/391 [00:53<00:00, 11.70it/s, loss=0.0184]
Checkpoint saved at epoch 92
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0189]
Epoch 93/300 - Average Loss: 0.0187███████████████████████████████████████████| 391/391 [00:58<00:00,  9.51it/s, loss=0.0189]
Checkpoint saved at epoch 93
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.44it/s, loss=0.0182]
Epoch 94/300 - Average Loss: 0.0187███████████████████████████████████████████| 391/391 [00:53<00:00, 10.86it/s, loss=0.0182]
Checkpoint saved at epoch 94
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0182]
Epoch 95/300 - Average Loss: 0.0186███████████████████████████████████████████| 391/391 [00:56<00:00, 11.69it/s, loss=0.0182]
Checkpoint saved at epoch 95
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0210]
Epoch 96/300 - Average Loss: 0.0186███████████████████████████████████████████| 391/391 [00:51<00:00, 11.61it/s, loss=0.0210]
Checkpoint saved at epoch 96
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0194]
Epoch 97/300 - Average Loss: 0.0186███████████████████████████████████████████| 391/391 [00:56<00:00, 11.61it/s, loss=0.0194]
Checkpoint saved at epoch 97
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0198]
Epoch 98/300 - Average Loss: 0.0186███████████████████████████████████████████| 391/391 [00:51<00:00, 11.64it/s, loss=0.0198]
Checkpoint saved at epoch 98
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.48it/s, loss=0.0150]
Epoch 99/300 - Average Loss: 0.0186███████████████████████████████████████████| 391/391 [00:56<00:00, 10.74it/s, loss=0.0150]
Checkpoint saved at epoch 99
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0193]
Epoch 100/300 - Average Loss: 0.0185██████████████████████████████████████████| 391/391 [00:50<00:00, 11.36it/s, loss=0.0193]
Checkpoint saved at epoch 100
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.64it/s, loss=0.0195]
Epoch 101/300 - Average Loss: 0.0185██████████████████████████████████████████| 391/391 [00:55<00:00, 11.67it/s, loss=0.0195]
Checkpoint saved at epoch 101
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0156]
Epoch 102/300 - Average Loss: 0.0185██████████████████████████████████████████| 391/391 [00:50<00:00, 11.54it/s, loss=0.0156]
Checkpoint saved at epoch 102
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.58it/s, loss=0.0197]
Epoch 103/300 - Average Loss: 0.0185██████████████████████████████████████████| 391/391 [00:55<00:00, 11.48it/s, loss=0.0197]
Checkpoint saved at epoch 103
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.53it/s, loss=0.0188]
Epoch 104/300 - Average Loss: 0.0185██████████████████████████████████████████| 391/391 [00:50<00:00, 10.26it/s, loss=0.0188]
Checkpoint saved at epoch 104
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.50it/s, loss=0.0195]
Epoch 105/300 - Average Loss: 0.0184██████████████████████████████████████████| 391/391 [00:53<00:00, 11.64it/s, loss=0.0195]
Checkpoint saved at epoch 105
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0192]
Epoch 106/300 - Average Loss: 0.0184██████████████████████████████████████████| 391/391 [00:58<00:00, 11.22it/s, loss=0.0192]
Checkpoint saved at epoch 106
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0181]
Epoch 107/300 - Average Loss: 0.0184██████████████████████████████████████████| 391/391 [00:53<00:00, 11.59it/s, loss=0.0181]
Checkpoint saved at epoch 107
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.58it/s, loss=0.0184]
Epoch 108/300 - Average Loss: 0.0184██████████████████████████████████████████| 391/391 [00:58<00:00, 11.56it/s, loss=0.0184]
Checkpoint saved at epoch 108
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.59it/s, loss=0.0202]
Epoch 109/300 - Average Loss: 0.0184██████████████████████████████████████████| 391/391 [00:53<00:00, 11.61it/s, loss=0.0202]
Checkpoint saved at epoch 109
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.45it/s, loss=0.0172]
Epoch 110/300 - Average Loss: 0.0184██████████████████████████████████████████| 391/391 [00:57<00:00, 10.87it/s, loss=0.0172]
Checkpoint saved at epoch 110
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.64it/s, loss=0.0166]
Epoch 111/300 - Average Loss: 0.0183██████████████████████████████████████████| 391/391 [00:51<00:00, 11.58it/s, loss=0.0166]
Checkpoint saved at epoch 111
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0188]
Epoch 112/300 - Average Loss: 0.0183██████████████████████████████████████████| 391/391 [00:56<00:00, 11.58it/s, loss=0.0188]
Checkpoint saved at epoch 112
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0192]
Epoch 113/300 - Average Loss: 0.0183██████████████████████████████████████████| 391/391 [00:51<00:00, 11.59it/s, loss=0.0192]
Checkpoint saved at epoch 113
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.59it/s, loss=0.0168]
Epoch 114/300 - Average Loss: 0.0184██████████████████████████████████████████| 391/391 [00:56<00:00, 11.55it/s, loss=0.0168]
Checkpoint saved at epoch 114
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.64it/s, loss=0.0177]
Epoch 115/300 - Average Loss: 0.0182██████████████████████████████████████████| 391/391 [00:51<00:00, 11.25it/s, loss=0.0177]
Checkpoint saved at epoch 115
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.44it/s, loss=0.0186]
Epoch 116/300 - Average Loss: 0.0183██████████████████████████████████████████| 391/391 [00:56<00:00, 11.64it/s, loss=0.0186]
Checkpoint saved at epoch 116
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0199]
Epoch 117/300 - Average Loss: 0.0182██████████████████████████████████████████| 391/391 [00:49<00:00, 11.61it/s, loss=0.0199]
Checkpoint saved at epoch 117
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.57it/s, loss=0.0167]
Epoch 118/300 - Average Loss: 0.0182██████████████████████████████████████████| 391/391 [00:54<00:00, 11.24it/s, loss=0.0167]
Checkpoint saved at epoch 118
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0174]
Epoch 119/300 - Average Loss: 0.0182██████████████████████████████████████████| 391/391 [00:58<00:00, 11.55it/s, loss=0.0174]
Checkpoint saved at epoch 119
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.59it/s, loss=0.0182]
Epoch 120/300 - Average Loss: 0.0182██████████████████████████████████████████| 391/391 [00:53<00:00, 11.59it/s, loss=0.0182]
Checkpoint saved at epoch 120
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:02<00:00,  6.30it/s, loss=0.0183]
Epoch 121/300 - Average Loss: 0.0182██████████████████████████████████████████| 391/391 [00:58<00:00, 11.00it/s, loss=0.0183]
Checkpoint saved at epoch 121
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.64it/s, loss=0.0172]
Epoch 122/300 - Average Loss: 0.0182██████████████████████████████████████████| 391/391 [00:50<00:00, 11.70it/s, loss=0.0172]
Checkpoint saved at epoch 122
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0199]
Epoch 123/300 - Average Loss: 0.0182██████████████████████████████████████████| 391/391 [00:55<00:00, 10.72it/s, loss=0.0199]
Checkpoint saved at epoch 123
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.64it/s, loss=0.0181]
Epoch 124/300 - Average Loss: 0.0182██████████████████████████████████████████| 391/391 [00:50<00:00, 11.65it/s, loss=0.0181]
Checkpoint saved at epoch 124
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.58it/s, loss=0.0198]
Epoch 125/300 - Average Loss: 0.0181██████████████████████████████████████████| 391/391 [00:55<00:00, 11.55it/s, loss=0.0198]
Checkpoint saved at epoch 125
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.55it/s, loss=0.0208]
Epoch 126/300 - Average Loss: 0.0181██████████████████████████████████████████| 391/391 [00:50<00:00, 11.04it/s, loss=0.0208]
Checkpoint saved at epoch 126
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.51it/s, loss=0.0180]
Epoch 127/300 - Average Loss: 0.0181██████████████████████████████████████████| 391/391 [00:54<00:00, 11.45it/s, loss=0.0180]
Checkpoint saved at epoch 127
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.59it/s, loss=0.0161]
Epoch 128/300 - Average Loss: 0.0181██████████████████████████████████████████| 391/391 [00:58<00:00, 10.97it/s, loss=0.0161]
Checkpoint saved at epoch 128
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0164]
Epoch 129/300 - Average Loss: 0.0181██████████████████████████████████████████| 391/391 [00:53<00:00, 11.32it/s, loss=0.0164]
Checkpoint saved at epoch 129
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0161]
Epoch 130/300 - Average Loss: 0.0180██████████████████████████████████████████| 391/391 [00:57<00:00, 11.71it/s, loss=0.0161]
Checkpoint saved at epoch 130
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.63it/s, loss=0.0187]
Epoch 131/300 - Average Loss: 0.0180██████████████████████████████████████████| 391/391 [00:53<00:00, 11.59it/s, loss=0.0187]
Checkpoint saved at epoch 131
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:02<00:00,  6.29it/s, loss=0.0200]
Epoch 132/300 - Average Loss: 0.0181██████████████████████████████████████████| 391/391 [00:57<00:00, 10.76it/s, loss=0.0200]
Checkpoint saved at epoch 132
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.63it/s, loss=0.0187]
Epoch 133/300 - Average Loss: 0.0181██████████████████████████████████████████| 391/391 [00:49<00:00, 11.48it/s, loss=0.0187]
Checkpoint saved at epoch 133
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.58it/s, loss=0.0183]
Epoch 134/300 - Average Loss: 0.0180██████████████████████████████████████████| 391/391 [00:54<00:00, 11.27it/s, loss=0.0183]
Checkpoint saved at epoch 134
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.64it/s, loss=0.0156]
Epoch 135/300 - Average Loss: 0.0180██████████████████████████████████████████| 391/391 [00:58<00:00, 11.71it/s, loss=0.0156]
Checkpoint saved at epoch 135
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0162]
Epoch 136/300 - Average Loss: 0.0179██████████████████████████████████████████| 391/391 [00:53<00:00, 11.68it/s, loss=0.0162]
Checkpoint saved at epoch 136
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.53it/s, loss=0.0159]
Epoch 137/300 - Average Loss: 0.0180██████████████████████████████████████████| 391/391 [00:58<00:00, 11.29it/s, loss=0.0159]
Checkpoint saved at epoch 137
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.52it/s, loss=0.0169]
Epoch 138/300 - Average Loss: 0.0180██████████████████████████████████████████| 391/391 [00:52<00:00, 11.36it/s, loss=0.0169]
Checkpoint saved at epoch 138
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0187]
Epoch 139/300 - Average Loss: 0.0179██████████████████████████████████████████| 391/391 [00:56<00:00, 11.67it/s, loss=0.0187]
Checkpoint saved at epoch 139
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.53it/s, loss=0.0159]
Epoch 140/300 - Average Loss: 0.0180██████████████████████████████████████████| 391/391 [00:51<00:00, 10.91it/s, loss=0.0159]
Checkpoint saved at epoch 140
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0185]
Epoch 141/300 - Average Loss: 0.0180██████████████████████████████████████████| 391/391 [00:54<00:00, 11.70it/s, loss=0.0185]
Checkpoint saved at epoch 141
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.65it/s, loss=0.0184]
Epoch 142/300 - Average Loss: 0.0179██████████████████████████████████████████| 391/391 [00:49<00:00, 11.64it/s, loss=0.0184]
Checkpoint saved at epoch 142
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.42it/s, loss=0.0167]
Epoch 143/300 - Average Loss: 0.0179██████████████████████████████████████████| 391/391 [00:54<00:00, 11.00it/s, loss=0.0167]
Checkpoint saved at epoch 143
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0182]
Epoch 144/300 - Average Loss: 0.0179██████████████████████████████████████████| 391/391 [00:57<00:00, 11.57it/s, loss=0.0182]
Checkpoint saved at epoch 144
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0197]
Epoch 145/300 - Average Loss: 0.0179██████████████████████████████████████████| 391/391 [00:52<00:00, 11.44it/s, loss=0.0197]
Checkpoint saved at epoch 145
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.59it/s, loss=0.0175]
Epoch 146/300 - Average Loss: 0.0179██████████████████████████████████████████| 391/391 [00:57<00:00, 11.64it/s, loss=0.0175]
Checkpoint saved at epoch 146
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0178]
Epoch 147/300 - Average Loss: 0.0178██████████████████████████████████████████| 391/391 [00:52<00:00, 10.74it/s, loss=0.0178]
Checkpoint saved at epoch 147
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.42it/s, loss=0.0186]
Epoch 148/300 - Average Loss: 0.0179██████████████████████████████████████████| 391/391 [00:56<00:00, 10.27it/s, loss=0.0186]
Checkpoint saved at epoch 148
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.57it/s, loss=0.0173]
Epoch 149/300 - Average Loss: 0.0179██████████████████████████████████████████| 391/391 [00:49<00:00, 11.16it/s, loss=0.0173]
Checkpoint saved at epoch 149
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0156]
Epoch 150/300 - Average Loss: 0.0178██████████████████████████████████████████| 391/391 [00:54<00:00, 11.42it/s, loss=0.0156]
Checkpoint saved at epoch 150
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0179]
Epoch 151/300 - Average Loss: 0.0178██████████████████████████████████████████| 391/391 [00:49<00:00, 11.63it/s, loss=0.0179]
Checkpoint saved at epoch 151
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0178]
Epoch 152/300 - Average Loss: 0.0177██████████████████████████████████████████| 391/391 [00:54<00:00, 11.59it/s, loss=0.0178]
Checkpoint saved at epoch 152
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.63it/s, loss=0.0166]
Epoch 153/300 - Average Loss: 0.0178██████████████████████████████████████████| 391/391 [00:49<00:00, 11.37it/s, loss=0.0166]
Checkpoint saved at epoch 153
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:01<00:00,  6.40it/s, loss=0.0175]
Epoch 154/300 - Average Loss: 0.0178██████████████████████████████████████████| 391/391 [00:54<00:00, 11.39it/s, loss=0.0175]
Checkpoint saved at epoch 154
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.59it/s, loss=0.0181]
Epoch 155/300 - Average Loss: 0.0178██████████████████████████████████████████| 391/391 [00:57<00:00, 11.73it/s, loss=0.0181]
Checkpoint saved at epoch 155
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0196]
Epoch 156/300 - Average Loss: 0.0177██████████████████████████████████████████| 391/391 [00:52<00:00, 11.65it/s, loss=0.0196]
Checkpoint saved at epoch 156
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0164]
Epoch 157/300 - Average Loss: 0.0178██████████████████████████████████████████| 391/391 [00:57<00:00, 11.70it/s, loss=0.0164]
Checkpoint saved at epoch 157
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.64it/s, loss=0.0169]
Epoch 158/300 - Average Loss: 0.0177██████████████████████████████████████████| 391/391 [00:51<00:00, 11.62it/s, loss=0.0169]
Checkpoint saved at epoch 158
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.50it/s, loss=0.0176]
Epoch 159/300 - Average Loss: 0.0177██████████████████████████████████████████| 391/391 [00:56<00:00, 11.47it/s, loss=0.0176]
Checkpoint saved at epoch 159
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.58it/s, loss=0.0163]
Epoch 160/300 - Average Loss: 0.0177██████████████████████████████████████████| 391/391 [00:49<00:00, 11.69it/s, loss=0.0163]
Checkpoint saved at epoch 160
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.59it/s, loss=0.0159]
Epoch 161/300 - Average Loss: 0.0178██████████████████████████████████████████| 391/391 [00:54<00:00, 11.54it/s, loss=0.0159]
Checkpoint saved at epoch 161
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0170]
Epoch 162/300 - Average Loss: 0.0177██████████████████████████████████████████| 391/391 [00:49<00:00, 11.30it/s, loss=0.0170]
Checkpoint saved at epoch 162
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.63it/s, loss=0.0168]
Epoch 163/300 - Average Loss: 0.0177██████████████████████████████████████████| 391/391 [00:54<00:00, 11.66it/s, loss=0.0168]
Checkpoint saved at epoch 163
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0175]
Epoch 164/300 - Average Loss: 0.0177██████████████████████████████████████████| 391/391 [00:49<00:00, 10.92it/s, loss=0.0175]
Checkpoint saved at epoch 164
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.46it/s, loss=0.0187]
Epoch 165/300 - Average Loss: 0.0177██████████████████████████████████████████| 391/391 [00:53<00:00, 11.63it/s, loss=0.0187]
Checkpoint saved at epoch 165
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0196]
Epoch 166/300 - Average Loss: 0.0176██████████████████████████████████████████| 391/391 [00:57<00:00, 11.64it/s, loss=0.0196]
Checkpoint saved at epoch 166
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0177]
Epoch 167/300 - Average Loss: 0.0177██████████████████████████████████████████| 391/391 [00:52<00:00, 11.66it/s, loss=0.0177]
Checkpoint saved at epoch 167
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0189]
Epoch 168/300 - Average Loss: 0.0177██████████████████████████████████████████| 391/391 [00:57<00:00, 11.61it/s, loss=0.0189]
Checkpoint saved at epoch 168
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0196]
Epoch 169/300 - Average Loss: 0.0176██████████████████████████████████████████| 391/391 [00:52<00:00, 11.64it/s, loss=0.0196]
Checkpoint saved at epoch 169
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:01<00:00,  6.32it/s, loss=0.0195]
Epoch 170/300 - Average Loss: 0.0176█████████████████████████████████████████▊| 390/391 [00:41<00:00,  9.00it/s, loss=0.0195]
Checkpoint saved at epoch 170
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0176]
Epoch 171/300 - Average Loss: 0.0176██████████████████████████████████████████| 391/391 [00:58<00:00, 11.71it/s, loss=0.0176]
Checkpoint saved at epoch 171
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.65it/s, loss=0.0184]
Epoch 172/300 - Average Loss: 0.0176██████████████████████████████████████████| 391/391 [00:53<00:00, 11.12it/s, loss=0.0184]
Checkpoint saved at epoch 172
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.64it/s, loss=0.0184]
Epoch 173/300 - Average Loss: 0.0176██████████████████████████████████████████| 391/391 [00:58<00:00, 11.63it/s, loss=0.0184]
Checkpoint saved at epoch 173
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0183]
Epoch 174/300 - Average Loss: 0.0176██████████████████████████████████████████| 391/391 [00:53<00:00, 11.60it/s, loss=0.0183]
Checkpoint saved at epoch 174
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.58it/s, loss=0.0167]
Epoch 175/300 - Average Loss: 0.0175██████████████████████████████████████████| 391/391 [00:58<00:00, 11.60it/s, loss=0.0167]
Checkpoint saved at epoch 175
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.46it/s, loss=0.0190]
Epoch 176/300 - Average Loss: 0.0175██████████████████████████████████████████| 391/391 [00:52<00:00, 11.60it/s, loss=0.0190]
Checkpoint saved at epoch 176
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0185]
Epoch 177/300 - Average Loss: 0.0176██████████████████████████████████████████| 391/391 [00:56<00:00, 11.60it/s, loss=0.0185]
Checkpoint saved at epoch 177
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0174]
Epoch 178/300 - Average Loss: 0.0175██████████████████████████████████████████| 391/391 [00:51<00:00, 11.63it/s, loss=0.0174]
Checkpoint saved at epoch 178
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0156]
Epoch 179/300 - Average Loss: 0.0175██████████████████████████████████████████| 391/391 [00:55<00:00, 11.32it/s, loss=0.0156]
Checkpoint saved at epoch 179
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.64it/s, loss=0.0173]
Epoch 180/300 - Average Loss: 0.0175██████████████████████████████████████████| 391/391 [00:50<00:00, 11.66it/s, loss=0.0173]
Checkpoint saved at epoch 180
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:01<00:00,  6.36it/s, loss=0.0181]
Epoch 181/300 - Average Loss: 0.0175██████████████████████████████████████████| 391/391 [00:55<00:00, 10.73it/s, loss=0.0181]
Checkpoint saved at epoch 181
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.63it/s, loss=0.0163]
Epoch 182/300 - Average Loss: 0.0176██████████████████████████████████████████| 391/391 [00:58<00:00, 11.66it/s, loss=0.0163]
Checkpoint saved at epoch 182
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0179]
Epoch 183/300 - Average Loss: 0.0175██████████████████████████████████████████| 391/391 [00:53<00:00, 11.65it/s, loss=0.0179]
Checkpoint saved at epoch 183
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.63it/s, loss=0.0185]
Epoch 184/300 - Average Loss: 0.0175██████████████████████████████████████████| 391/391 [00:58<00:00, 11.69it/s, loss=0.0185]
Checkpoint saved at epoch 184
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0183]
Epoch 185/300 - Average Loss: 0.0175██████████████████████████████████████████| 391/391 [00:53<00:00, 11.24it/s, loss=0.0183]
Checkpoint saved at epoch 185
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.64it/s, loss=0.0170]
Epoch 186/300 - Average Loss: 0.0175██████████████████████████████████████████| 391/391 [00:58<00:00, 11.61it/s, loss=0.0170]
Checkpoint saved at epoch 186
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:01<00:00,  6.39it/s, loss=0.0180]
Epoch 187/300 - Average Loss: 0.0175██████████████████████████████████████████| 391/391 [00:52<00:00, 11.32it/s, loss=0.0180]
Checkpoint saved at epoch 187
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.63it/s, loss=0.0182]
Epoch 188/300 - Average Loss: 0.0174██████████████████████████████████████████| 391/391 [00:55<00:00, 11.63it/s, loss=0.0182]
Checkpoint saved at epoch 188
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.58it/s, loss=0.0168]
Epoch 189/300 - Average Loss: 0.0175██████████████████████████████████████████| 391/391 [00:50<00:00, 11.66it/s, loss=0.0168]
Checkpoint saved at epoch 189
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.59it/s, loss=0.0180]
Epoch 190/300 - Average Loss: 0.0175██████████████████████████████████████████| 391/391 [00:55<00:00, 11.00it/s, loss=0.0180]
Checkpoint saved at epoch 190
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0165]
Epoch 191/300 - Average Loss: 0.0175██████████████████████████████████████████| 391/391 [00:49<00:00, 11.62it/s, loss=0.0165]
Checkpoint saved at epoch 191
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.47it/s, loss=0.0189]
Epoch 192/300 - Average Loss: 0.0175██████████████████████████████████████████| 391/391 [00:54<00:00, 11.52it/s, loss=0.0189]
Checkpoint saved at epoch 192
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0172]
Epoch 193/300 - Average Loss: 0.0174██████████████████████████████████████████| 391/391 [00:58<00:00, 11.52it/s, loss=0.0172]
Checkpoint saved at epoch 193
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0169]
Epoch 194/300 - Average Loss: 0.0174██████████████████████████████████████████| 391/391 [00:53<00:00, 11.67it/s, loss=0.0169]
Checkpoint saved at epoch 194
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0159]
Epoch 195/300 - Average Loss: 0.0174██████████████████████████████████████████| 391/391 [00:58<00:00, 11.63it/s, loss=0.0159]
Checkpoint saved at epoch 195
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0167]
Epoch 196/300 - Average Loss: 0.0174██████████████████████████████████████████| 391/391 [00:53<00:00, 10.84it/s, loss=0.0167]
Checkpoint saved at epoch 196
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0185]
Epoch 197/300 - Average Loss: 0.0175██████████████████████████████████████████| 391/391 [00:58<00:00, 10.95it/s, loss=0.0185]
Checkpoint saved at epoch 197
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.53it/s, loss=0.0156]
Epoch 198/300 - Average Loss: 0.0174██████████████████████████████████████████| 391/391 [00:52<00:00, 11.67it/s, loss=0.0156]
Checkpoint saved at epoch 198
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0173]
Epoch 199/300 - Average Loss: 0.0174██████████████████████████████████████████| 391/391 [00:57<00:00, 11.58it/s, loss=0.0173]
Checkpoint saved at epoch 199
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0160]
Epoch 200/300 - Average Loss: 0.0174██████████████████████████████████████████| 391/391 [00:52<00:00, 11.80it/s, loss=0.0160]
Checkpoint saved at epoch 200
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.58it/s, loss=0.0194]
Epoch 201/300 - Average Loss: 0.0174██████████████████████████████████████████| 391/391 [00:56<00:00, 11.58it/s, loss=0.0194]
Checkpoint saved at epoch 201
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.64it/s, loss=0.0175]
Epoch 202/300 - Average Loss: 0.0174██████████████████████████████████████████| 391/391 [00:51<00:00, 11.59it/s, loss=0.0175]
Checkpoint saved at epoch 202
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:01<00:00,  6.41it/s, loss=0.0173]
Epoch 203/300 - Average Loss: 0.0174██████████████████████████████████████████| 391/391 [00:56<00:00, 11.08it/s, loss=0.0173]
Checkpoint saved at epoch 203
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0167]
Epoch 204/300 - Average Loss: 0.0174██████████████████████████████████████████| 391/391 [00:49<00:00, 11.59it/s, loss=0.0167]
Checkpoint saved at epoch 204
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.58it/s, loss=0.0169]
Epoch 205/300 - Average Loss: 0.0173██████████████████████████████████████████| 391/391 [00:54<00:00, 11.50it/s, loss=0.0169]
Checkpoint saved at epoch 205
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0156]
Epoch 206/300 - Average Loss: 0.0174██████████████████████████████████████████| 391/391 [00:49<00:00, 11.17it/s, loss=0.0156]
Checkpoint saved at epoch 206
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.63it/s, loss=0.0178]
Epoch 207/300 - Average Loss: 0.0173██████████████████████████████████████████| 391/391 [00:54<00:00, 11.63it/s, loss=0.0178]
Checkpoint saved at epoch 207
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0176]
Epoch 208/300 - Average Loss: 0.0174██████████████████████████████████████████| 391/391 [00:49<00:00, 11.26it/s, loss=0.0176]
Checkpoint saved at epoch 208
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.53it/s, loss=0.0186]
Epoch 209/300 - Average Loss: 0.0173██████████████████████████████████████████| 391/391 [00:54<00:00, 11.30it/s, loss=0.0186]
Checkpoint saved at epoch 209
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0168]
Epoch 210/300 - Average Loss: 0.0174██████████████████████████████████████████| 391/391 [00:58<00:00, 11.19it/s, loss=0.0168]
Checkpoint saved at epoch 210
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.64it/s, loss=0.0168]
Epoch 211/300 - Average Loss: 0.0173██████████████████████████████████████████| 391/391 [00:53<00:00, 11.66it/s, loss=0.0168]
Checkpoint saved at epoch 211
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0148]
Epoch 212/300 - Average Loss: 0.0173██████████████████████████████████████████| 391/391 [00:58<00:00, 11.65it/s, loss=0.0148]
Checkpoint saved at epoch 212
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.59it/s, loss=0.0171]
Epoch 213/300 - Average Loss: 0.0174██████████████████████████████████████████| 391/391 [00:53<00:00, 10.82it/s, loss=0.0171]
Checkpoint saved at epoch 213
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.46it/s, loss=0.0160]
Epoch 214/300 - Average Loss: 0.0173██████████████████████████████████████████| 391/391 [00:57<00:00, 11.55it/s, loss=0.0160]
Checkpoint saved at epoch 214
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.63it/s, loss=0.0154]
Epoch 215/300 - Average Loss: 0.0172██████████████████████████████████████████| 391/391 [00:51<00:00, 11.55it/s, loss=0.0154]
Checkpoint saved at epoch 215
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0169]
Epoch 216/300 - Average Loss: 0.0173██████████████████████████████████████████| 391/391 [00:56<00:00, 11.32it/s, loss=0.0169]
Checkpoint saved at epoch 216
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.66it/s, loss=0.0165]
Epoch 217/300 - Average Loss: 0.0173██████████████████████████████████████████| 391/391 [00:51<00:00, 11.39it/s, loss=0.0165]
Checkpoint saved at epoch 217
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0190]
Epoch 218/300 - Average Loss: 0.0173██████████████████████████████████████████| 391/391 [00:56<00:00, 11.58it/s, loss=0.0190]
Checkpoint saved at epoch 218
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.56it/s, loss=0.0160]
Epoch 219/300 - Average Loss: 0.0173██████████████████████████████████████████| 391/391 [00:51<00:00, 10.77it/s, loss=0.0160]
Checkpoint saved at epoch 219
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.50it/s, loss=0.0187]
Epoch 220/300 - Average Loss: 0.0172██████████████████████████████████████████| 391/391 [00:55<00:00, 11.15it/s, loss=0.0187]
Checkpoint saved at epoch 220
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0172]
Epoch 221/300 - Average Loss: 0.0173██████████████████████████████████████████| 391/391 [00:49<00:00, 11.67it/s, loss=0.0172]
Checkpoint saved at epoch 221
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.57it/s, loss=0.0177]
Epoch 222/300 - Average Loss: 0.0173██████████████████████████████████████████| 391/391 [00:54<00:00, 10.96it/s, loss=0.0177]
Checkpoint saved at epoch 222
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.63it/s, loss=0.0175]
Epoch 223/300 - Average Loss: 0.0172██████████████████████████████████████████| 391/391 [00:48<00:00, 11.47it/s, loss=0.0175]
Checkpoint saved at epoch 223
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0155]
Epoch 224/300 - Average Loss: 0.0172██████████████████████████████████████████| 391/391 [00:54<00:00, 11.58it/s, loss=0.0155]
Checkpoint saved at epoch 224
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.43it/s, loss=0.0176]
Epoch 225/300 - Average Loss: 0.0173██████████████████████████████████████████| 391/391 [00:58<00:00, 11.17it/s, loss=0.0176]
Checkpoint saved at epoch 225
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0166]
Epoch 226/300 - Average Loss: 0.0172██████████████████████████████████████████| 391/391 [00:51<00:00, 11.59it/s, loss=0.0166]
Checkpoint saved at epoch 226
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0162]
Epoch 227/300 - Average Loss: 0.0173██████████████████████████████████████████| 391/391 [00:56<00:00, 11.57it/s, loss=0.0162]
Checkpoint saved at epoch 227
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0166]
Epoch 228/300 - Average Loss: 0.0172██████████████████████████████████████████| 391/391 [00:51<00:00, 11.44it/s, loss=0.0166]
Checkpoint saved at epoch 228
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0160]
Epoch 229/300 - Average Loss: 0.0172██████████████████████████████████████████| 391/391 [00:56<00:00, 11.59it/s, loss=0.0160]
Checkpoint saved at epoch 229
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.57it/s, loss=0.0175]
Epoch 230/300 - Average Loss: 0.0172██████████████████████████████████████████| 391/391 [00:51<00:00, 11.51it/s, loss=0.0175]
Checkpoint saved at epoch 230
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.57it/s, loss=0.0179]
Epoch 231/300 - Average Loss: 0.0172██████████████████████████████████████████| 391/391 [00:55<00:00, 11.27it/s, loss=0.0179]
Checkpoint saved at epoch 231
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0172]
Epoch 232/300 - Average Loss: 0.0172██████████████████████████████████████████| 391/391 [00:50<00:00, 11.65it/s, loss=0.0172]
Checkpoint saved at epoch 232
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.63it/s, loss=0.0195]
Epoch 233/300 - Average Loss: 0.0172██████████████████████████████████████████| 391/391 [00:55<00:00, 11.30it/s, loss=0.0195]
Checkpoint saved at epoch 233
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0175]
Epoch 234/300 - Average Loss: 0.0172██████████████████████████████████████████| 391/391 [00:50<00:00, 11.62it/s, loss=0.0175]
Checkpoint saved at epoch 234
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.59it/s, loss=0.0186]
Epoch 235/300 - Average Loss: 0.0172██████████████████████████████████████████| 391/391 [00:55<00:00, 11.67it/s, loss=0.0186]
Checkpoint saved at epoch 235
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.45it/s, loss=0.0175]
Epoch 236/300 - Average Loss: 0.0172██████████████████████████████████████████| 391/391 [00:59<00:00, 11.49it/s, loss=0.0175]
Checkpoint saved at epoch 236
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.63it/s, loss=0.0166]
Epoch 237/300 - Average Loss: 0.0172██████████████████████████████████████████| 391/391 [00:52<00:00, 11.53it/s, loss=0.0166]
Checkpoint saved at epoch 237
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0171]
Epoch 238/300 - Average Loss: 0.0172██████████████████████████████████████████| 391/391 [00:57<00:00, 10.94it/s, loss=0.0171]
Checkpoint saved at epoch 238
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0175]
Epoch 239/300 - Average Loss: 0.0171██████████████████████████████████████████| 391/391 [00:53<00:00, 11.55it/s, loss=0.0175]
Checkpoint saved at epoch 239
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0175]
Epoch 240/300 - Average Loss: 0.0172██████████████████████████████████████████| 391/391 [00:58<00:00, 11.33it/s, loss=0.0175]
Checkpoint saved at epoch 240
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.51it/s, loss=0.0188]
Epoch 241/300 - Average Loss: 0.0172██████████████████████████████████████████| 391/391 [00:52<00:00, 11.19it/s, loss=0.0188]
Checkpoint saved at epoch 241
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.59it/s, loss=0.0185]
Epoch 242/300 - Average Loss: 0.0171██████████████████████████████████████████| 391/391 [00:56<00:00, 11.14it/s, loss=0.0185]
Checkpoint saved at epoch 242
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0165]
Epoch 243/300 - Average Loss: 0.0172██████████████████████████████████████████| 391/391 [00:51<00:00, 11.54it/s, loss=0.0165]
Checkpoint saved at epoch 243
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0189]
Epoch 244/300 - Average Loss: 0.0172██████████████████████████████████████████| 391/391 [00:56<00:00, 11.30it/s, loss=0.0189]
Checkpoint saved at epoch 244
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.59it/s, loss=0.0164]
Epoch 245/300 - Average Loss: 0.0171██████████████████████████████████████████| 391/391 [00:51<00:00, 11.64it/s, loss=0.0164]
Checkpoint saved at epoch 245
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.58it/s, loss=0.0186]
Epoch 246/300 - Average Loss: 0.0171██████████████████████████████████████████| 391/391 [00:55<00:00, 11.30it/s, loss=0.0186]
Checkpoint saved at epoch 246
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.47it/s, loss=0.0179]
Epoch 247/300 - Average Loss: 0.0171██████████████████████████████████████████| 391/391 [01:00<00:00, 11.24it/s, loss=0.0179]
Checkpoint saved at epoch 247
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.65it/s, loss=0.0188]
Epoch 248/300 - Average Loss: 0.0172██████████████████████████████████████████| 391/391 [00:53<00:00, 11.69it/s, loss=0.0188]
Checkpoint saved at epoch 248
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.63it/s, loss=0.0160]
Epoch 249/300 - Average Loss: 0.0171██████████████████████████████████████████| 391/391 [00:49<00:00, 11.62it/s, loss=0.0160]
Checkpoint saved at epoch 249
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0177]
Epoch 250/300 - Average Loss: 0.0171██████████████████████████████████████████| 391/391 [00:54<00:00, 11.63it/s, loss=0.0177]
Checkpoint saved at epoch 250
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0173]
Epoch 251/300 - Average Loss: 0.0171██████████████████████████████████████████| 391/391 [00:59<00:00, 11.67it/s, loss=0.0173]
Checkpoint saved at epoch 251
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.47it/s, loss=0.0165]
Epoch 252/300 - Average Loss: 0.0171██████████████████████████████████████████| 391/391 [00:53<00:00, 11.06it/s, loss=0.0165]
Checkpoint saved at epoch 252
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.59it/s, loss=0.0154]
Epoch 253/300 - Average Loss: 0.0171██████████████████████████████████████████| 391/391 [00:56<00:00, 11.66it/s, loss=0.0154]
Checkpoint saved at epoch 253
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.65it/s, loss=0.0175]
Epoch 254/300 - Average Loss: 0.0171██████████████████████████████████████████| 391/391 [00:51<00:00, 11.64it/s, loss=0.0175]
Checkpoint saved at epoch 254
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0163]
Epoch 255/300 - Average Loss: 0.0171██████████████████████████████████████████| 391/391 [00:56<00:00, 10.97it/s, loss=0.0163]
Checkpoint saved at epoch 255
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.58it/s, loss=0.0150]
Epoch 256/300 - Average Loss: 0.0171██████████████████████████████████████████| 391/391 [00:51<00:00, 11.34it/s, loss=0.0150]
Checkpoint saved at epoch 256
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0184]
Epoch 257/300 - Average Loss: 0.0171██████████████████████████████████████████| 391/391 [00:56<00:00, 11.62it/s, loss=0.0184]
Checkpoint saved at epoch 257
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:01<00:00,  6.41it/s, loss=0.0180]
Epoch 258/300 - Average Loss: 0.0171██████████████████████████████████████████| 391/391 [01:00<00:00, 11.48it/s, loss=0.0180]
Checkpoint saved at epoch 258
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0173]
Epoch 259/300 - Average Loss: 0.0171██████████████████████████████████████████| 391/391 [00:53<00:00, 11.32it/s, loss=0.0173]
Checkpoint saved at epoch 259
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0175]
Epoch 260/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:58<00:00, 11.50it/s, loss=0.0175]
Checkpoint saved at epoch 260
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0161]
Epoch 261/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:53<00:00, 11.63it/s, loss=0.0161]
Checkpoint saved at epoch 261
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.64it/s, loss=0.0165]
Epoch 262/300 - Average Loss: 0.0171██████████████████████████████████████████| 391/391 [00:58<00:00, 11.60it/s, loss=0.0165]
Checkpoint saved at epoch 262
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.49it/s, loss=0.0174]
Epoch 263/300 - Average Loss: 0.0171██████████████████████████████████████████| 391/391 [00:53<00:00, 11.06it/s, loss=0.0174]
Checkpoint saved at epoch 263
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.64it/s, loss=0.0160]
Epoch 264/300 - Average Loss: 0.0171██████████████████████████████████████████| 391/391 [00:56<00:00, 11.64it/s, loss=0.0160]
Checkpoint saved at epoch 264
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.66it/s, loss=0.0189]
Epoch 265/300 - Average Loss: 0.0171██████████████████████████████████████████| 391/391 [00:52<00:00, 11.62it/s, loss=0.0189]
Checkpoint saved at epoch 265
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0178]
Epoch 266/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:57<00:00, 11.33it/s, loss=0.0178]
Checkpoint saved at epoch 266
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.59it/s, loss=0.0155]
Epoch 267/300 - Average Loss: 0.0171██████████████████████████████████████████| 391/391 [00:52<00:00, 11.29it/s, loss=0.0155]
Checkpoint saved at epoch 267
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.58it/s, loss=0.0160]
Epoch 268/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:57<00:00, 11.61it/s, loss=0.0160]
Checkpoint saved at epoch 268
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.48it/s, loss=0.0151]
Epoch 269/300 - Average Loss: 0.0171██████████████████████████████████████████| 391/391 [00:51<00:00, 11.58it/s, loss=0.0151]
Checkpoint saved at epoch 269
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0156]
Epoch 270/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:55<00:00, 11.69it/s, loss=0.0156]
Checkpoint saved at epoch 270
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0162]
Epoch 271/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:49<00:00, 11.44it/s, loss=0.0162]
Checkpoint saved at epoch 271
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.59it/s, loss=0.0175]
Epoch 272/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:54<00:00, 11.36it/s, loss=0.0175]
Checkpoint saved at epoch 272
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0150]
Epoch 273/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:49<00:00, 11.62it/s, loss=0.0150]
Checkpoint saved at epoch 273
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.46it/s, loss=0.0167]
Epoch 274/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:54<00:00, 11.19it/s, loss=0.0167]
Checkpoint saved at epoch 274
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.63it/s, loss=0.0166]
Epoch 275/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:57<00:00, 11.27it/s, loss=0.0166]
Checkpoint saved at epoch 275
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.66it/s, loss=0.0174]
Epoch 276/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:53<00:00, 11.60it/s, loss=0.0174]
Checkpoint saved at epoch 276
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0153]
Epoch 277/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:58<00:00, 11.68it/s, loss=0.0153]
Checkpoint saved at epoch 277
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.58it/s, loss=0.0161]
Epoch 278/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:53<00:00, 10.95it/s, loss=0.0161]
Checkpoint saved at epoch 278
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0183]
Epoch 279/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:58<00:00, 11.55it/s, loss=0.0183]
Checkpoint saved at epoch 279
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.52it/s, loss=0.0177]
Epoch 280/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:52<00:00, 11.63it/s, loss=0.0177]
Checkpoint saved at epoch 280
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.67it/s, loss=0.0156]
Epoch 281/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:56<00:00, 11.61it/s, loss=0.0156]
Checkpoint saved at epoch 281
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.60it/s, loss=0.0181]
Epoch 282/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:52<00:00, 10.65it/s, loss=0.0181]
Checkpoint saved at epoch 282
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.58it/s, loss=0.0167]
Epoch 283/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:57<00:00, 11.35it/s, loss=0.0167]
Checkpoint saved at epoch 283
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0172]
Epoch 284/300 - Average Loss: 0.0169██████████████████████████████████████████| 391/391 [00:51<00:00, 11.60it/s, loss=0.0172]
Checkpoint saved at epoch 284
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.45it/s, loss=0.0168]
Epoch 285/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:56<00:00, 11.54it/s, loss=0.0168]
Checkpoint saved at epoch 285
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.65it/s, loss=0.0154]
Epoch 286/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:50<00:00, 11.23it/s, loss=0.0154]
Checkpoint saved at epoch 286
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.64it/s, loss=0.0159]
Epoch 287/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:55<00:00, 11.61it/s, loss=0.0159]
Checkpoint saved at epoch 287
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0158]
Epoch 288/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:50<00:00, 11.71it/s, loss=0.0158]
Checkpoint saved at epoch 288
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.63it/s, loss=0.0169]
Epoch 289/300 - Average Loss: 0.0169██████████████████████████████████████████| 391/391 [00:55<00:00, 11.65it/s, loss=0.0169]
Checkpoint saved at epoch 289
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.58it/s, loss=0.0152]
Epoch 290/300 - Average Loss: 0.0169██████████████████████████████████████████| 391/391 [00:50<00:00, 10.93it/s, loss=0.0152]
Checkpoint saved at epoch 290
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.53it/s, loss=0.0173]
Epoch 291/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:54<00:00, 11.25it/s, loss=0.0173]
Checkpoint saved at epoch 291
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.65it/s, loss=0.0147]
Epoch 292/300 - Average Loss: 0.0169██████████████████████████████████████████| 391/391 [00:58<00:00, 11.74it/s, loss=0.0147]
Checkpoint saved at epoch 292
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.62it/s, loss=0.0170]
Epoch 293/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:53<00:00, 11.63it/s, loss=0.0170]
Checkpoint saved at epoch 293
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0164]
Epoch 294/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:58<00:00, 11.63it/s, loss=0.0164]
Checkpoint saved at epoch 294
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.58it/s, loss=0.0148]
Epoch 295/300 - Average Loss: 0.0169██████████████████████████████████████████| 391/391 [00:53<00:00, 11.34it/s, loss=0.0148]
Checkpoint saved at epoch 295
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [01:00<00:00,  6.48it/s, loss=0.0151]
Epoch 296/300 - Average Loss: 0.0170██████████████████████████████████████████| 391/391 [00:58<00:00, 11.35it/s, loss=0.0151]
Checkpoint saved at epoch 296
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.63it/s, loss=0.0157]
Epoch 297/300 - Average Loss: 0.0169██████████████████████████████████████████| 391/391 [00:52<00:00, 11.60it/s, loss=0.0157]
Checkpoint saved at epoch 297
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:58<00:00,  6.65it/s, loss=0.0190]
Epoch 298/300 - Average Loss: 0.0169██████████████████████████████████████████| 391/391 [00:57<00:00, 11.59it/s, loss=0.0190]
Checkpoint saved at epoch 298
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.61it/s, loss=0.0170]
Epoch 299/300 - Average Loss: 0.0169██████████████████████████████████████████| 391/391 [00:52<00:00, 11.64it/s, loss=0.0170]
Checkpoint saved at epoch 299
Training: 100%|███████████████████████████████████████████████████████████████| 391/391 [00:59<00:00,  6.63it/s, loss=0.0170]
Epoch 300/300 - Average Loss: 0.0169██████████████████████████████████████████| 391/391 [00:57<00:00, 11.67it/s, loss=0.0170]
Checkpoint saved at epoch 300
Epochs: 100%|████████████████████████████████████████████████████████████████████████████| 260/260 [4:43:51<00:00, 65.50s/it]
(mae_env) tarioyou ~/masked-auto-encoder % python inference.py
Loaded pretrained model weights.
Reconstructed image saved to data/img_reconstructed.png
(mae_env) tarioyou ~/masked-auto-encoder % 
"""


# Extract epochs and losses
pattern = r"Epoch (\d+)/\d+ - Average Loss: ([\d.]+)"
matches = re.findall(pattern, log_data)

epochs = [int(m[0]) for m in matches]
losses = [float(m[1]) for m in matches]

pattern_per = r"Epoch (\d+)/\d+ .*? loss=([\d.]+)"
matches_per = re.findall(pattern_per, log_data)  # Fixed this line

epochs_per = [int(m[0]) for m in matches_per]
losses_per = [float(m[1]) for m in matches_per]

print(losses_per)

# Plot the loss curve
plt.figure(figsize=(10, 5))
plt.plot(epochs, losses, marker='o', linestyle='-',
         color='b', label="Average Loss")
plt.plot(epochs_per, losses_per, marker='o', linestyle='-', color='r',
         label="Epoch Loss")  # Changed color to differentiate
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()  # Added legend
plt.grid()
plt.savefig("loss.png")
plt.show()
