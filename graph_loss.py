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
