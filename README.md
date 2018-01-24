# Countering Adversarial Images Using Input Transformations

# Overview
This repo contains code used for [Countering Adversarial Images Using Input Transformations](https://arxiv.org/pdf/1711.00117.pdf).
The code has implementation for [adversarial attacks](#adversarial_attack), [image transformations](#image_transformation), [training](#training), and [testing](#classify) ConvNet models

## Image Transformations
### Image Quilting
![Image Quilting](adversarial/test/images/sample/lena_quilting.png)

More description for image quilting can be found [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-siggraph01.pdf)

### TVM
![Total Variance Minimization](adversarial/test/images/sample/lena_tvm.png)

### JPEG
Transform image by compressing and uncompressing image using jpeg compression

### Quantization
Transform image by quantization


## Adversarial Attacks

### [Fast Gradient Sign Method(FGSM)](https://arxiv.org/abs/1412.6572)

### [IFGS](https://arxiv.org/abs/1611.01236)

### [DEEPFOOL](https://arxiv.org/abs/1511.04599)

### [CWL](https://arxiv.org/abs/1608.04644)


# Setup
Requires Python 2.7, [PyTorch v0.2.0](www.pytorch.org), [Faiss](https://github.com/facebookresearch/faiss) (for Image Quilting)

## Requirements

### Faiss
Faiss is required for quilting.
It needs to be installed from [here](https://github.com/facebookresearch/faiss).
It is not part of the standard installation as it doesn't have a pip support and requires configuring BLAS/Lapack flags as described [here](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md) .

### pytorch
Pytorch for your can be installed from [here](http://pytorch.org/)

### External dependencies
Our code uses some external code(inception models, tv_bregman from scikit-image) which is not included in source and are downloaded as a part of the installation process. See setup.py for more details.

All other dependencies are installed via `setup.py`. See requirements in [setup.py](setup.py) for more details.


```bash
# Install from source  
pip install .

```

# Usage

To import the package

```python
import adversarial
```

## Demos
```
python adversarial/examples/demo.py
```

For more details see [examples](adversarial/examples/demo.py)


## API
For a quick introduction into the capabilities, have a look at the example directory, or read the details below.
By default test data will be used. To update path for data sources, either update path_config.json or pass corresponding args


### Generate quilting patches
[`index_patches.py`](adversarial/index_patches.py) creates FAISS indices of patches for quilting

```python  
import adversarial
from index_patches import create_faiss_patches, parse_args

args = parse_args()
# Update args if needed
args.patch_size = 5
create_faiss_patches(args)

```

The following args are supported

- `--patch_size`          Patch size(square) to crop from image (default: 5)
- `--num_patches`         Number of patches to generate (default: 1000000)
- `--pca_dims`            PCA dimension for FAISS (default: 64)
- `--patches_file`        Path where patches should be stored
- `--index_file`          Path where indices for patches should be stores


### Generate tar data index
If you have saved your dataset in tar format then you can use
[`gen_tar_index.py`](adversarial/gen_tar_index.py) to generate indices for tar data. Once indices are generated then data could be directly read from tar files using these indices. This is much faster and requires lesser memory than untarring everything at once or using standard tarfile package.

```python  
import adversarial
from gen_tar_index import generate_tar_index, parse_args

args = parse_args()
generate_tar_index(args)

```

The following args are supported

- `--tar_path`          Path for tar file or directory
- `--index_root`        Directory path to store tar index object
- `--path_prefix`       prefix in all tar member names'


<a name="image_transformation"></a>
### Image transformation
[`gen_transformed_images.py`](adversarial/gen_transformed_images.py) has implementation to apply image transformation and save them.
Transformations can be applied distributedly as well.

```python
import adversarial
from gen_transformed_images import generate_transformed_images
from lib import opts
# load default args for transformation functions
args = opts.parse_args(opts.OptType.TRANSFORMATION)
args.operation = "transformation_on_raw"
args.defenses = ["tvm"]
args.partition_size = 1  # Number of samples to generate

generate_transformed_images(args)

```

For more details see [examples](adversarial/examples/demo.py)

Besides [Common arguments](#common_args) and [Adversarial arguments](adversarial_args), the following arguments are supported

- `--operation`           Operation to run. Supported operations are:  
    `transformation_on_raw`: Apply transformations on raw images  
    `transformation_on_adv`: Apply transformations on adversarial images  
    `cat_data`: Concatenate output from distributed `transformation_on_adv`

- `--data_type`           Data type (`train` or `raw`) for `transformation_on_raw` (Default: `train`)
- `--out_dir`             Directory path for output of `cat_data`
- `--partition_dir`       Directory path to output transformed data
- `--data_batches`        Number of data batches to generate. Used for random crops for ensembling.
- `--partition`           Distributed data partition (default: 0)
- `--partition_size`      The size of each data partition.  
    For `transformation_on_raw`, partition_size represents number of classes for each process  
    For `transformation_on_adv`, partition_size represents number of images for each process  
- `--n_threads`           Number of threads for `transformation_on_raw`


<a name="adversarial_attack"></a>
### Adversarial Attacks
[`gen_adversarial_images.py`](adversarial/gen_adversarial_images.py) has implementation to generate adversarial attacks.


```python
import adversarial
from gen_adversarial_images import generate_adversarial_images
from lib import opts
# load default args for adversary functions
args = opts.parse_args(opts.OptType.ADVERSARIAL)
args.model = "resnet50"
args.adversary_to_generate = "fgs"
args.partition_size = 1  # Number of samples to generate
args.data_type = "val"  # input dataset type
args.normalize = True  # apply normalization on input data
args.attack_type = "blackbox"  # For <whitebox> attack, use transformed models
args.pretrained = True  # Use pretrained model from model-zoo

generate_adversarial_images(args)

```

For more details see [examples](adversarial/examples/demo.py)

For list of supported arguments, see [Common arguments](#common_args) and [Adversarial arguments](adversarial_args)


<a name="training"></a>
### Train
[`train_model.py`](adversarial/train_model.py) has implementation for training convnets

```python
import adversarial
from train_model import train_model
from lib import opts
# load default args
args = opts.parse_args(opts.OptType.TRAIN)
args.defenses = None  # defense=<(raw, tvm, quilting, jpeg, quantization)>
args.model = "resnet50"
args.normalize = True  # apply normalization on input data

train_model(args)

```

For more details see [examples](adversarial/examples/demo.py)

Besides [Common arguments](#common_args), the following arguments are supported

- `--resume`                    Resume training from checkpoint (if available)
- `--lr`                        Initial learning rate defined in [constants.py] (lr=0.045 for inception*, 0.1 for other models)
- `--lr_decay`                  Exponential learning rate decay defined in [constants.py] (0.94 for inception_v4, 0.1 for other models)
- `--lr_decay_stepsize`         Decay learning rate after every stepsize epochs defined in [constants.py] (0.94 for inception_v4, 0.1 for other models)
- `--momentum`                  Momentum (default: 0.9)
- `--weight_decay`              Amount of weight decay (default: 1e-4)
- `--start_epoch`               Index of first epoch (default: 0)
- `--end_epoch`                 Index of last epoch (default: 90)   
- `--preprocessed_epoch_data`   Augmented and transformed data for each epoch is pre-generated (default: `False`)

<a name="classify"></a>
### Classify
[`classify_images.py`](adversarial/classify_images.py) has implementation for classifying images.
The pretrained models on imagenet using tvm and	quilting input transformations can be downloaded from the following links(update `models_root` arg to the path where models are downloaded).
- [resnet50_quilting](https://s3.amazonaws.com/adversarial-images/models/resnet50_quilting.tar.gz)
- [resnet50_tvm](https://s3.amazonaws.com/adversarial-images/models/resnet50_tvm.tar.gz)
- [resnet101_quilting](https://s3.amazonaws.com/adversarial-images/models/resnet101_quilting.tar.gz)
- [resnet101_tvm](https://s3.amazonaws.com/adversarial-images/models/resnet101_tvm.tar.gz)
- [densenet169_quilting](https://s3.amazonaws.com/adversarial-images/models/densenet169_quilting.tar.gz)
- [densenet169_tvm](https://s3.amazonaws.com/adversarial-images/models/densenet169_tvm.tar.gz)
- [inception_v4_quilting](https://s3.amazonaws.com/adversarial-images/models/inception_v4_quilting.tar.gz)
- [inception_v4_tvm](https://s3.amazonaws.com/adversarial-images/models/inception_v4_tvm.tar.gz)

```python
import adversarial
from classify_images import classify_images
from lib import opts
# load default args
args = opts.parse_args(opts.OptType.CLASSIFY)

classify_images(args)

```

For more details see [examples](adversarial/examples/demo.py)

Besides [Common arguments](#common_args), the following arguments are supported


- `--ensemble`            Ensembling type, `None`, `avg`, `max` (default: `None`)
- `--ncrops`              List of number of crops for each defense to use for ensembling (default: `None`)
- `--crop_frac`           List of crop fraction for each defense to use for ensembling (default: `None`)
- `--crop_type`           List of crop type(`center`, `random`, `sliding`(hardset for 9 crops)) for each defense to use for ensembling (default: `None`)


<a name="common_args"></a>
### Common arguments

Following parameters are used by multiple functions
`generate_transformed_images.py`, `train_model`, `classify_images`

#### Paths
- `--data_root`             Main data directory to save and read data
- `--models_root`           Directory path to store/load models
- `--tar_dir`               Directory path for transformed images(train/val) stored in tar files
- `--tar_index_dir`         Directory path for index files for transformed images in tar files
- `--quilting_index_root`   Directory path for quilting index files
- `--quilting_patch_root`   Directory path for quilting patch files

#### Train/Classifier params
- `--model`                 Model to use (default: `resnet50`)
- `--device`                Device to use: cpu or gpu (default: `gpu`)
- `--normalize`             Normalize image data
- `--batchsize`             Batch size for training and testing (default: 256)
- `--preprocessed_data`     Transformations/Defenses are already applied on saved images (default: `False`)
- `--defenses`              List of defenses to apply like raw(Nothing), tvm, quilting, jpeg, quantization (default: `None`)
- `--pretrained`            Pse pretrained model from model-zoo (default: `False`)

#### Tranformation params
- `--tvm_weight`            Weight for TVM
- `--pixel_drop_rate`       Pixel drop rate to use in TVM
- `--tvm_method`            Reconstruction method to use in TVM(default: bregman)
- `--quilting_patch_size`   Patch size to use in quilting
- `--quilting_neighbors`    Number of nearest neighbors to use for quilting patches to randomly chose patch from (default: 1)
- `--quantize_depth`        Bit depth for quantization defense (default: 8)


<a name="adversarial_args"></a>
#### Adversarial arguments
Following arguments are used in adversarial operations by `gen_transformed_images.py`

- `--n_samples`             Max number of samples to test on
- `--attack_type`           Attack type, `None`(No attack), `blackbox`, `whitebox` (default: `None`)
- `--adversary`             Adversary to use, `fgs`, `ifgs`, `cwl2`, `deepfool` (default: `None`)
- `--adversary_model`       Adversarial model to use (default: `resnet50`)
- `--learning_rate`         Learning rate for iterative adversarial attacks (default read from constants)
- `--adv_strength`          Adversarial strength for non iterative adversarial attacks (default read from constants)
- `--adversarial_root`      Directory path adversary data
