# Countering Adversarial Images Using Input Transformations

# Overview
This package implements the experiments described in the paper [Countering Adversarial Images Using Input Transformations](https://arxiv.org/pdf/1711.00117.pdf).
It contains implementations for [adversarial attacks](#adversarial_attack), [defenses based image transformations](#image_transformation), [training](#training), and [testing](#classify) convolutional networks under adversarial attacks using our defenses. We also provide [pre-trained models](#pretrained).

If you use this code, please cite our paper:

- Chuan Guo, Mayank Rana, Moustapha Cisse, and Laurens van der Maaten. **Countering Adversarial Images using Input Transformations**. arXiv 1711.00117, 2017. [[PDF](https://arxiv.org/pdf/1711.00117.pdf)]

## Adversarial Defenses
The code implements the following four defenses against adversarial images, all of which are based on image transformations:
- Image quilting
- Total variation minimization
- JPEG compression
- Pixel quantization

Please refer to the paper for details on these defenses. A detailed description of the original image quilting algorithm can be found [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-siggraph01.pdf); a detailed description of our solver for total variation minimization can be found [here](ftp://ftp.math.ucla.edu/pub/camreport/cam08-29.pdf).

## Adversarial Attacks

The code implements the following four approaches to generating adversarial images:
- [Fast gradient sign method (FGSM)](https://arxiv.org/abs/1412.6572)
- [Iterative FGSM](https://arxiv.org/abs/1611.01236)
- [DeepFool](https://arxiv.org/abs/1511.04599)
- [Carlini-Wagner attack](https://arxiv.org/abs/1608.04644)


# Installation
To use this code, first install Python, [PyTorch](www.pytorch.org), and [Faiss](https://github.com/facebookresearch/faiss) (to perform image quilting). We tested the code using Python 2.7, PyTorch v0.2.0, and scikit-image 0.11; your mileage may vary when using other versions.

Pytorch can be installed using the instructions [here](http://pytorch.org/). Faiss is required to run the image quilting algorithm; it is not automatically included because faiss does not have a pip support and because it requires configuring BLAS and LAPACK flags, as described [here](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md). Please install faiss using the instructions given [here](https://github.com/facebookresearch/faiss).

The code uses several other external dependencies (for training Inception models, performing Bregman iteration, etc.). These dependencies are automatically downloaded and installed when you install this package via `pip`:
```bash
# Install from source  
cd adversarial_image_defenses
pip install .

```

# Usage

To import the package in Python:
```python
import adversarial
```

The functionality implemented in this package is demonstrated in [this example](https://github.com/facebookresearch/adversarial_image_defenses/blob/master/adversarial/examples/demo.py). Run the example via:
```bash
python adversarial/examples/demo.py
```


## API
The full functionality of the package is exposed via several runnable Python scripts. All these scripts require the user to specify the path to the Imagenet dataset, the path to pre-trained models, and the path to quilted images (once they are computed) in `lib/path_config.json`. Alternatively, the paths can be passed as input arguments into the scripts.


### Generate quilting patches
[`index_patches.py`](adversarial/index_patches.py) creates a faiss index of images patches. This index can be used to perform quilting of images.

Code example:
```python  
import adversarial
from index_patches import create_faiss_patches, parse_args

args = parse_args()
# Update args if needed
args.patch_size = 5
create_faiss_patches(args)
```

Alternatively, run `python index_patches.py`. The following arguments are supported:
- `--patch_size`          Patch size (square) that will be used in quilting (default: 5).
- `--num_patches`         Number of patches to generate (default: 1000000).
- `--pca_dims`            PCA dimension for faiss (default: 64).
- `--patches_file`        File in which patches are saved.
- `--index_file`          File in which faiss index of patches is saved.


<a name="image_transformation"></a>
### Image transformations
[`gen_transformed_images.py`](adversarial/gen_transformed_images.py) has applies an image transformation on (adversarial or non-adversarial) ImageNet images, and saves them to disk. Image transformations such as image quilting are too computationally intensive to be performed on-the-fly during network training, which is why we precompute the transformed images.

Code example:
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

Alternatively, run `python gen_transformed_images.py`. In addition to the [common arguments](#common_args) and [adversarial arguments](adversarial_args), the following arguments are supported:
- `--operation`           Operation to run. Supported operations are:  
    `transformation_on_raw`: Apply transformations on raw images.
    `transformation_on_adv`: Apply transformations on adversarial images.
    `cat_data`: Concatenate output from distributed `transformation_on_adv`.
- `--data_type`           Data type (`train` or `raw`) for `transformation_on_raw` (default: `train`).
- `--out_dir`             Directory path for output of `cat_data`.
- `--partition_dir`       Directory path to output transformed data.
- `--data_batches`        Number of data batches to generate. Used for random crops for ensembling.
- `--partition`           Distributed data partition (default: 0).
- `--partition_size`      The size of each data partition.  
    For `transformation_on_raw`, partition_size represents number of classes for each process.  
    For `transformation_on_adv`, partition_size represents number of images for each process.  
- `--n_threads`           Number of threads for `transformation_on_raw`.


### Generate TAR data index
Many file systems perform poorly when dealing with millions of small files (such as images). Therefore, we generally TAR our image datasets (obtained by running `generate_transformed_images`). Next, we use
[`gen_tar_index.py`](adversarial/gen_tar_index.py) to generate a file index for the TAR file. The file index facilitates fast, random-access reading of the TAR file; it is much faster and requires less memory than untarring the data or using `tarfile` package.

Code example:
```python  
import adversarial
from gen_tar_index import generate_tar_index, parse_args

args = parse_args()
generate_tar_index(args)
```

Alternatively, run `python gen_tar_index.py`. The following arguments are supported:
- `--tar_path`          Path for TAR file or directory.
- `--index_root`        Directory in which to store TAR index file.
- `--path_prefix`       Prefix to identify TAR member names to be indexed.


<a name="adversarial_attack"></a>
### Adversarial Attacks
[`gen_adversarial_images.py`](adversarial/gen_adversarial_images.py) implements the generation of adversarial images for the ImageNet dataset.

Code example:
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

Alternatively, run `python gen_adversarial_images.py`. For a list of the supported arguments, see [common arguments](#common_args) and [adversarial arguments](adversarial_args).


<a name="training"></a>
### Training
[`train_model.py`](adversarial/train_model.py) implements the training of convolutional networks on (transformed or non-transformed) ImageNet images.

Code example:
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

Alternatively, run `python train_model.py`. In addition to the [common arguments](#common_args), the following arguments are supported:
- `--resume`                    Resume training from checkpoint (if available).
- `--lr`                        Initial learning rate defined in [constants.py] (lr=0.045 for Inception-v4, 0.1 for other models).
- `--lr_decay`                  Exponential learning rate decay defined in [constants.py] (0.94 for inception_v4, 0.1 for other models).
- `--lr_decay_stepsize`         Decay learning rate after every stepsize epochs defined in [constants.py] (0.94 for inception_v4, 0.1 for other models).
- `--momentum`                  Momentum (default: 0.9).
- `--weight_decay`              Amount of weight decay (default: 1e-4).
- `--start_epoch`               Index of first epoch (default: 0).
- `--end_epoch`                 Index of last epoch (default: 90).
- `--preprocessed_epoch_data`   Augmented and transformed data for each epoch is pre-generated (default: `False`).

<a name="classify"></a>
### Testing
[`classify_images.py`](adversarial/classify_images.py) implements the testing of a training convolutional network on an dataset of (adversarial or non-adversarial / transformed or non-transformed) ImageNet images.

Code exammple:
```python
import adversarial
from classify_images import classify_images
from lib import opts
# load default args
args = opts.parse_args(opts.OptType.CLASSIFY)

classify_images(args)
```

Alternatively, run `python classify_images.py`. In addition to the [common arguments](#common_args), the following arguments are supported:
- `--ensemble`            Ensembling type, `None`, `avg`, `max` (default: `None`).
- `--ncrops`              List of number of crops for each defense to use for ensembling (default: `None`).
- `--crop_frac`           List of crop fraction for each defense to use for ensembling (default: `None`).
- `--crop_type`           List of crop type(`center`, `random`, `sliding`(hardset for 9 crops)) for each defense to use for ensembling (default: `None`).

<a name="pretrained"></a>
### Pre-trained models
We provide pre-trained models that were trained on ImageNet images that were processed using total variation minimization (TVM) or image quilting can be downloaded from the following links (set the `models_root` argument to the path that contains these model model files):

- [ResNet-50_model trained on quilted images](https://s3.amazonaws.com/adversarial-images/models/resnet50_quilting.tar.gz)
- [ResNet-50_model trained on TVM images](https://s3.amazonaws.com/adversarial-images/models/resnet50_tvm.tar.gz)
- [ResNet-101_model trained on quilted images](https://s3.amazonaws.com/adversarial-images/models/resnet101_quilting.tar.gz)
- [ResNet-101_model trained on TVM images](https://s3.amazonaws.com/adversarial-images/models/resnet101_tvm.tar.gz)
- [DenseNet-169_model trained on quilted images](https://s3.amazonaws.com/adversarial-images/models/densenet169_quilting.tar.gz)
- [DenseNet-169_model trained on TVM images](https://s3.amazonaws.com/adversarial-images/models/densenet169_tvm.tar.gz)
- [Inception-v4_model trained on quilted images](https://s3.amazonaws.com/adversarial-images/models/inception_v4_quilting.tar.gz)
- [Inception-v4_model trained on TVM images](https://s3.amazonaws.com/adversarial-images/models/inception_v4_tvm.tar.gz)


<a name="common_args"></a>
### Common arguments

The following arguments are used by multiple scripts, including
`generate_transformed_images`, `train_model`, and `classify_images`:

#### Paths
- `--data_root`             Main data directory to save and read data.
- `--models_root`           Directory path to store/load models.
- `--tar_dir`               Directory path for transformed images(train/val) stored in TAR files.
- `--tar_index_dir`         Directory path for index files for transformed images in TAR files.
- `--quilting_index_root`   Directory path for quilting index files.
- `--quilting_patch_root`   Directory path for quilting patch files.

#### Train/Classifier params
- `--model`                 Model to use (default: `resnet50`).
- `--device`                Device to use: cpu or gpu (default: `gpu`).
- `--normalize`             Normalize image data.
- `--batchsize`             Batch size for training and testing (default: 256).
- `--preprocessed_data`     Transformations/Defenses are already applied on saved images (default: `False`).
- `--defenses`              List of defenses to apply: `raw` (no defense), `tvm`, `quilting`, `jpeg`, `quantization` (default: `None`).
- `--pretrained`            Use pretrained model from PyTorch model zoo (default: `False`).

#### Tranformation params
- `--tvm_weight`            Regularization weight for total variation minimization (TVM).
- `--pixel_drop_rate`       Pixel drop rate to use in TVM.
- `--tvm_method`            Reconstruction method to use in TVM (default: `bregman`).
- `--quilting_patch_size`   Patch size to use in image quilting.
- `--quilting_neighbors`    Number of nearest patches to sample from in image quilting (default: 1).
- `--quantize_depth`        Bit depth for quantization defense (default: 8).


<a name="adversarial_args"></a>
#### Adversarial arguments
The following arguments are used whem generating adversarial images with `gen_transformed_images.py`:

- `--n_samples`             Maximum number of samples to test on.
- `--attack_type`           Attack type: `None` (no attack), `blackbox`, `whitebox` (default: `None`).
- `--adversary`             Adversary to use: `fgs`, `ifgs`, `cwl2`, `deepfool` (default: `None`).
- `--adversary_model`       Model to use for generating adversarial images (default: `resnet50`).
- `--learning_rate`         Learning rate for iterative adversarial attacks (default: read from constants).
- `--adv_strength`          Adversarial strength for non-iterative adversarial attacks (default: read from constants).
- `--adversarial_root`      Path containing adversarial images.
