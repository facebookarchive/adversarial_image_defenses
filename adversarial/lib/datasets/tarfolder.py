from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch.utils.data as data

import sys
from os import listdir
from os.path import isdir, isfile, join, basename
import tarfile
from lib.datasets.tar_metadata import (TarDirMetaData, TarFileMetadata,
                                        DataFileMetadata, extract_classname_from_member)
from PIL import Image
try:
    import cPickle as pickle
except ImportError:
    import pickle

if sys.version_info[0] == 3:  # for python3
    from io import StringIO
    py3 = True
else:
    from cStringIO import StringIO
    py3 = False


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
TAR_EXTENSIONS = ['.tar', '.TAR', '.tar.gz', '.TAR.GZ']


def _is_tar_file(filename):
    return any(filename.endswith(extension) for extension in TAR_EXTENSIONS)


def _is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# creates an object of tarinfo from tarfile
def indextar(dbtarfile, path_prefix=''):
    tar_file_obj = TarFileMetadata(dbtarfile)
    with tarfile.open(dbtarfile, 'r|') as db:
        counter = 0
        for tarinfo in db:
            if _is_image_file(tarinfo.name):
                # Only consider filenames with path_prefix
                if not path_prefix or path_prefix in tarinfo.name:
                    data_file_obj = DataFileMetadata(tarinfo.name,
                                                     tarinfo.offset_data,
                                                     tarinfo.size)
                    tar_file_obj.add_file(data_file_obj)
            counter += 1
            if counter % 1000 == 0:
                # tarfile object maintains a list of members of the archive,
                # and keeps this updated whenever you read or write members
                # free ram by clearing the member list...
                db.members = []

    if len(tar_file_obj) == 0:
        print('No file with {} prefix found in {}'.format(path_prefix, dbtarfile))
        return None

    return tar_file_obj


def gen_tar_index(tar_path, index_outdir, path_prefix='', verbose=False):
    # assertions
    assert isinstance(tar_path, str), \
        "tar_path \'{}\'should be of str type".format(tar_path)
    assert (isdir(tar_path) or isfile(tar_path)), \
        "No tar file(s) exist at path {}".format(tar_path)
    assert isinstance(index_outdir, str), \
        "Expected index_outdir to be of str type"

    # Collect name of all tar files at given path
    tarfiles = []
    if (isfile(tar_path) and
            _is_tar_file(tar_path)):
        tarfiles.append(tar_path)
        outfile = str(basename(tar_path).split('.')[0] + '.index')
    elif isdir(tar_path):
        for f in listdir(tar_path):
            if _is_tar_file(f) and isfile(join(tar_path, f)):
                tarfiles.append(join(tar_path, f))
        outfile = str(basename(tar_path) + '.index')

    if len(tarfiles) == 0:
        raise(RuntimeError("No tarfile found at the given path"))

    # Read all tar file indices in a single object
    tar_dir_obj = TarDirMetaData(tar_path)
    count = {}
    for idx, tar in enumerate(tarfiles):
        tar_file_obj = indextar(tar, path_prefix)
        count[tar] = len(tar_file_obj) if tar_file_obj else 0
        if tar_file_obj:
            tar_dir_obj.add_file(tar_file_obj)
            sys.stdout.write("\r%0.2f%%" % ((float(idx) * 100) / len(tarfiles)))
            sys.stdout.flush()
    sys.stdout.write("\n")
    if verbose:
        print("Number of files for each tarfile:")
        print(count)

    outfile = join(index_outdir, outfile)

    # Save tar index object
    f = open(outfile, 'wb')
    pickle.dump(tar_dir_obj, f)
    f.close()

    print('Saved tar index object in file {}'.format(outfile))
    return outfile


def get_tar_index_files(tar_dir, index_dir, tar_ext='.tar.gz'):
    assert tar_dir is not None
    assert index_dir is not None
    tarfiles = []
    indexfiles = []
    for f in listdir(tar_dir):
        if (_is_tar_file(f) and isfile(join(tar_dir, f)) and
                isfile(join(index_dir, f.replace(tar_ext, '.index')))):
            tarfiles.append(join(tar_dir, f))
            indexfiles.append(join(index_dir, f.replace('.tar.gz', '.index')))
    return tarfiles, indexfiles


def tar_lookup(tarfile, datafile_metadata):
    assert _is_tar_file(tarfile)
    assert isinstance(datafile_metadata, DataFileMetadata)
    with open(tarfile, 'r') as tar:
        tar.seek(int(datafile_metadata.offset))
        buffer = tar.read(int(datafile_metadata.size))
        return buffer


def make_dataset(tar_file_obj, class_to_idx, path_prefix=''):
    assert isinstance(tar_file_obj, TarFileMetadata)
    assert isinstance(path_prefix, str)
    images = []
    members_name = []
    members = {}
    for idx in range(len(tar_file_obj)):
        data_file = tar_file_obj[idx]
        filename, offset, size = data_file.get_metadata()
        if _is_image_file(filename):
            if not path_prefix or (path_prefix in filename and path_prefix != filename):
                members_name.append(filename)
                members[filename] = data_file
    members_name = sorted(members_name)
    for member_name in members_name:
        # item = (DataFileMetadata, class index)
        item = (members[member_name],
                class_to_idx[extract_classname_from_member(member_name)])
        images.append(item)

    return images


def default_loader(tarfile_path, datafile_metadata):
    assert isinstance(tarfile_path, str)
    assert isinstance(datafile_metadata, DataFileMetadata)
    buffer = tar_lookup(tarfile_path, datafile_metadata)
    if py3:
        img = Image.open(StringIO.read(StringIO(buffer)))
    else:
        img = Image.open(StringIO.StringIO(buffer))
    return img.convert('RGB')


def get_classes_from_tar_dir(tar_dir_obj):
    # convert set to list
    classes = list(tar_dir_obj.classes)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx


# reads TarDirMetaData or TarFileMetadata object from a serialized file
# which stores metadata for tar files
def read_tar_index(tar_index_file):
    assert tar_index_file is not None and isfile(tar_index_file)

    # Load tar indices
    f = open(tar_index_file, 'rb')
    tar_dir_obj = pickle.load(f)
    f.close()
    assert (isinstance(tar_dir_obj, TarDirMetaData) or
            isinstance(tar_dir_obj, TarFileMetadata))
    # For only one tar file, wrap it into dir class to maintain consistency
    if isinstance(tar_dir_obj, TarFileMetadata):
        tar_dir_obj = TarDirMetaData(tar_dir_obj)

    return tar_dir_obj


class ImageTarFile(data.Dataset):
    """A data loader where the images are tarred and arranged in this way:

        root/prefix_for_image_type/class_name/xxx.png

        root/tvm/dog/xxx.png
        root/tvm/dog/xxy.png
        root/tvm/dog/xxz.png

        root/quilting/cat/123.png
        root/quilting/cat/nsdf3.png
        root/quilting/cat/asd932_.png

    Args:
        tar_index_file (string): Path for TarDirMetadata/TarFileMetadata object
        path_prefix (string): path prefix in all tar files (default="")
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (tar member_name, taget class index) tuples
    """

    def __init__(self, tar_index_file, path_prefix='', transform=None,
                 target_transform=None, loader=default_loader):

        assert isinstance(tar_index_file, str), \
            "Expect tar_index_file to be of str type"
        assert isinstance(path_prefix, str), \
            "Expect path_prefix to be of str type"

        tar_dir_obj = read_tar_index(tar_index_file)
        classes, class_to_idx = get_classes_from_tar_dir(tar_dir_obj)
        # get image indexes:
        imgs, tar_file_list, img2tarfile = [], [], []
        for idx in range(len(tar_dir_obj)):
            tar_file = tar_dir_obj[idx]
            _imgs = make_dataset(tar_file, class_to_idx, path_prefix)
            imgs += _imgs  # NOTE: Does this need to be sorted again by target?
            img2tarfile += [idx] * len(_imgs)
            tar_file_list.append(tar_file.tarfile)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in " + str(len(tar_dir_obj.tarfies)) +
                                " TAR files.\n" +
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        # store some fields:
        self.tar_dir_obj = tar_dir_obj
        self.img2tarfile = img2tarfile
        self.tar_file_list = tar_file_list
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        data_file_obj, target = self.imgs[index]
        tar_file = self.tar_file_list[self.img2tarfile[index]]
        img = self.loader(tar_file, data_file_obj)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_class_from_index(self, index):
        assert index < len(self.classes), "index can't be greater than numer of classes"
        return self.classes[index]

    def get_classes(self):
        return self.classes
