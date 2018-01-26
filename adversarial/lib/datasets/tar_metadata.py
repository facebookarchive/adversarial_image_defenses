from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def extract_classname_from_member(member_name):
    # expects filename format as "root/class_name/img.ext"
    class_name = str(member_name.split('/')[-2])
    return class_name


# Handles a directory of tar files
class TarDirMetaData(object):

    def __init__(self, tar_dir):
        assert isinstance(tar_dir, str)
        self.tar_dir = tar_dir
        self.tarfiles = []
        self.classes = set()

    def __len__(self):
        return len(self.tarfiles)

    def add_file(self, tarfile):
        assert isinstance(tarfile, TarFileMetadata)
        self.tarfiles.append(tarfile)
        for class_name in tarfile.classes:
            if class_name not in self.classes:
                self.classes.add(class_name)

    def __getitem__(self, index):
        assert index < len(self.tarfiles)
        return self.tarfiles[index]


# Handles a tarfile with data files
class TarFileMetadata(object):

    def __init__(self, tarfile):
        assert isinstance(tarfile, str)
        self.tarfile = tarfile
        self.files = []
        self.classes = set()

    def __len__(self):
        return len(self.files)

    def add_file(self, file_metadata):
        assert isinstance(file_metadata, DataFileMetadata)
        self.files.append(file_metadata)
        filename = file_metadata.filename
        class_name = extract_classname_from_member(filename)
        if class_name not in self.classes:
            self.classes.add(class_name)

    def __getitem__(self, index):
        assert index < len(self.files)
        return self.files[index]


# Handles a data file inside a tarfile
class DataFileMetadata(object):
    def __init__(self, filename, offset, size):
        self.filename = filename
        self.offset = offset
        self.size = size

    def get_metadata(self):
        return self.filename, self.offset, self.size
