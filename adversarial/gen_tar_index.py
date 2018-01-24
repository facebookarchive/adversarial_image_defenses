# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
from lib.datasets.tarfolder import gen_tar_index
import sys
import os


def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate tar indices')
    parser.add_argument('--tar_path',
                        default=None,
                        type=str, metavar='N',
                        help='Path for tar file or directory')
    parser.add_argument('--index_root',
                        default=None,
                        type=str, metavar='N',
                        help='Directory path to store tar index object')
    parser.add_argument('--path_prefix',
                        default='',
                        type=str, metavar='N',
                        help='prefix in member name')

    args = parser.parse_args(args)
    return args


def generate_tar_index(args):
    assert args.tar_path is not None
    assert args.index_root is not None
    if not os.path.isdir(args.index_root):
        os.mkdir(args.index_root)

    gen_tar_index(args.tar_path, args.index_root, args.path_prefix)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    generate_tar_index(args)
