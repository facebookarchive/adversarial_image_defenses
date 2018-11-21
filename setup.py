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

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import subprocess
import zipfile
import sys
if sys.version_info[0] == 3:  # for python3
    import urllib.request as urllib
    py3 = True
else:  # for python2
    import urllib as urllib
    py3 = False

ext_code_download = True

readme = open('README.md').read()

TRANSFORMATION_DIR = str("adversarial/lib/transformations")
cwd = os.path.dirname(os.path.abspath(__file__))

# Download 3rd party code (Respective licenses are applicable)
if ext_code_download:
    # Download inception_v4 and inceptionresnetv2
    MODEL_DIR = str("adversarial/lib/models")
    INCEPTION_V4_URL = ("https://raw.githubusercontent.com/Cadene/"
                        "tensorflow-model-zoo.torch/"
                        "f43005c4b4cdd745e9788b22e182c91453c54daf/inceptionv4"
                        "/pytorch_load.py")
    INCEPTION_RESNET_V2_URL = ("https://raw.githubusercontent.com/Cadene/"
                               "tensorflow-model-zoo.torch/"
                               "f43005c4b4cdd745e9788b22e182c91453c54daf/"
                               "inceptionresnetv2/pytorch_load.py")
    urlopener = urllib.URLopener()
    urlopener.retrieve(INCEPTION_V4_URL, os.path.join(MODEL_DIR, "inceptionv4.py"))
    urlopener.retrieve(INCEPTION_RESNET_V2_URL,
                       os.path.join(MODEL_DIR, "inceptionresnetv2.py"))

    # Download denoising code for tv_bregman from scikit-image
    DENOISE_URL = ("https://raw.githubusercontent.com/scikit-image/scikit-image/"
                   "902a9a68add274c4125a358b29e3263b9d94f686/skimage/"
                   "restoration/_denoise_cy.pyx")
    urlopener = urllib.URLopener()
    urlopener.retrieve(DENOISE_URL, os.path.join(TRANSFORMATION_DIR, "_denoise_cy.pyx"))
    # Apply patch to support TVM compressed sensing
    # _tv_bregman.patch was created from commit 902a9a6
    # would need to be updated if the source gets updated
    cmd = ("(cd adversarial/lib/transformations && patch -p0 -o tv_bregman.pyx) "
            "< adversarial/lib/transformations/_tv_bregman.patch")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    process.communicate()

    # Download and unzip the code for maxflow
    MAXFLOW_URL = "http://mouse.cs.uwaterloo.ca/code/maxflow-v3.01.zip"
    urlopener = urllib.URLopener()
    maxflow_file = os.path.join(TRANSFORMATION_DIR, "maxflow.zip")
    urlopener.retrieve(MAXFLOW_URL, maxflow_file)
    zip_ref = zipfile.ZipFile(maxflow_file, 'r')
    zip_ref.extractall(TRANSFORMATION_DIR)
    zip_ref.close()

# Create Extension to build quilting code
quilting_c_src = [
    str(os.path.join(TRANSFORMATION_DIR, 'graph.cpp')),
    str(os.path.join(TRANSFORMATION_DIR, 'maxflow.cpp')),
    str(os.path.join(TRANSFORMATION_DIR, 'quilting.cpp')),
    str(os.path.join(TRANSFORMATION_DIR, 'findseam.cpp'))
]

include_dirs = [
    cwd,
    os.path.join(cwd, TRANSFORMATION_DIR),
]
library_dirs = [os.path.join(cwd, 'adversarial', 'lib')]
c_compile_args = [str('-std=c++11')]

extensions = []
quilting_ext = Extension(str("libquilting"),
                         sources=quilting_c_src,
                         language='c++',
                         include_dirs=include_dirs,
                         library_dirs=library_dirs,
                         extra_compile_args=c_compile_args)
extensions = [quilting_ext]

# Create Extension to build cython code for TVM
cython_ext = Extension(str('tv_bregman'),
                       sources=[str("adversarial/lib/transformations/tv_bregman.pyx")],
                       )
extensions.append(cython_ext)

requirements = ['pillow', 'torchvision', 'scipy', 'scikit-image',
                'Cython', 'enum34']
if py3:
    requirements += ['progressbar33']
else:
    requirements += ['progressbar']


# cython Extension needs numpy include dir path
# this will be called only after installing numpy from setup_requires
class CustomBuildExt(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


setup(
    # Metadata
    name="adversarial",
    version="0.1.0",
    author="Mayank Rana",
    author_email="mayankrana@fb.com",
    url="https://github.com/facebookresearch/adversarial_image_defenses",
    description="Code for Countering Adversarial Images using Input Transformations",
    long_description=readme,
    license='CC-BY-4.0',

    # Package Info
    packages=['adversarial'],
    package_dir={'adversarial': 'adversarial'},
    include_package_data=True,
    zip_safe=False,
    install_requires=requirements,
    setup_requires=['setuptools>=18.0', 'cython', 'numpy'],
    cmdclass={'build_ext': CustomBuildExt},
    ext_package='adversarial.lib.transformations',
    ext_modules=extensions,
)
