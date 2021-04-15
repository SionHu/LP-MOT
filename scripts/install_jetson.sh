#!/bin/bash

# Jetpack 4.4 (OpenCV, CUDA, TensorRT) is required before running this script
DIR=$HOME

set -e

# set up environment variables
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc 
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc 
source ~/.bashrc

# install pip, numpy, pycuda, tensorflow, cython-bbox
sudo apt-get update
sudo apt-get install python3-pip libhdf5-serial-dev hdf5-tools libcanberra-gtk-module
sudo -H pip3 install cython
sudo -H pip3 install numpy pycuda cython-bbox
sudo -H pip3 install --no-cache-dir --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow==1.15.2+nv20.4

# install scipy
sudo apt-get install libatlas-base-dev gfortran
sudo -H pip3 install scipy==1.5.0

# install llvm (This may take a while)
cd $DIR
wget http://releases.llvm.org/7.0.1/llvm-7.0.1.src.tar.xz
tar -xvf llvm-7.0.1.src.tar.xz
cd llvm-7.0.1.src
mkdir llvm_build_dir
cd llvm_build_dir/
cmake ../ -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="ARM;X86;AArch64"
make -j4
sudo make install
cd bin/
echo "export LLVM_CONFIG=\""`pwd`"/llvm-config\"" >> ~/.bashrc
echo "alias llvm='"`pwd`"/llvm-lit'" >> ~/.bashrc
source ~/.bashrc
sudo -H pip3 install llvmlite==0.31.0

# install numba
sudo -H pip3 install numba==0.48
