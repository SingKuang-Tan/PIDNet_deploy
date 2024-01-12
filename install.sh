# update numpy 
pip3 install numpy==1.19.4
pip3 install pillow
pip3 install yacs

# install torch
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get -y install libopenblas-base libopenmpi-dev libomp-dev
pip3 install Cython
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
rm torch-1.10.0-cp36-cp36m-linux_aarch64.whl

#install torchvision
sudo apt-get -y install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
cd ~/
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.11.1
python3 setup.py install --user

# get PIDNet
git clone https://github.com/Monarch-Tractor/PIDNet.git

#install bazel
sudo apt-get -y install openjdk-11-jdk
cd ~/
mkdir bazel
cd bazel
curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/5.1.1/bazel-5.1.1-dist.zip
unzip bazel-5.1.1-dist.zip
bash ./compile.sh
sudo cp output/bazel /usr/local/bin/

#install torch-tensorrt
cd ~/
git clone --branch v1.0.0 https://github.com/pytorch/TensorRT.git
export BUILD_VERSION=1.0.0
cd TensorRT
cp ~/PIDNet/WORKSPACE ./
cd py
sudo python3 setup.py install --use-cxx11-abi