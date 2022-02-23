conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
conda install cffi pandas scipy tqdm matplotlib h5py msgpack-numpy

conda install -c conda-forge -c fvcore -c iopath fvcore iopath
conda install pytorch3d -c pytorch3d

pip install -r requirements.txt
pip install -e .

cd PytorchEMD
python setup.py install
cd ..