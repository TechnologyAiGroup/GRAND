pip install networkx==2.5.1
pip install numpy==1.21.2
pip install scikit-learn==1.2.2
# For CUDA 11.3
pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
wget -nc https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_sparse-0.6.13-cp39-cp39-linux_x86_64.whl
wget -nc https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
wget -nc https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_spline_conv-1.2.1-cp39-cp39-linux_x86_64.whl
wget -nc https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_cluster-1.6.0-cp39-cp39-linux_x86_64.whl
# pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
# pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
# pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
# pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.11.0+cu113.html
pip install torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
pip install torch_sparse-0.6.13-cp39-cp39-linux_x86_64.whl
pip install torch_cluster-1.6.0-cp39-cp39-linux_x86_64.whl
pip install torch_spline_conv-1.2.1-cp39-cp39-linux_x86_64.whl
pip install torch-geometric==2.0.4
