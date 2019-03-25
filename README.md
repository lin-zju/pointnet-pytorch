# PointNet Pytorch

This is a pytorch implementation of the original [PointNet](http://stanford.edu/~rqi/pointnet/).

## Data preparation

You will need to download the [ModelNet40](http://modelnet.cs.princeton.edu/) dataset under the directory `data/`. The data set should be named `modelnet40_ply_hdf5_2048`

## Training and Testing

To train the network, run

```
python run.py
```

After the training is done, run

```
python run.py --test
```

to obtain classification accuracy for the 40 classes.
