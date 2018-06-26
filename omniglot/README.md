# Omniglot experiment

This code performs the Omniglot task (fast learning of image-label mappings).

To run this code, you must download [the Python version of the Omniglot dataset](https://github.com/brendenlake/omniglot), and move the `omniglot-master` directory inside this directory. You will also need the scikit-image library (in addition to PyTorch).

To reproduce the results shown in the paper:

```
python3 omniglot.py --nbclasses 5  --nbiter 5000000 --rule oja --activ tanh --steplr 1000000 --prestime 1 --prestimetest 1 --gamma .666 --alpha free --lr 3e-5 

```
