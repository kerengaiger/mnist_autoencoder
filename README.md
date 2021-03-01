# mnist_autoencoder
AutoEncoder implementation in Pytorch trained on the MNIST data.


## Usage
### Calling the code
#### train
Train an AutoEncoder on MNIST data, calculate test error and save trained model

##### Running example
```sh
python train.py --lr 0.001 --alpha 0.5 --batch_size 32 --epochs 10 --noise_var 0.5 --valid_split 0.22 --loss bce --plot_images
-- plot_kernel --save_dir ./outputs/ --model_file 'mnist_autoencoder.pt
```
#### inference
Load the trained model, sample few images from dataset and reduce their generated noise and generate new images. 

##### Running example
```sh
python inference.py ---model_file mnist_autoencoder.pt --save_dir ./outputs/ --noise_var 0.5 --imgs_to_gen 3
--imgs_to_denoise 3
```

## Installation

Clone the repository:

```sh
git clone git@gitlab.com:vulcans/vulns_sols_matching.git
```

## Contributing

Package main contributer is Keren Gaiger: keren.gaiger@gmail.com


