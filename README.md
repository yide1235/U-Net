# U-Net
## implementation of U-Net
Implement the U-net architecture for cell image data segmentation using PyTorch.
![](./img/UNet_arch.png)
Figure 1: U-Net architecture[1]

#### Data augmentation:
Since the size of the data is too small for training a neural network with a huge number of
parameters. Under this situation, the code has the following data augmentation applied:
1. Horizontal/Vertical flip
2. Zooming
3. Rotation

#### Final Results:
![](./img/1.png)
![](./img/2.png)
![](./img/3.png)
![](./img/4.png)
I used lr=0.01 SGD(momentum=0.8, weight_decay=0.0005) 600 epoch for training this network. I am
using my laptop rtx3070 gpu, used for one hour to train 600 epochs. I tried a lot of parameters, among all my parameters fine-tuning, this is my best result. However, I found two problems in this assignment, 1. For some pictures, the mask only marked one of several cells in the picture, so once the network had trained, it recognized all the cells, which is a better result. 2. For cross entropy loss, the test loss is not going down, even if the training loss is going down perfectly.

#### References
[1]Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for
biomedical image segmentation. InInternational Conference on Medical image computing and
computer-assisted intervention, pages 234–241. Springer, 2015.
