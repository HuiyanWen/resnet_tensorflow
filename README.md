## Generate dataset
I've uploaded the partial gnt dataset, you can transfer it to any form as you need. Here, you can use resnet_tensorflow1.12/tfrecord_op/gnt2tfrecord.py to convert gnt to tfrecord within one minutes. After activating tensorflow-gpu, you can print python gnt2tfrecord.py -h to get the helpful information about parameters.

When you want to generate the training set, it's matter that to set the shuffle=1, or the network will be hard to converge.

![Image text](https://github.com/HuiyanWen/resnet_tensorflow/blob/master/imgs/2.png)

If you are in Windows, Ubuntu or other systems which have the GUI, you can verify tfrecords by printing the images and theirs label. The python document involved is resnet_tensorflow1.12/tfrecord_op/test_tfrecord.py.

![Image text](https://github.com/HuiyanWen/resnet_tensorflow/blob/master/imgs/3.png)
## Train, Validate and Inference
All of the three sections were wrote in ./main.py, and you can print python main.py -h to get the helpful information.

#### How to start training
```
python  main.py --mode train 
```
![Image text](https://github.com/HuiyanWen/resnet_tensorflow/blob/master/imgs/4.png)
#### How to start validation
```
python  main.py --mode validation 
```
#### How to start inference
Here, it needs you have the GUI in order that you can install pillow.
```
python  main.py --mode inference --png_dir xxx 
```
![Image text](https://github.com/HuiyanWen/resnet_tensorflow/blob/master/imgs/7.png)
![Image text](https://github.com/HuiyanWen/resnet_tensorflow/blob/master/imgs/6.png)
## How to use tensorboard
```
tensorboard –logdir=./log3755/train
```
![Image text](https://github.com/HuiyanWen/resnet_tensorflow/blob/master/imgs/8.png)
![Image text](https://github.com/HuiyanWen/resnet_tensorflow/blob/master/imgs/9.png)
## How to optimize the parameters
Firstly, you can change the learning_rate, decay_step, optimizer in ./main.py, I suggested that you can set them to 0.0001, 0.9, adam.
Secondly, weight_decay, batch_norm_decay, batch_norm_epsilon in ./model/resnet_utils.py also can influence the accuracy.

## Get the checkpoints and images
https://pan.baidu.com/s/1AnRdHv_ff-513OI5n7Cepg code: iiww
