# HW 1 Worksheet

---

This is the worksheet for Homework 1. Your deliverables for this homework are:

- [ ] This worksheet with all answers filled in. If you include plots/images, be sure to include all the required files. Alternatively, you can export it as a PDF and it will be self-sufficient.
- [ ] Kaggle submission and writeup (details below)
- [ ] Github repo with all of your code! You need to either fork it or just copy the code over to your repo. A simple way of doing this is provided below. Include the link to your repo below. If you would like to make the repo private, please dm us and we'll send you the GitHub usernames to add as collaborators.

[`MY GITHUB REPO HERE`](https://github.com/jaansiparsa/model-zhou.git)

## To move to your own repo:

First follow `README.md` to clone the code. Additionally, create an empty repo on GitHub that you want to use for your code. Then run the following commands:

```bash
$ git remote rename origin staff # staff is now the provided repo
$ git remote add origin <your repos remote url>
$ git push -u origin main
```



# Part -1: PyTorch review

Feel free to ask your NMEP friends if you don't know!

## -1.0 What is the difference between `torch.nn.Module` and `torch.nn.functional`?

All custom models are made by subclassing nn.Module; it provides functionality for definining layers, weights, activation functions, etc. (all the stuff you'd typically need for neural network modules) in Pytorch. A model you'd make with nn.Module would have a forward method and inside that forward method you'd be able to use functions contained in torch.nn.funcional (ReLU, max pooling, etc). 

## -1.1 What is the difference between a Dataset and a DataLoader?

A Dataset is the actual data points/their labels, while a DataLoader is how you batch and shuffle the data.

## -1.2 What does `@torch.no_grad()` above a function header do?

In training you want to do backprop so you keep gradients on but in predictions/inference you don't want to change any weights so you turn the gradients off by signaling @torch.no_grad() 


# Part 0: Understanding the codebase

Read through `README.md` and follow the steps to understand how the repo is structured.

## 0.0 What are the `build.py` files? Why do we have them?

The build.py files have all the cofigs used in the actual code. This makes it so that any time you want to change the configs you don't have to change the core code and can just change the build.py files. It also makes it so that you can experiment with different configs (through different build files).

## 0.1 Where would you define a new model?

Files in the models folder (e.g. lenet.py and resnet.py)

## 0.2 How would you add support for a new dataset? What files would you need to change?

Classes within datasets.py

## 0.3 Where is the actual training code?

main.py

## 0.4 Create a diagram explaining the structure of `main.py` and the entire code repo.

Be sure to include the 4 main functions in it (`main`, `train_one_epoch`, `validate`, `evaluate`) and how they interact with each other. Also explain where the other files are used. No need to dive too deep into any part of the code for now, the following parts will do deeper dives into each part of the code. For now, read the code just enough to understand how the pieces come together, not necessarily the specifics. You can use any tool to create the diagram (e.g. just explain it in nice markdown, draw it on paper and take a picture, use draw.io, excalidraw, etc.)

`YOUR ANSWER HERE`

## main.py
### parse_option():
Specifies training settings for the model.
The build and config files also help set up the training environment in both main() and parse()

### train_one_epoch():
Trains the model and updates weights!
Training data comes from the dataset in the datasets.py file/data folder.
This function also uses the optimizer in optimizer.py

### validate():
Calculates loss and accuracy
Train and validate are used together during training. One epoch includes both training and validation.

### evaluate()
Runs on test data for final model predictions.

### main()
Outlines the entire training process
Uses train, validate, and evaluate functions
Also uses configs and dataloaders to create the training and validation datasets.


# Part 1: Datasets

The following questions relate to `data/build.py` and `data/datasets.py`.

## 1.0 Builder/General

### 1.0.0 What does `build_loader` do?

It prepares the training, validation, and test datasets to be used in training/testing based on the configs you give it.

### 1.0.1 What functions do you need to implement for a PyTorch Datset? (hint there are 3)

get_item, len, and transforms

## 1.1 CIFAR10Dataset

### 1.1.0 Go through the constructor. What field actually contains the data? Do we need to download it ahead of time?

self.dataset stores the data. You can set download=True and it downloads.


### 1.1.1 What is `self.train`? What is `self.transform`?

self.train is True/False for whether or not the data is training data
self.transform is the preprocessing/augmentations of the images

### 1.1.2 What does `__getitem__` do? What is `index`?

getitem gets the image and corresponding labels at a given index

### 1.1.3 What does `__len__` do?

len returns the number of samples in the dataset

### 1.1.4 What does `self._get_transforms` do? Why is there an if statement?

If the data is training data, it applies an augmentation

### 1.1.5 What does `transforms.Normalize` do? What do the parameters mean? (hint: take a look here: https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html)

It normalizes the data to the mean and standard deviation in the parameters

## 1.2 MediumImagenetHDF5Dataset

### 1.2.0 Go through the constructor. What field actually contains the data? Where is the data actually stored on honeydew? What other files are stored in that folder on honeydew? How large are they?

self.file contains the data. Other files such as cifar-10 and tiny-imagenet data are stored in the data folder in honeydew

> *Some background*: HDF5 is a file format that stores data in a hierarchical structure. It is similar to a python dictionary. The files are binary and are generally really efficient to use. Additionally, `h5py.File()` does not actually read the entire file contents into memory. Instead, it only reads the data when you access it (as in `__getitem__`). You can learn more about [hdf5 here](https://portal.hdfgroup.org/display/HDF5/HDF5) and [h5py here](https://www.h5py.org/).

### 1.2.1 How is `_get_transforms` different from the one in CIFAR10Dataset?

_get_transforms in the CIFAR dataset normalizes and resizes the images as well, but the one in MediumImageNet doesn't.

### 1.2.2 How is `__getitem__` different from the one in CIFAR10Dataset? How many data splits do we have now? Is it different from CIFAR10? Do we have labels/annotations for the test set?

Medium Image Net doesn't have labels for each image during testing, and only labels images with a -1 then. But it still splits the data with train/val/test.

### 1.2.3 Visualizing the dataset

Visualize ~10 or so examples from the dataset. There's many ways to do it - you can make a separate little script that loads the datasets and displays some images, or you can update the existing code to display the images where it's already easy to load them. In either case, you can use use `matplotlib` or `PIL` or `opencv` to display/save the images. Alternatively you can also use `torchvision.utils.make_grid` to display multiple images at once and use `torchvision.utils.save_image` to save the images to disk.

Be sure to also get the class names. You might notice that we don't have them loaded anywhere in the repo - feel free to fix it or just hack it together for now, the class names are in a file in the same folder as the hdf5 dataset.

[link to image](https://github.com/jaansiparsa/model-zhou/blob/main/data/output_image2.png)



# Part 2: Models

The following questions relate to `models/build.py` and `models/models.py`.

## What models are implemented for you?

lenet and resnet

## What do PyTorch models inherit from? What functions do we need to implement for a PyTorch Model? (hint there are 2)

They inherit from torch.nn.Module. They need to implement a __init__ function and a forward() function.


## How many layers does our implementation of LeNet have? How many parameters does it have? (hint: to count the number of parameters, you might want to run the code)
6 layers in features + 5 layers in classifier = 11 total layers
99276 total params



# Part 3: Training

The following questions relate to `main.py`, and the configs in `configs/`.

## 3.0 What configs have we provided for you? What models and datasets do they train on?

lenet_base - trains on CIFAR10
resnet_base - trains on CIFAR10
resnet_medium - trains on medium_imagenet

## 3.1 Open `main.py` and go through `main()`. In bullet points, explain what the function does.
I'm pretty sure I already explained this but ok
- gets configs 
- splits data into training/val/testing
- trains: train_one_epoch() and validate()
- finally evaluates on test data

## 3.2 Go through `validate()` and `evaluate()`. What do they do? How are they different? 
> Could we have done better by reusing code? Yes. Yes we could have but we didn't... sorry... >

validate() calculates loss to update weights as part of training process (using training data)
evaluate() also calculates accuracy but doesn't update weights because it's own test data`

# Part 4: AlexNet

## Implement AlexNet. Feel free to use the provided LeNet as a template. For convenience, here are the parameters for AlexNet:

```
Input NxNx3 # For CIFAR 10, you can set img_size to 70
Conv 11x11, 64 filters, stride 4, padding 2
MaxPool 3x3, stride 2
Conv 5x5, 192 filters, padding 2
MaxPool 3x3, stride 2
Conv 3x3, 384 filters, padding 1
Conv 3x3, 256 filters, padding 1
Conv 3x3, 256 filters, padding 1
MaxPool 3x3, stride 2
nn.AdaptiveAvgPool2d((6, 6)) # https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
flatten into a vector of length x # what is x?
Dropout 0.5
Linear with 4096 output units
Dropout 0.5
Linear with 4096 output units
Linear with num_classes output units
```

> ReLU activation after every Conv and Linear layer. DO **NOT** Forget to add activatioons after every layer. Do not apply activation after the last layer.

## 4.1 How many parameters does AlexNet have? How does it compare to LeNet? With the same batch size, how much memory do LeNet and AlexNet take up while training? 
> (hint: use `gpuststat`)

`Alexnet: 57.82M params, 1456MB memory. Lenet: 99.28K params, 272MB memory`

## 4.2 Train AlexNet on CIFAR10. What accuracy do you get?

Report training and validation accuracy on AlexNet and LeNet. Report hyperparameters for both models (learning rate, batch size, optimizer, etc.). We get ~77% validation with AlexNet.

> You can just copy the config file, don't need to write it all out again. >
> Also no need to tune the models much, you'll do it in the next part. >

Validation accuracies: Alexnet max: 77.35%, Lenet max: 67.69
Training accuracies: Alexnet max: 99.87, Lenet max: 73.83

Alexnet:
AUG:
  COLOR_JITTER: 0.4
DATA:
  BATCH_SIZE: 256
  DATASET: "cifar10"
  IMG_SIZE: 70
  NUM_WORKERS: 32
  PIN_MEMORY: True
MODEL:
  NAME: alexnet
  NUM_CLASSES: 200
  DROP_RATE: 0.0
TRAIN:
  EPOCHS: 50
  WARMUP_EPOCHS: 10
  LR: 3e-4
  MIN_LR: 3e-5
  WARMUP_LR: 3e-5
  LR_SCHEDULER:
    NAME: "cosine"
  OPTIMIZER:
    NAME: "adamw"
    EPS: 1e-8
    BETAS: (0.9, 0.999)
    MOMENTUM: 0.9
OUTPUT: "output/alexnet_cifar"
SAVE_FREQ: 5
PRINT_FREQ: 99999
PRINT_FREQ: 99999

Lenet: same as configs in configs file


# Part 5: Weights and Biases

> Parts 5 and 6 are independent. Feel free to attempt them in any order you want.

> Background on W&B. W&B is a tool for tracking experiments. You can set up experiments and track metrics, hyperparameters, and even images. It's really neat and we highly recommend it. You can learn more about it [here](https://wandb.ai/site).
> 
> For this HW you have to use W&B. The next couple parts should be fairly easy if you setup logging for configs (hyperparameters) and for loss/accuracy. For a quick tutorial on how to use it, check out [this quickstart](https://docs.wandb.ai/quickstart). We will also cover it at HW party at some point this week if you need help.

## 5.0 Setup plotting for training and validation accuracy and loss curves. Plot a point every epoch.

`code is pushed to my github!`

## 5.1 Plot the training and validation accuracy and loss curves for AlexNet and LeNet. Attach the plot and any observations you have below.

q5base.png images folder
Alexnet had much higher accuracies and steeper curves

## 5.2 For just AlexNet, vary the learning rate by factors of 3ish or 10 (ie if it's 3e-4 also try 1e-4, 1e-3, 3e-3, etc) and plot all the loss plots on the same graph. What do you observe? What is the best learning rate? Try at least 4 different learning rates.

result in images/lr_comparison
3e-4: red
1e-4: pink
1e-3: mint green
3e-3: orange

The bigger the learning rate, the worse the accuracies tended to be. Also, the bigger the learning rate, the quicker the loss would just plateau.

## 5.3 Do the same with batch size, keeping learning rate and everything else fixed. Ideally the batch size should be a power of 2, but try some odd batch sizes as well. What do you observe? Record training times and loss/accuracy plots for each batch size (should be easy with W&B). Try at least 4 different batch sizes.

Comparison graphs in images/batch_comparisons
also the red is 256 batch size and the pink is 128 batch size, green is 420, yellow is 512
The lower the batch size, the higher the val loss (typically) 
The odd batch size (not a power of 2) had the lowest val loss
All converged around the same accuracy but 256 batch size did the best

## 5.4 As a followup to the previous question, we're going to explore the effect of batch size on _throughput_, which is the number of images/sec that our model can process. You can find this by taking the batch size and dividing by the time per epoch. Plot the throughput for batch sizes of powers of 2, i.e. 1, 2, 4, ..., until you reach CUDA OOM. What is the largest batch size you can support? What trends do you observe, and why might this be the case?
You only need to observe the training for ~ 5 epochs to average out the noise in training times; don't train to completion for this question! We're only asking about the time taken. If you're curious for a more in-depth explanation, feel free to read [this intro](https://horace.io/brrr_intro.html). 

tried starting at 1 but it kept taking outrageously long and then would crash out so then 
I just started at 32 and then went until 16384 (2^14) OUT OF MEMORY ERROR YAAYAAYY FINALLY
basically throughput increased as batch size increased
wasnt like directly proportional to batch size but some sort of positive relationship
image of graph is in images/ihatedthis
if i have to type python main.py --cfg configs/alexnet.yaml again i will scream

## 5.5 Try different data augmentations. Take a look [here](https://pytorch.org/vision/stable/transforms.html) for torchvision augmentations. Try at least 2 new augmentation schemes. Record loss/accuracy curves and best accuracies on validation/train set.

augmentation code in datasets.py
image of comparison in images/augmentations
The original augmentation had the best results.
I tried a new augmentation where it did a random vertical flip rather than horizontal flip and added RandomPerspective() which had lower training accuracy and validation accuracy. This makes sense since objects look really different vertically from normal and typically look the same horizontally reflected. 
I also tried only having a random vertical flip and this had high training accuracy and high val loss but really low val accuracy so it definitely was overfitting

## 5.6 (optional) Play around with more hyperparameters. I recommend playing around with the optimizer (Adam, SGD, RMSProp, etc), learning rate scheduler (constant, StepLR, ReduceLROnPlateau, etc), weight decay, dropout, activation functions (ReLU, Leaky ReLU, GELU, Swish, etc), etc.

`YOUR ANSWER HERE`



# Part 6: ResNet

## 6.0 Implement and train ResNet18

In `models/*`, we provided some skelly/guiding comments to implement ResNet. Implement it and train it on CIFAR10. Report training and validation curves, hyperparameters, best validation accuracy, and training time as compared to AlexNet. 

`YOUR ANSWER HERE`
Alexnet takes about 7 mins to train
Resnet took 41 mins
all the info is in images/resnet.png

## 6.1 (optional) Visualize examples

Visualize a couple of the predictions on the validation set (20 or so). Be sure to include the ground truth label and the predicted label. You can use `wandb.log()` to log images or also just save them to disc any way you think is easy.

`YOUR ANSWER HERE`


# Part 7: Kaggle submission

To make this more fun, we have scraped an entire new dataset for you! 🎉

We called it MediumImageNet. It contains 1.5M training images, and 190k images for validation and test each. There are 200 classes distributed approximately evenly. The images are available in 224x224 and 96x96 in hdf5 files. The test set labels are not provided :). 

The dataset is downloaded onto honeydew at `/data/medium-imagenet`. Feel free to play around with the files and learn more about the dataset.

For the kaggle competition, you need to train on the 1.5M training images and submit predictions on the 190k test images. You may validate on the validation set but you may not use is as a training set to get better accuracy (aka don't backprop on it). The test set labels are not provided. You can submit up to 10 times a day (hint: that's a lot).

Your Kaggle scores should approximately match your validation scores. If they do not, something is wrong.

(Soon) when you run the training script, it will output a file called `submission.csv`. This is the file you need to submit to Kaggle. You're required to submit at least once. 

## Kaggle writeup

We don't expect anything fancy here. Just a brief summary of what you did, what worked, what didn't, and what you learned. If you want to include any plots, feel free to do so. That's brownie points. Feel free to write it below or attach it in a separate file.

**REQUIREMENT**: Everyone in your group must be able to explain what you did! Even if one person carries (I know, it happens) everyone must still be able to explain what's going on!

Now go play with the models and have some competitive fun! 🎉

Ethan submitted this part!
