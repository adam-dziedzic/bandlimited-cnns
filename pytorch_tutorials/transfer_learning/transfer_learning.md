# Transfer Learning

In practice, very few people train an entire 
Convolutional Network from scratch (with random 
initialization), because it is relatively rare to have a dataset of sufficient size. 
Instead, it is common to pretrain a ConvNet on a very large dataset (e.g. ImageNet, 
which contains 1.2 million images with 1000 categories), and then use the ConvNet either as an 
initialization or a fixed feature extractor for the task of interest. The three major Transfer 
Learning scenarios look as follows:
1. 