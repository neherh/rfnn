## Recurrent Fusion Neural Network

### Intro

This project utilizes a recurrent fusion neural network (RFNN) in order to improve lane detection. To do so, the neural network improves upon an scnn network by adding a fusion and temporal portion of the network in which noise is filtered which produces a finer prediction.

### Implementation

1. Preform preprocessing by converting Labeled data (valid) of points to images by running label2image.py. Image size is same size as the raw images.
    ```Shell
    python label2image.py
    ```

2. Run PyTorch Code


____________________________________________________________________________________________________________________
## Acknowledgements

Inspiration and network derivations stemmed from: 

Xingang Pan, Jianping Shi, Ping Luo, Xiaogang Wang, Xiaoou Tang. ["Spatial As Deep: Spatial CNN for Traffic Scene Understanding"](https://arxiv.org/abs/1712.06080), AAAI2018
