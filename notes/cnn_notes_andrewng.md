# CNN Notes

+ Edge detection
  +

## Operations

+ Padding
  + No padding causes shrinking outputs and loss of information from the edges.
  + **same** padding adds pixels around the image such that the output image would have same shape and each pixels is given equal priority during the convolution operation.
  + **valid** padding gives output size of `(n-f+1, n-f+1)` where `(n, n)` is image size and `(f, f)` is filter size.
  + Generally **filter/kernel sizes** are chosen with **odd number** of rows and columns
+ Strided Convolutions
  + Image of size `(n, n)`, when convolved with filter of size `(f, f)`, padding `p` and stride `s`, the output image shape will be
    $$(\lfloor\frac{n+2p-f}{s}+1\rfloor, \lfloor\frac{n+2p-f}{s}+1\rfloor)$$
+ Convolutions over volumes
  + assert number of channels on image == number of channels on filter
  + Filter of shape (f1, f2, f3) convolving on a volume of shape (i1, i2, i3) gives a layer of shape (o1, o2)
  + Generally, these (o1, o2) shaped layers obtained from various filters are stacked depth wise to get the final shape of (o1, o2, num_filters)
  + number of parameters in a conv layer = `num_filters * (f1 x f2 x f3)` where `(f1, f2, f3)` is filter size.
+ Types of layers in convolutional network
  + Convolution
  + Pool
  + Fully Connected

+ Pooling
  + Types of pooling
    + Max pooling
    + Average pooling
  + Why Pooling?
    + Makes the feature learning more robust
  + Hyper parameters
    + Pool size
    + Stride
  + Pooling operations are performed independently on each channel. Hence, the number of channels would be same after pooling operation. That is
    $$(N_H, N_W, N_C) \implies (\lfloor\frac{N_H-f}{s}+1\rfloor, \lfloor\frac{N_W-f}{s}+1\rfloor, N_C)$$

+ Why Convolutions?
  + **Parameter sharing**: Same feature detector may be used for detection at multiple locations of the image. This would also help in **avoiding overfitting**.
  + **Sparsity of connections**: each outpuut depends only on small number of inputs (i.e., takes into account the local variations)
  + **Translational invariance**

## CNN architectures

+ LeNet-5
