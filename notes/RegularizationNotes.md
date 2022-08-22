# Model Building Notes

## Tuning Hyperparameters

### Batchsize

+ small batch sizes might offer better generalization capability
+ Start with smaller batch size.

+ **higher batch sizes => lower asymptotic test accuracy**
  + for same learning rate, increased batch size results in poor test accuracy.
  + Consider increasing learning rates, while increasing batch size
  + we can *recover the lost test accuracy from a larger batch size by increasing the learning rate*
  
+ starting with a large batch size doesn’t “get the model stuck” in some neighbourhood of bad local optimums.
  + The model can switch to a lower batch size or higher learning rate anytime to achieve better test accuracy
  + larger batch sizes make larger gradient steps than smaller batch sizes for the same number of samples seen
  + for the same average Euclidean norm distance from the initial weights of the model, larger batch sizes have larger variance in the distance.
  + large batch size means the model makes very large gradient updates and very small gradient updates.
  + The size of the update depends heavily on which particular samples are drawn from the dataset.
  + On the other hand using small batch size means the model makes updates that are all about the same size. The size of the update only weakly depends on which particular samples are drawn from the dataset

  + the distribution of gradients for larger batch sizes has a much heavier tail

  + better solutions can be far away from the initial weights and if the loss is averaged over the batch then large batch sizes simply do not allow the model to travel far enough to reach the better solutions for the same number of training epochs

  + one can compensate for a **larger batch size** by **increasing the learning rate** or number of epochs so that the models can find faraway solutions
  + for a fixed number of steps, the model is limited in how far it can travel using SGD, independent of batch size
  + **SGD may better over ADAM**: ADAM finds solutions with much larger weights, which might explain why it has lower test accuracy and is not generalizing as well
  + for SGD the weights are initialized to approximately the magnitude you want them to be and most of the learning is shuffling the weights along the hyper-sphere of the initial radius. As for ADAM, the model completely ignores the initialization

> References: [Kevin Shen article](https://medium.com/mini-distill/effect-of-batch-size-on-training-dynamics-21c14f7a716e)

## Convloution operations

### Padding

+ same
  + Adds zeros at the borders
+ causal
+ valid
  + equivalent to *no padding*V
