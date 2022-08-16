Dataset methods
=======

+ batch(batch_size, drop_reminder=False,)
  + stacks consecutive <batch_size> elements from the dataset.
+ shuffle(buffer_size)
+ from_tensor_slices(tensors, name=None) -> Dataset
  + slices along the first dimension
  + all the tensors must have same length along the first dimension

+ unique(name=None) -> Dataset
  + supported by elements of tf.int32, tf.int64 or tf.string type
  + Whole dataset must be able to fit into memory

+ cardinality() -> tf.int64 | tf.data.INFINITE_CARDINALITY | tf.data.UNKNOWN_CARDINALITY
+ enumerate(start=0, name=None)
+ save(path, compression=None, shard_func=None, checkpoint_args=None)
  + compression takes GZIP or NONE
+ load(dataset_path) -> Dataset
+ take(count, name=None) -> Dataset
  + count = -1 or >size of dataset returns new dataset containing all elements

+ take_while(predicate_function)
+ apply(transformation_func)   ; transformation_func(Dataset) -> Dataset
+ concatenate(data_set_to_be_concatenated)

+ cache
+ get_single_element()

+ prefetch(buffer_size)
+ reduce()
+ repeat()
