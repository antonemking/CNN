## Requirements

- Python 3
- Tensorflow > 0.12
- Numpy

## Training

Print parameters:

```
 python3 train.py --h
```

```
optional arguments:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 128)
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes (default: '3,4,5')
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularizaion lambda (default: 0.0)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 100)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 100)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 100)
  --allow_soft_placement ALLOW_SOFT_PLACEMENT
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement LOG_DEVICE_PLACEMENT
                        Log placement of ops on devices
  --nolog_device_placement

```

Train:

```
 python3 train.py
```

## Evaluating using test file(test.csv)

```
 python3 eval.py --eval_train --checkpoint_dir="./runs/1518619812/checkpoints/"
```

## Evaluating using interactive session --allows you to type sentences

```
 python3 eval.py --interactive --checkpoint_dir="./runs/1518619812/checkpoints/"
```
Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.

