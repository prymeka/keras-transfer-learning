# keras-transfer-learning

A convenience package to simplify Transfer Learning and Fine-Tuning of Keras pre-trained CNN models.

To see how it works check out `example.ipynb`.

To run from the command line, for example, use:
```python
python keras_transfer_learning.py -d mnist -m resnet50
```
The above will fine-tune the `ResNet50` model with `imagenet` weights on the MNIST dataset.

To see all possible CLI arguments use:
```python
python keras_transfer_learning.py --help
```
