# Week 3

This week moves from tabular data to images: a small neural network built with TensorFlow/Keras learns to recognize handwritten digits from the MNIST dataset, and is then tried on a digit image of your own.

## Files

| File | Description |
|:--|:--|
| `rakam_tanima.py` | Trains a dense neural network on MNIST (60,000 training and 10,000 test images of the digits 0–9), prints the test accuracy, and shows predictions for the first five test images. |
| `rakam_test.py` | Loads `el_yazim.png`, converts it to the MNIST format, and asks the trained model which digit it is. |
| `el_yazim.png` | A sample test image: the digit 3 drawn in black on a white 280×280 background. Replace it with your own handwriting. |

## Setup

```bash
pip install tensorflow numpy pillow matplotlib --break-system-packages
```

TensorFlow supports only certain Python versions. If `pip` cannot find a matching release, create a virtual environment with a supported Python version and install the packages there.

## Run

```bash
cd examples/week3

# Train the model (downloads MNIST, about 11 MB, on the first run)
python3 rakam_tanima.py
```

`rakam_test.py` does not run on its own: it uses the `model` trained by `rakam_tanima.py` and calls `plt` without importing it. Run both in the same Python session, for example:

```bash
python3 -i rakam_tanima.py
```

```python
>>> import matplotlib.pyplot as plt
>>> exec(open("rakam_test.py").read())
```

Close the image window to see the result, which should be `Tahmin edilen rakam: 3` for the supplied image. In a Jupyter or Colab notebook, paste the two scripts into consecutive cells and add `import matplotlib.pyplot as plt` to the second one.

## What the training script does

1. Loads MNIST: 28×28 grayscale images of handwritten digits.
2. Scales the pixel values from 0–255 to 0–1.
3. Flattens each image from 28×28 into a vector of 784 numbers.
4. Builds a network with two hidden layers (128 and 64 neurons, ReLU), a 20% dropout layer to reduce overfitting, and a 10-neuron softmax output, one neuron per digit.
5. Trains for 5 epochs with the Adam optimizer, holding out 10% of the training data for validation.
6. Evaluates on the 10,000 test images and prints the accuracy (`Test doğruluğu`).
7. Prints the true and predicted labels for the first five test images.

Expect a test accuracy of roughly 97–98% (97.5% in a test run with TensorFlow 2.22). Exact numbers vary between runs.

## What the test script does

1. Opens `el_yazim.png` and converts it to grayscale.
2. Inverts the colors, because MNIST digits are white on a black background.
3. Resizes the image to 28×28 and displays it.
4. Scales the pixels to 0–1, flattens them to 784 values, and prints the predicted digit.

## Using your own image

Write one digit with a thick black pen on white paper and photograph it, or draw it in a paint program, then save it as `el_yazim.png` in this folder. For the best results:

- Use a square image, so that resizing to 28×28 does not distort the digit.
- Center the digit and let it fill about two thirds of the image, as MNIST digits do.
- Use thick strokes. Thin lines almost disappear at 28×28.
- Keep the background plain and evenly lit.

If your digit is already white on black, remove the `ImageOps.invert(img)` line in `rakam_test.py`.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'tensorflow'`** — see Setup above.
- **`InternalError: No candidates could be compiled`** (after a warning that `libdevice` was not found) — TensorFlow found an NVIDIA GPU but not the CUDA `libdevice` files it needs to compile the model for it. Either run on the CPU, which is fast enough for this model:

  ```bash
  CUDA_VISIBLE_DEVICES="" python3 rakam_tanima.py
  ```

  or tell TensorFlow where CUDA is. It looks for `nvvm/libdevice` inside the given folder, so on Ubuntu with the `nvidia-cuda-toolkit` package:

  ```bash
  mkdir -p ~/.local/cuda/nvvm
  ln -s /usr/lib/nvidia-cuda-toolkit/libdevice ~/.local/cuda/nvvm/libdevice
  XLA_FLAGS=--xla_gpu_cuda_data_dir=$HOME/.local/cuda python3 rakam_tanima.py
  ```
- **`NameError: name 'model' is not defined`** or **`name 'plt' is not defined`** — `rakam_test.py` was run on its own. Run it in the same session as `rakam_tanima.py`, as shown under Run.
- **`FileNotFoundError: el_yazim.png`** — run the script from inside `examples/week3`.
- **Wrong prediction on your own image** — the network only knows digits that look like MNIST. Check the preview window: the digit should be white, centered, and clearly visible on a black background.
