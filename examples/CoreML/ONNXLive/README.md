# ONNXLive Tutorial:
Convert style transfer models from PyTorch to CoreML.

ONNX is a neural network exchange format that can be exported and imported from/to many deep learning frameworks.
In this tutorial, we will use it to convert PyTorch style transfer models to CoreML and
run them on an iPhone on the live stream from the camera.

## Overview

This tutorial goes through 4 steps:

1. [Download (or train) PyTorch style transfer models](#download-or-train-pytorch-style-transfer-models)
2. [Convert the PyTorch models to ONNX models](#convert-the-pytorch-models-to-onnx-models)
3. [Convert the ONNX models to CoreML models](#convert-the-onnx-models-to-coreml-models)
4. [Run the CoreML models in a style transfer iOS App](#run-the-coreml-models-in-a-style-transfer-ios-app)

## Preparations

We're working in a virtualenv to not screw with your local packages.
We're using Python 3.6 for this tutorial, but other versions should work too.

    python3.6 -m venv venv
    source ./venv/bin/activate

You need to install pytorch

    pip install torchvision
    
and the onnx->coreml converter.

    git clone https://github.com/onnx/onnx-coreml
    cd onnx-coreml
    git submodule update --recursive --init
    ./install.sh
    cd ..
    
You'd also need to install XCode if you want to run the iOS style transfer app on your iPhone.
Converting the models will work on Linux too, but to run the iOS app, you need a Mac.

## Download (or train) PyTorch style transfer models

For this tutorial, we use the style transfer models that are published with pytorch in https://github.com/pytorch/examples/tree/master/fast_neural_style .
If you want to use different PyTorch or ONNX models, feel free to skip this step.

These models are meant for doing style transfer on still images and are not optimized to be fast enough for video,
but if we reduce the resolution enough, they work well on videos as well.

Let's download these models.

    git clone https://github.com/pytorch/examples
    cd examples/fast_neural_style

If you want to train the models yourself, the pytorch/examples repository we just cloned has more information on how to do this.
For now, we'll just download pretrained models with the script provided by the repository:

    ./download_saved_models.sh

This script downloads the pretrained PyTorch models and puts them into the `saved_models` folder.
There should now be 4 files, `candy.pth`, `mosaic.pth`, `rain_princess.pth` and `udnie.pth`.

## Convert the PyTorch models to ONNX models

Now we have the pretrained PyTorch models as `.pth` files in the `saved_models` folder and want to convert them to ONNX.
The model definition is in the pytorch/examples repository we cloned before, and with a few lines of python we can export it.
Basically, instead of running the neural net, we call `torch.onnx._export`, which is provided with PyTorch.
However, in this case we don't even need to do that, because there already is a script `neural_style/neural_style.py` doing this for us.
You can take a look at that script if you want to apply it to different models.

Since the PyTorch to ONNX export is based on tracing your neural network, this will internally run the network.
For this, it needs an input image to apply the style transfer to. This can just be an blank image.
However, the pixel size of this image is important, as this will be the size for the exported style transfer model.
To get good performance, we'll use a resolution of 250x540. Feel free to take a larger resolution if you care less about
FPS and more about style transfer quality.

Let's create a blank image of the resolution we want

    convert -size 250x540 xc:white png24:dummy.jpg

and use that to export the PyTorch models

    python ./neural_style/neural_style.py eval --content-image dummy.jpg --output-image dummy-out.jpg --model ./saved_models/candy.pth --cuda 0 --export_onnx ./saved_models/candy.onnx
    python ./neural_style/neural_style.py eval --content-image dummy.jpg --output-image dummy-out.jpg --model ./saved_models/udnie.pth --cuda 0 --export_onnx ./saved_models/udnie.onnx
    python ./neural_style/neural_style.py eval --content-image dummy.jpg --output-image dummy-out.jpg --model ./saved_models/rain_princess.pth --cuda 0 --export_onnx ./saved_models/rain_princess.onnx
    python ./neural_style/neural_style.py eval --content-image dummy.jpg --output-image dummy-out.jpg --model ./saved_models/mosaic.pth --cuda 0 --export_onnx ./saved_models/mosaic.onnx

You should end up getting 4 files, `candy.onnx`, `mosaic.onnx`, `rain_princess.onnx` and `udnie.onnx`,
created from the corresponding `.pth` files.

## Convert the ONNX models to CoreML models

Now that we have the ONNX models, we can convert them to CoreML models to run on iOS devices.
For this, we use the onnx-coreml converter we installed previously.
The converter comes with a `convert-onnx-to-coreml` script, which the installation steps above added to our path,
but unfortunately that doesn't work for us, because we need to mark the input and output of the network as an image
and while this is supported by the converter, it is only supported when calling the converter from python.

Looking at the style transfer model (for example opening the .onnx file in an application like [Netron](https://github.com/lutzroeder/Netron)),
we see that the input is named '0' and the output is named '186'. These are just numeric ids assigned by PyTorch.
These are the ones we need to mark as images.

So let's write a small python file `onnx_to_coreml.py`

    import sys
    from onnx import onnx_pb2
    from onnx_coreml import convert
    
    model_in = sys.argv[1]
    model_out = sys.argv[2]
    
    model_file = open(model_in, 'rb')
    model_proto = onnx_pb2.ModelProto()
    model_proto.ParseFromString(model_file.read())
    coreml_model = convert(model_proto, image_input_names=['0'], image_output_names=['186'])
    coreml_model.save(model_out)
    
and run it

    python onnx_to_coreml.py ./saved_models/candy.onnx ./saved_models/candy.mlmodel
    python onnx_to_coreml.py ./saved_models/udnie.onnx ./saved_models/udnie.mlmodel
    python onnx_to_coreml.py ./saved_models/rain_princess.onnx ./saved_models/rain_princess.mlmodel
    python onnx_to_coreml.py ./saved_models/mosaic.onnx ./saved_models/mosaic.mlmodel

Now, there should be 4 CoreML models in your `saved_models` directory: `candy.mlmodel`, `mosaic.mlmodel`, rain_princess.mlmodel` and `udnie.mlmodel`.

## Run the CoreML models in a style transfer iOS App

This repository (i.e. the one you're currently reading the README.md of) contains an iOS app able to run CoreML style transfer models
on a live camera stream from your phone camera. Let's clone the repository

    git clone https://github.com/onnx/tutorials
    
and open the `tutorials/examples/CoreML/ONNXLive/ONNXLive.xcodeproj` project in XCode.
We recommend using XCode 9.3 and an iPhone X. There might be issues running on older devices or XCode versions.

In the `Models/` folder, the project contains some .mlmodel files. We're going to replace them with the models we just created.

Then, run the app on your iPhone and your all set. Tapping on the screen switches through the models.

## Conclusion

We hope this tutorial gave you an overview of what ONNX is about and how you can use it to convert neural networks
between frameworks, in this case a few style transfer models from PyTorch to CoreML.

Feel free to experiment with these steps and test them on your own models.
Please let us know if you hit any issues or want to give feedback. We'd like to hear what you think.
