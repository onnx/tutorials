# [ONNX](https://github.com/onnx/onnx) Tutorials

[Open Neural Network Exchange (ONNX)](http://onnx.ai/) is an open standard format for representing machine learning models. ONNX is supported by [a community of partners](https://onnx.ai/supported-tools) who have implemented it in many frameworks and tools.

These images are available for convenience to get started with ONNX and tutorials on this page
  * [Docker image for ONNX and Caffe2/PyTorch](pytorch_caffe2_docker.md)
  * [Docker image for ONNX, ONNX Runtime, and various converters](https://github.com/onnx/onnx-docker/tree/master/onnx-ecosystem)


## Getting ONNX models

* Choose a pre-trained ONNX model from the [ONNX Model Zoo](https://github.com/onnx/models). Many pre-trained ONNX models are provided for common scenarios.
* Use services like [Azure Custom Vision service](https://docs.microsoft.com/en-us/azure/cognitive-services/Custom-Vision-Service/custom-vision-onnx-windows-ml) that generate customized ONNX models for your data
* Convert models from various frameworks (see below)


## Converting to ONNX format
| Framework / Tool | Installation | Tutorial | 
| --- | --- | --- |
| [Caffe](https://github.com/BVLC/caffe) | [apple/coremltools](https://github.com/apple/coremltools) and [onnx/onnxmltools](https://github.com/onnx/onnxmltools) | [Example](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/caffe_coreml_onnx.ipynb) | 
| [Caffe2](http://caffe2.ai) | [part of caffe2 package](https://github.com/pytorch/pytorch/tree/master/caffe2/python/onnx) | [Example](tutorials/Caffe2OnnxExport.ipynb) | 
| [Chainer](https://chainer.org/) | [chainer/onnx-chainer](https://github.com/chainer/onnx-chainer) | [Example](tutorials/ChainerOnnxExport.ipynb) |
| [Cognitive Toolkit (CNTK)](https://www.microsoft.com/en-us/cognitive-toolkit/) | [built-in](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine) | [Example](tutorials/CntkOnnxExport.ipynb) | 
| [CoreML (Apple)](https://developer.apple.com/documentation/coreml) | [onnx/onnx-coreml](https://github.com/onnx/onnx-coreml) and [onnx/onnxmltools](https://github.com/onnx/onnxmltools) | [Example](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/coreml_onnx.ipynb) | 
| [Keras](https://github.com/keras-team/keras) | [onnx/keras-onnx](https://github.com/onnx/keras-onnx) | [Example](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/keras_onnx.ipynb) | n/a |
| [LibSVM](https://github.com/cjlin1/libsvm) | [onnx/onnxmltools](https://github.com/onnx/onnxmltools) | [Example](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/libsvm_onnx.ipynb) | n/a |
| [LightGBM](https://github.com/Microsoft/LightGBM) | [onnx/onnxmltools](https://github.com/onnx/onnxmltools) | [Example](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/lightgbm_onnx.ipynb) | n/a |
| [MATLAB](https://www.mathworks.com/) | [Deep Learning Toolbox](https://www.mathworks.com/matlabcentral/fileexchange/67296) | [Example](https://www.mathworks.com/help/deeplearning/ref/exportonnxnetwork.html) |
| [ML.NET](https://github.com/dotnet/machinelearning/) | [built-in](https://www.nuget.org/packages/Microsoft.ML/) | [Example](https://github.com/dotnet/machinelearning/blob/master/test/Microsoft.ML.Tests/OnnxConversionTest.cs) |
| [MXNet (Apache)](http://mxnet.incubator.apache.org/) | part of mxnet package [docs](http://mxnet.incubator.apache.org/api/python/contrib/onnx.html) [github](https://github.com/apache/incubator-mxnet/tree/master/python/mxnet/contrib/onnx) | [Example](tutorials/MXNetONNXExport.ipynb) |
| [PyTorch](http://pytorch.org/) | [part of pytorch package](http://pytorch.org/docs/master/onnx.html) | [Example](tutorials/PytorchOnnxExport.ipynb), [exporting different ONNX opsets](https://github.com/onnx/tutorials/blob/master/tutorials/ExportModelFromPyTorchToDifferentONNXOpsetVersions.md), [Extending support](tutorials/PytorchAddExportSupport.md) |
| [SciKit-Learn](http://scikit-learn.org/) | [onnx/sklearn-onnx](https://github.com/onnx/sklearn-onnx) | [Example](http://onnx.ai/sklearn-onnx/index.html) | n/a |
| [SINGA (Apache)](http://singa.apache.org/) - [Github](https://github.com/apache/incubator-singa/blob/master/python/singa/sonnx.py) (experimental) | [built-in](https://github.com/apache/incubator-singa/blob/master/doc/en/docs/installation.md) | [Example](https://github.com/apache/incubator-singa/tree/master/examples/onnx) |
| [TensorFlow](https://www.tensorflow.org/) | [onnx/tensorflow-onnx](https://github.com/onnx/tensorflow-onnx) | [Examples](https://github.com/onnx/tensorflow-onnx/tree/master/examples) |


## Scoring ONNX Models
Once you have an ONNX model, it can be scored with a variety of tools.

| Framework / Tool | Installation | Tutorial | 
| --- | --- | --- |
| [Caffe2](http://caffe2.ai) | [Caffe2](https://github.com/pytorch/pytorch/tree/master/caffe2/python/onnx) | [Example](tutorials/OnnxCaffe2Import.ipynb) |
| [Cognitive Toolkit (CNTK)](https://www.microsoft.com/en-us/cognitive-toolkit/) | [built-in](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine) | [Example](tutorials/OnnxCntkImport.ipynb)|
| [CoreML (Apple)](https://developer.apple.com/documentation/coreml) | [onnx/onnx-coreml](https://github.com/onnx/onnx-coreml) and [CoreMLTools](https://github.com/apple/coremltools) | [Example](tutorials/OnnxCoremlImport.ipynb)|
| [MATLAB](https://www.mathworks.com/) | [Deep Learning Toolbox Converter](https://www.mathworks.com/matlabcentral/fileexchange/67296) | [Documentation and Examples](https://www.mathworks.com/help/deeplearning/ref/importonnxnetwork.html) |
| [Menoh](https://github.com/pfnet-research/menoh) | [Github Packages](https://github.com/pfnet-research/menoh/releases) or from [Nuget](https://www.nuget.org/packages/Menoh/) | [Example](tutorials/OnnxMenohHaskellImport.ipynb) |
| [ML.NET](https://github.com/dotnet/machinelearning/) | [Microsoft.ML Nuget Package](https://www.nuget.org/packages/Microsoft.ML/) | [Example](https://github.com/dotnet/machinelearning/blob/master/test/Microsoft.ML.OnnxTransformerTest/OnnxTransformTests.cs) |
| [MXNet (Apache)](http://mxnet.incubator.apache.org/) - [Github](https://github.com/apache/incubator-mxnet/tree/master/python/mxnet/contrib/onnx) | [MXNet](http://mxnet.incubator.apache.org/versions/master/install/index.html?platform=Linux&language=Python&processor=CPU) |  [API](http://mxnet.incubator.apache.org/api/python/contrib/onnx.html)<br>[Example](tutorials/OnnxMxnetImport.ipynb) |
[ONNX Runtime](https://github.com/microsoft/onnxruntime) | Python (Pypi) - [onnxruntime](https://pypi.org/project/onnxruntime/) and [onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu)<br>C/C# (Nuget) - [Microsoft.ML.OnnxRuntime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/) and [Microsoft.ML.OnnxRuntime.Gpu](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu/)| APIs: [Python](https://aka.ms/onnxruntime-python), [C#](https://github.com/Microsoft/onnxruntime/blob/master/docs/CSharp_API.md), [C](https://github.com/Microsoft/onnxruntime/blob/master/docs/C_API.md), [C++](https://github.com/Microsoft/onnxruntime/blob/master/onnxruntime/core/session/inference_session.h)<br>Examples - [Python](https://microsoft.github.io/onnxruntime/auto_examples/plot_load_and_predict.html#), [C#](https://github.com/Microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.Tests/InferenceTest.cs#L54), [C](https://github.com/Microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp) |
| [SINGA (Apache)](http://singa.apache.org/) - [Github](https://github.com/apache/incubator-singa/blob/master/python/singa/sonnx.py) [experimental]| [built-in](https://github.com/apache/incubator-singa/blob/master/doc/en/docs/installation.md) | [Example](https://github.com/apache/incubator-singa/tree/master/examples/onnx) |
| [Tensorflow](https://www.tensorflow.org/) | [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) | [Example](tutorials/OnnxTensorflowImport.ipynb)|
| [TensorRT](https://developer.nvidia.com/tensorrt) | [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt) | [Example](https://github.com/onnx/onnx-tensorrt/blob/master/README.md) |
| [Windows ML](https://docs.microsoft.com/en-us/windows/ai/windows-ml) | Pre-installed on [Windows 10](https://docs.microsoft.com/en-us/windows/ai/release-notes) | [API](https://docs.microsoft.com/en-us/windows/ai/api-reference)<br>Tutorials - [C++ Desktop App](https://docs.microsoft.com/en-us/windows/ai/get-started-desktop), [C# UWP App](https://docs.microsoft.com/en-us/windows/ai/get-started-uwp)<br> [Examples](https://docs.microsoft.com/en-us/windows/ai/tools-and-samples) |


## End-to-End Tutorials

### Serving
  * [Serving ONNX models with MXNet Model Server](tutorials/ONNXMXNetServer.ipynb)
  * [ONNX model hosting with AWS SageMaker and MXNet](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/mxnet_onnx_eia/mxnet_onnx_eia.ipynb) 
  * Serving ONNX models with ONNX Runtime on Azure ML
    * [FER Facial Expression Recognition](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-inference-facial-expression-recognition-deploy.ipynb)
    * [MNIST Handwritten Digits](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-inference-mnist-deploy.ipynb)
    * [Resnet50 Image Classification](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-modelzoo-aml-deploy-resnet50.ipynb)
  * [Inferencing ONNX models using ONNX Runtime Python API](https://microsoft.github.io/onnxruntime/auto_examples/plot_load_and_predict.html#sphx-glr-auto-examples-plot-load-and-predict-py)

### ONNX as an intermediary format
  * [Convert a PyTorch model to Tensorflow using ONNX](tutorials/PytorchTensorflowMnist.ipynb)


### Conversion to deployment
  * [Converting SuperResolution model from PyTorch to Caffe2 with ONNX and deploying on mobile device](tutorials/PytorchCaffe2SuperResolution.ipynb)
  * [Transferring SqueezeNet from PyTorch to Caffe2 with ONNX and to Android app](tutorials/PytorchCaffe2MobileSqueezeNet.ipynb)
  * [Converting Style Transfer model from PyTorch to CoreML with ONNX and deploying to an iPhone](https://github.com/onnx/tutorials/tree/master/examples/CoreML/ONNXLive)
  * [Serving PyTorch Models on AWS Lambda with Caffe2 & ONNX](https://machinelearnings.co/serving-pytorch-models-on-aws-lambda-with-caffe2-onnx-7b096806cfac)
  * [MXNet to ONNX to ML.NET with SageMaker, ECS and ECR](https://cosminsanda.com/posts/mxnet-to-onnx-to-ml.net-with-sagemaker-ecs-and-ecr/) - external link
  * [Convert CoreML YOLO model to ONNX, score with ONNX Runtime, and deploy in Azure](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-convert-aml-deploy-tinyyolo.ipynb)
  

## Other ONNX tools

* [Verifying correctness and comparing performance](tutorials/CorrectnessVerificationAndPerformanceComparison.ipynb)
* [Visualizing an ONNX model](tutorials/VisualizingAModel.md) (useful for debugging)
* [Netron: a viewer for ONNX models](https://github.com/lutzroeder/Netron)
* [Example of operating on ONNX protobuf](https://github.com/onnx/onnx/blob/master/onnx/examples/Protobufs.ipynb)
* [Float16 <-> Float32 converter](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/float32_float16_onnx.ipynb)

## Contributing

We welcome improvements to the convertor tools and contributions of new ONNX bindings. Check out [contributor guide](https://github.com/onnx/onnx/blob/master/docs/CONTRIBUTING.md) to get started.

Use ONNX for something cool? Send the tutorial to this repo by submitting a PR.
