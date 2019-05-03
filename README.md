# [ONNX](https://github.com/onnx/onnx) Tutorials

[Open Neural Network Exchange (ONNX)](http://onnx.ai/) is an open standard format for representing machine learning models. ONNX is supported by [a community of partners](https://onnx.ai/supported-tools) who have implemented it in many frameworks and tools.

## Getting ONNX models

* Choose a pre-trained ONNX model from the [ONNX Model Zoo](https://github.com/onnx/models). A lot of pre-trained ONNX models are provided for common scenarios.
* Convert models from mainstream frameworks. More tutorials are below.

| Framework / tool | Installation | Exporting to ONNX (frontend) | Importing ONNX models (backend) |
| --- | --- | --- | --- |
| [Caffe](https://github.com/BVLC/caffe) | [apple/coremltools](https://github.com/apple/coremltools) and [onnx/onnxmltools](https://github.com/onnx/onnxmltools) | [Exporting](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/caffe_coreml_onnx.ipynb) | n/a |
| [Caffe2](http://caffe2.ai) | [part of caffe2 package](https://github.com/pytorch/pytorch/tree/master/caffe2/python/onnx) | [Exporting](tutorials/Caffe2OnnxExport.ipynb) | [Importing](tutorials/OnnxCaffe2Import.ipynb) |
| [Chainer](https://chainer.org/) | [chainer/onnx-chainer](https://github.com/chainer/onnx-chainer) | [Exporting](tutorials/ChainerOnnxExport.ipynb) | coming soon |
| [Cognitive Toolkit (CNTK)](https://www.microsoft.com/en-us/cognitive-toolkit/) | [built-in](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine) | [Exporting](tutorials/CntkOnnxExport.ipynb) | [Importing](tutorials/OnnxCntkImport.ipynb) |
| [Apple CoreML](https://developer.apple.com/documentation/coreml) | [onnx/onnx-coreml](https://github.com/onnx/onnx-coreml) and [onnx/onnxmltools](https://github.com/onnx/onnxmltools) | [Exporting](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/coreml_onnx.ipynb) | [Importing](tutorials/OnnxCoremlImport.ipynb) |
| [Keras](https://github.com/keras-team/keras) | [onnx/keras-onnx](https://github.com/onnx/keras-onnx) | [Exporting](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/keras_onnx.ipynb) | n/a |
| [LibSVM](https://github.com/cjlin1/libsvm) | [onnx/onnxmltools](https://github.com/onnx/onnxmltools) | [Exporting](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/libsvm_onnx.ipynb) | n/a |
| [LightGBM](https://github.com/Microsoft/LightGBM) | [onnx/onnxmltools](https://github.com/onnx/onnxmltools) | [Exporting](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/lightgbm_onnx.ipynb) | n/a |
| [MATLAB](https://www.mathworks.com/) | [onnx converter on matlab central file exchange](https://www.mathworks.com/matlabcentral/fileexchange/67296) | [Exporting](https://www.mathworks.com/help/deeplearning/ref/exportonnxnetwork.html) | [Importing](https://www.mathworks.com/help/deeplearning/ref/importonnxnetwork.html) |
| [Menoh](https://github.com/pfnet-research/menoh) | [pfnet-research/menoh](https://github.com/pfnet-research/menoh) | n/a | [Importing](tutorials/OnnxMenohHaskellImport.ipynb) |
| [ML.NET](https://github.com/dotnet/machinelearning/) | [built-in](https://www.nuget.org/packages/Microsoft.ML/) | [Exporting](https://github.com/dotnet/machinelearning/blob/master/test/Microsoft.ML.Tests/OnnxConversionTest.cs) | [Importing](https://github.com/dotnet/machinelearning/blob/master/test/Microsoft.ML.OnnxTransformerTest/OnnxTransformTests.cs) |
| [Apache MXNet](http://mxnet.incubator.apache.org/) | part of mxnet package [docs](http://mxnet.incubator.apache.org/api/python/contrib/onnx.html) [github](https://github.com/apache/incubator-mxnet/tree/master/python/mxnet/contrib/onnx) | [Exporting](tutorials/MXNetONNXExport.ipynb) | [Importing](tutorials/OnnxMxnetImport.ipynb) |
| [PyTorch](http://pytorch.org/) | [part of pytorch package](http://pytorch.org/docs/master/onnx.html) | [Exporting](tutorials/PytorchOnnxExport.ipynb), [Exporting - Different Opsets - ONNX](tutorials/ExportModelFromPyTorchToDifferentONNXOpsetVersions.md), [Extending support](tutorials/PytorchAddExportSupport.md) | coming soon |
| [SciKit-Learn](http://scikit-learn.org/) | [onnx/sklearn-onnx](https://github.com/onnx/sklearn-onnx) | [Exporting](http://onnx.ai/sklearn-onnx/index.html) | n/a |
| [TensorFlow](https://www.tensorflow.org/) | [onnx/onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) and [onnx/tensorflow-onnx](https://github.com/onnx/tensorflow-onnx) | [Exporting - ONNX-Tensorflow](tutorials/OnnxTensorflowExport.ipynb)<br>[Exporting - Tensorflow-ONNX](https://github.com/onnx/tensorflow-onnx/blob/master/examples/call_coverter_via_python.py) | [Importing](tutorials/OnnxTensorflowImport.ipynb) [experimental] |
| [TensorRT](https://developer.nvidia.com/tensorrt) | [onnx/onnx-tensorrt](https://github.com/onnx/onnx-tensorrt) | n/a | [Importing](https://github.com/onnx/onnx-tensorrt/blob/master/README.md) |

* Use services like [Azure Custom Vision service](https://docs.microsoft.com/en-us/azure/cognitive-services/Custom-Vision-Service/custom-vision-onnx-windows-ml) that generate customized ONNX models for your data

## End-to-end tutorials

* For preparation
  * [Docker image for Caffe2/PyTorch/ONNX tutorials](pytorch_caffe2_docker.md)
  * [Docker image for ONNX, ONNX Runtime, and Converters](https://github.com/onnx/onnx-docker/tree/master/onnx-ecosystem)

* For serving
  * [Serving ONNX models with MXNet Model Server](tutorials/ONNXMXNetServer.ipynb)
  * [ONNX model hosting with AWS SageMaker and MXNet](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/mxnet_onnx_eia/mxnet_onnx_eia.ipynb) 
  * Serving ONNX models with ONNX Runtime on Azure ML - [FER Facial Expression Recognition](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-inference-facial-expression-recognition-deploy.ipynb), [MNIST Handwritten Digits](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-inference-mnist-deploy.ipynb), [Resnet50 Image Classification](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-modelzoo-aml-deploy-resnet50.ipynb)
  * [Inferencing ONNX models using ONNX Runtime Python API](https://microsoft.github.io/onnxruntime/auto_examples/plot_load_and_predict.html#sphx-glr-auto-examples-plot-load-and-predict-py)

* For conversion
  * [Convert a PyTorch model to Tensorflow using ONNX](tutorials/PytorchTensorflowMnist.ipynb)

  For direct conversion to/from ONNX format, see the "Exporting" and "Importing" columns in the table under [Getting ONNX Models](tutorials#getting-onnx-models)

* From conversion to deployment
  * [Converting SuperResolution model from PyTorch to Caffe2 with ONNX and deploying on mobile device](tutorials/PytorchCaffe2SuperResolution.ipynb)
  * [Transferring SqueezeNet from PyTorch to Caffe2 with ONNX and to Android app](tutorials/PytorchCaffe2MobileSqueezeNet.ipynb)
  * [Converting Style Transfer model from PyTorch to CoreML with ONNX and deploying to an iPhone](https://github.com/onnx/tutorials/tree/master/examples/CoreML/ONNXLive)
  * [Serving PyTorch Models on AWS Lambda with Caffe2 & ONNX](https://machinelearnings.co/serving-pytorch-models-on-aws-lambda-with-caffe2-onnx-7b096806cfac)
  * [MXNet to ONNX to ML.NET with SageMaker, ECS and ECR](https://cosminsanda.com/posts/mxnet-to-onnx-to-ml.net-with-sagemaker-ecs-and-ecr/) - external link
  * [Convert CoreML YOLO model to ONNX, score with ONNX Runtime, and deploy in Azure](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-convert-aml-deploy-tinyyolo.ipynb)
  

## ONNX tools

* [Verifying correctness and comparing performance](tutorials/CorrectnessVerificationAndPerformanceComparison.ipynb)
* [Visualizing an ONNX model](tutorials/VisualizingAModel.md) (useful for debugging)
* [Netron: a viewer for ONNX models](https://github.com/lutzroeder/Netron)
* [Example of operating on ONNX protobuf](https://github.com/onnx/onnx/blob/master/onnx/examples/Protobufs.ipynb)
* [Float16 <-> Float32 converter](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/float32_float16_onnx.ipynb)

## Contributing

We welcome improvements to the convertor tools and contributions of new ONNX bindings. Check out [contributor guide](https://github.com/onnx/onnx/blob/master/docs/CONTRIBUTING.md) to get started.

Use ONNX for something cool? Send the tutorial to this repo by submitting a PR.
