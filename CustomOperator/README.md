# Required Steps

  - [1](#step1) - Adding the custom operator implementation in C++ and registering it with TorchScript
  - [2](#step2) - Exporting the custom Operator to ONNX, using:
  <br />             - a combination of existing ONNX ops
  <br />              or
  <br />              - a custom ONNX Operator
  - [3](#step3) - Adding the custom operator implementation and registering it in ONNX Runtime

<a name="step1"></a>
# Implement the Custom Operator
For this step, you need to have PyTorch installed on your system. Try installing PyTorch nightly build from [here](https://pytorch.org/get-started/locally/).
If you have a custom op that you need to add in PyTorch as a C++ extension, you need to implement the op and build it with ```setuptools```.
Start by implementing the operator in C++. Below we have the example C++ code group norm operator:

```cpp
#include <torch/script.h>
#include "Eigen/Dense"

template <typename T>
using ConstEigenVectorArrayMap = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;
template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;

torch::Tensor custom_group_norm(torch::Tensor X, torch::Tensor num_groups, torch::Tensor scale, torch::Tensor bias, torch::Tensor eps) {

  float* X_data = X.data<float>();
  float* scale_data = scale.data<float>();
  float* bias_data = bias.data<float>();
  int num_groups_i = int(num_groups.data<float>()[0]);
  float epsilon_ = eps.data<float>()[0];
  torch::Tensor output = torch::zeros(X.sizes());
  float* out = output.data<float>();
  const int64_t N = X.size(0);
  const int64_t C = X.size(1) / num_groups_i;  // assume [N C*num_groups H W]  per the spec

  int64_t sample_size = 1;
  for (size_t i = 2; i < X.dim(); ++i) {
    sample_size *= X.size(i);
  }
  sample_size *= C;

  std::vector<float> Xi;
  for (auto i = 0; i < N * num_groups_i; ++i) {
    ConstEigenVectorArrayMap<float> Xi(X_data + sample_size * i, sample_size);
    const float Xi_mean = Xi.mean();
    const float squared_norm = (Xi - Xi_mean).matrix().squaredNorm();
    const float inv_stdev = 1.0f / std::sqrt(squared_norm / sample_size + epsilon_);
    EigenVectorArrayMap<float> Yi(out + sample_size * i, sample_size);
    const float channel_scale = inv_stdev * scale_data[i % (C * num_groups_i)];
    const float channel_shift = bias_data[i % (C * num_groups_i)] - Xi_mean * channel_scale;
    Yi = Xi * channel_scale + channel_shift;
   }

  return output;
}
```
For this example, we use the [Eigen](https://eigen.tuxfamily.org/dox/index.html) library. To do this, we just need download and extract Eigen header files. You can find this library [here](https://eigen.tuxfamily.org/dox/GettingStarted.html).
<br />
Next, you need to register this operator with TorchScript compiler using ```torch::RegisterOperator``` function in the same cpp file. The first argument is operator name and namespace separated by ```::```. The next argument is a reference to your function. 

```cpp
static auto registry = torch::RegisterOperators("mynamespace::custom_group_norm", &custom_group_norm);
```

Once you have your C++ function, you can build it using ```setuptools.Extension```. Create a ```setup.py script``` in the same directory where you have your C++ code. ```CppExtension.BuildExtension``` Takes care of the required compiler flags such as required include paths, and flags required during mixed C++/CUDA mixed compilation.

For this example, we only provide the forward pass function needed for inferencing. Similarly, you can implement the backward pass if needed.

```python
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='custom_group_norm',
      ext_modules=[cpp_extension.CppExtension('custom_group_norm', ['custom_group_norm.cpp'])],
                    include_dirs = [<path_to_eigen_header_file>])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
```

Make sure to include required header files in ```include_dirs``` list.

Now, running the command ```python setup.py install``` from your source directory, you can to build and install your extension.
The shared object should be generated under ```build``` directory. 
You can load it using:
```torch.ops.load_library("<path_to_object_file>)```
Then you can refer to your custom operator: 
```torch.ops.<namespace_name>.<operator_name>```

<a name="step2"></a>
# Export the Operator to ONNX

You can export your custom operator using existing ONNX ops, or you can create custom ONNX ops to use.
In both cases, you need to add the symbolic method to the exporter, and register your custom symbolic using ```torch.onnx.register_custom_op_symbolic```.
The first argument contains the namespace and operator name, separated by ```::```. You also need to pass a reference to the custom symbolic method, and the ONNX opset version. You can add your script in a python file under the source directory.
```python
def my_group_norm(g, input, num_groups, scale, bias, eps):
    return g.op("mydomain::mygroupnorm", input, num_groups, scale, bias, epsilon_f=eps)

from torch.onnx import register_custom_op_symbolic
register_custom_op_symbolic('mynamespace::custom_group_norm', my_group_norm, 9)
```

In the symbolic method, you need to implement the ONNX subgraph to use for exporting your custom op. If you are using existing ONNX operators (from the default ONNX domain), you don't have to add the domain name prefix.
In our example, we want to use a custom ONNX op from our custom domain. Therefore, we need to add the domain name prefix in the following format:
```"<domain_name>::<onnx_op>"```

If you want to assign a version when registering your custom domain, you can do that using ```torch.onnx.set_custom_domain_version``` API. Otherwise, version 1 will be assigned to the custom domain by default.

Now, You can create a ```torch.nn.module``` using your custom op, and export it to ONNX using ```torch.onnx.export```. Make sure to specify input and output names at export, as this will help you later when implementing the kernel for this operator:

```python 
import torch

def export_custom_op():
    class CustomModel(torch.nn.Module):
        def forward(self, x, num_groups, scale, bias):
            return torch.ops.mydomain.custom_group_norm(x, num_groups, scale, bias, torch.tensor([0.]))

    X = torch.randn(3, 2, 1, 2)
    num_groups = torch.tensor([2.])
    scale = torch.tensor([2., 1.])
    bias = torch.tensor([1., 0.])
    inputs = (X, num_groups, scale, bias)

    f = './model.onnx'
    torch.onnx.export(CustomModel(), inputs, f,
                       opset_version=9,
                       example_outputs=None,
                       input_names=["X", "num_groups", "scale", "bias"], output_names=["Y"])
```

To be able to use this custom ONNX operator for inferencing, we add our custom operator to an inference engine. If you are using existing ONNX ops only, you do not need to go through this last step.

<a name="step3"></a>
# Implement the Operator in ONNX Runtime #

The last step is to implement this op in ONNX Runtime, and build it. For this step, you need to have ONNX Runtime installed on your system. You can install ONNXRuntime v1.0.0 using:
```
pip install onnxruntime
```
or find the nuget package from [here](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/).

The last step is to implement this op in ONNX Runtime, and build. We show how to do this using the custom operator C API's (API's are experimental for now).
First, you need to create a custom domain of type ```Ort::CustomOpDomain```. This domain name is the same name provided in the symbolic method (step 2) when exporting the model.

```cpp
Ort::CustomOpDomain custom_op_domain("org.pytorch.mydomain");
```
Next, you need to create a ```ORT::CustomOp``` object, write its kernel implementation, and add it to your custom domain:

```cpp
struct Input {
  const char* name;
  std::vector<int64_t> dims;
  std::vector<float> values;
};

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

template <typename T>
struct GroupNormKernel {
	private:
   float epsilon_;
   Ort::CustomOpApi ort_;

	public:
  GroupNormKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info) : ort_(ort) {
    epsilon_ = ort_.KernelInfoGetAttribute<float>(info, "epsilon");
  }

  void Compute(OrtKernelContext* context);
};


struct GroupNormCustomOp : Ort::CustomOpBase<GroupNormCustomOp, GroupNormKernel<float>> {
  void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) { return new GroupNormKernel<float>(api, info); };
  const char* GetName() const { return "testgroupnorm"; };

  size_t GetInputTypeCount() const { return 4; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
};
```

The Compute function is implemented [in the source file](https://github.com/neginraoof/CustomOperators/blob/master/CuctomOperator/ort_custom_op/custom_op.cc).
Once you have the custom kernel and schema, you can add them to the domain using the C API as below:
```cpp
GroupNormCustomOp custom_op;
custom_op_domain.Add(&custom_op);
```

In the repository, you can find our example group norm implementation along with a sample ONNX Runtime unit test to verify the expected output.
You can use cmake to build your custom operator with the required dependencies. Add a file named ```CMakeLists.txt``` under the same directory where you have your source files.

You can link the required libraries in your cmake file using ```target_link_libraries``` :
```
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS <PATH_TO_YOUR_INSTALLATION_DIRECTORY>)
target_link_libraries(customop PUBLIC ${ONNXRUNTIME_LIBRARY})
```

And include the required headers using ```include_directories```
```
include_directories(<PATH_TO_EIGEN_HEADER_FILE>)
```

An example ```CMakeLists.txt``` file we could be found [here](https://github.com/neginraoof/CustomOperators/blob/master/CuctomOperator/ort_custom_op/CMakeLists.txt).

Once you have the cmake file, create a build directory from the same location and try ```cd build```. Execute the command ```cmake ..``` to configure the project and build it using ```make``` command.

Now that you have registered your operator, you should be able to run your model and test it. You can find the source code and test for a sample custom operator [here](https://github.com/neginraoof/CustomOperators/blob/master/CuctomOperator/ort_custom_op/custom_op_test.cc). 



### References:
1- [Extending TorchScript with Custom C++ Operators](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html)

2- [ONNX Runtime: Adding a New Op](https://github.com/microsoft/onnxruntime/blob/master/docs/AddingCustomOp.md)

