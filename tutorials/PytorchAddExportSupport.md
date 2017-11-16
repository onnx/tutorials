## Fail to export the model in PyTorch
Sometimes, when you try to export a model, you may receive some message like
the following:
```
UserWarning: ONNX export failed on elu because torch.onnx.symbolic.elu does not exist
RuntimeError: ONNX export failed: Couldn't export operator elu
```
The export fails because PyTorch does not support exporting `elu` operator.
Furthermore, you have reached out to the ONNX team, but no response has been
received yet. Is it possible to add support for this operator by yourself?
The answer is Yes, but the difficulty (and the amount of code you will need
to touch) depends on the answers to the following questions:

## Determine how hard to add the support for the wanted operator
### Question 1: Is the operator you want standardized in ONNX?
Answer:
- Yes. Great, it will be very straightforward to add the support for the
missing operators.
- No. In this situation, it may be difficult to do the work by yourself.
Check the [Standardization Section](#standardize_op).

### Question 2: Can the ONNX operator be imported by the backend framework, such as Caffe2?
Answer:
- Yes. Terrific, we are able to run the exported ONNX model.
- No. In this situation, you can only export model. Please contact the
importer (such as onnx-caffe2) developers, more work is needed.

## How to add the support to export an operator in PyTorch
### Condition 1: If the operator in PyTorch is an ATen operator.
To determine whether the operator is an ATen operator or not, please check
`torch/csrc/autograd/generated/VariableType.h` (available in generated code in pytorch install dir). If you find the
corresponding function in this header file, most likely, it is an ATen
operator.

**Define symbolic functions.** In this case, you should obey the following rules.
- Define the symbolic function in [`torch/onnx/symbolic.py`](https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic.py). Make sure the
function has the same name as the ATen operator/function defined in
`VariableType.h`.
- The first parameter is always the exported ONNX graph.
- Parameter names must EXACTLY match the names in `VariableType.h`, because
dispatch is done with keyword arguments.
- Parameter ordering does NOT necessarily match what is in `VariableType.h`,
tensors (inputs) are always first, then non-tensor arguments.
- In the symbolic function, we usually need to create an ONNX node in the graph.
Here is an example to create a node for `Elu` ONNX operator:
`g.op("Elu", input, alpha_f=_scalar(alpha))`. More details are included in
[API Secition](#api).
- If the input argument is a tensor, but ONNX asks for a scalar, we have to
explicitly do the conversion. The helper function `_scalar` can convert a
scalar tensor into a python scalar, and `_if_scalar_type_as` can turn a
python scalar into a PyTorch tensor.

In case of adding support for operator `elu`, we can find the following
declaration in `VariableType.h`:
```cpp
virtual Tensor elu(const Tensor & input, Scalar alpha, bool inplace) const override;
```
Apparently, `elu` is implemented in ATen library. So we can define a symbolic
function called `elu` in `torch/onnx/symbolic.py` like the following:
```python
def elu(g, input, alpha, inplace=False):
    return g.op("Elu", input, alpha_f=_scalar(alpha))
```

### Condition 2: If the operator in PyTorch is not an ATen operator.
If you cannot find the corresponding function in `VariableType.h`,
that means you need to define the symoblic function in the PyTorch
Function class. For example, you need to create a `symbolic` function
for operator `Dropout` in [torch/nn/_functions/dropout.py](https://github.com/pytorch/pytorch/blob/99037d627da68cdf53d3d0315deceddfadf03bba/torch/nn/_functions/dropout.py#L14).

**Define symbolic functions.** To define the symbolic functions for
non-ATen ops, the following rules should be obeyed.
- Create a symbolic function named `symbolic` in the corresponding Function
class.
- The first parameter is always the exported ONNX graph.
- Parameter names except the first must EXACTLY match the names in `forward`.
- The output tuple size must match the outputs of `forward`.
- In the symbolic function, we usually need to create an ONNX node in the graph.
Check the [API Section](#api) for more details.


## <a name="api"></a> Export related APIs in PyTorch
Symbolic functions should be implemented in Python. All of these functions interact with Python methods which are implemented via C++-Python bindings, but intuitively the interface they provide looks like this:

```python
def operator/symbolic(g, *inputs):
  """
  Modifies Graph (e.g., using "op"), adding the ONNX operations representing
  this PyTorch function, and returning a Node or tuple of Nodes specifying the
  ONNX outputs whose values correspond to the original PyTorch return values
  of the autograd Function (or None if an output is not supported by ONNX).

  Arguments:
    g (Graph): graph to write the ONNX representation into
    inputs (Node..): list of nodes representing the variables which contain
        the inputs for this function
  """

class Node(object):
  """Represents an intermediate tensor value computed in ONNX."""
  def type(self):
    """Returns the Type of the node."""

class Type(object):
  def sizes(self):
    """Returns a tuple of ints representing the shape of a tensor this describes."""

class Graph(object):
  def op(self, opname, *inputs, **attrs):
    """
    Create an ONNX operator 'opname', taking 'args' as inputs
    and attributes 'kwargs' and add it to the current graph,
    returning the node representing the single output of this
    operator (see the `outputs` keyword argument for multi-return
    nodes).

    The set of operators and the inputs/attributes they take
    is documented at https://github.com/onnx/onnx/blob/master/docs/Operators.md

    Arguments:
        opname (string): The ONNX operator name, e.g., `Abs` or `Add`.
        args (Node...): The inputs to the operator; usually provided
            as arguments to the `symbolic` definition.
        kwargs: The attributes of the ONNX operator, with keys named
            according to the following convention: `alpha_f` indicates
            the `alpha` attribute with type `f`.  The valid type specifiers are
            `f` (float), `i` (int), `s` (string) or `t` (Tensor).  An attribute
            specified with type float accepts either a single float, or a
            list of floats (e.g., you would say `dims_i` for a `dims` attribute
            that takes a list of integers).
        outputs (int, optional):  The number of outputs this operator returns;
            by default an operator is assumed to return a single output.
            If `outputs` is greater than one, this functions returns a tuple
            of output `Node`, representing each output of the ONNX operator
            in positional.
    """
```

## <a name="standardize_op"></a> Standardize the operator in ONNX
If there is no appropriate operator in ONNX to translate to, you will have to
add it to ONNX (ONNX will reject any operators that it does not understand).

**Experimental.** If you just need export to work in a one-off case, without
getting it into ONNX proper, you can add your operator as an *experimental*
operator.

## More ONNX symbolic examples
[ATen operators in symbolic.py](https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic.py)

[Index](https://github.com/pytorch/pytorch/blob/99037d627da68cdf53d3d0315deceddfadf03bba/torch/autograd/_functions/tensor.py#L24)

[Negate](https://github.com/pytorch/pytorch/blob/99037d627da68cdf53d3d0315deceddfadf03bba/torch/autograd/_functions/basic_ops.py#L50)

[ConstantPadNd](https://github.com/pytorch/pytorch/blob/99037d627da68cdf53d3d0315deceddfadf03bba/torch/nn/_functions/padding.py#L8)
