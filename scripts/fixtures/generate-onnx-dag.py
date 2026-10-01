"""Regenerate tiny ONNX DAG fixtures; requires torch and onnx (no downloads)."""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import torch

ROOT = Path(__file__).parent

def save(name, nodes, inputs, outputs, initializers=()):
    graph = helper.make_graph(nodes, name, inputs, outputs, list(initializers))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, ROOT / f'{name}.onnx')


def tensor(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


save('tiny-unsupported', [helper.make_node('Abs', ['x'], ['y'], name='unsupported_abs')],
     [tensor('x', ['batch', 3])], [tensor('y', ['batch', 3])])
save('tiny-ordered-dag', [
    helper.make_node('MatMul', ['a', 'b'], ['product'], name='ordered_product'),
    helper.make_node('Sub', ['c', 'product'], ['difference'], name='ordered_difference'),
    helper.make_node('Concat', ['difference', 'product'], ['joined'], name='ordered_join', axis=1),
], [tensor('a', ['batch', 2, 3]), tensor('b', ['batch', 3, 4]), tensor('c', ['batch', 2, 4])],
     [tensor('joined', ['batch', 4, 4]), tensor('difference', ['batch', 2, 4])])
save('tiny-shared-weight', [
    helper.make_node('MatMul', ['x', 'W'], ['left'], name='left_projection'),
    helper.make_node('MatMul', ['x', 'W'], ['right'], name='right_projection'),
    helper.make_node('Add', ['left', 'right'], ['y'], name='merged'),
], [tensor('x', ['batch', 3])], [tensor('y', ['batch', 2])],
     [numpy_helper.from_array(np.ones((3, 2), dtype=np.float32), name='W')])


class TinyResidualCnn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.head = torch.nn.Linear(3, 2)

    def forward(self, x):
        residual = self.conv2(torch.relu(self.conv1(x))) + x
        return self.head(self.pool(residual).flatten(1))


torch.manual_seed(17)
model = TinyResidualCnn().eval()
path = ROOT / 'tiny-residual-cnn.onnx'
torch.onnx.export(model, (torch.zeros(2, 3, 8, 8),), path, dynamo=False,
                  input_names=['image'], output_names=['logits'], opset_version=17,
                  dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}})
onnx_model = onnx.shape_inference.infer_shapes(onnx.load(path))
onnx_model.graph.name = 'Tiny residual CNN'
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, path)
