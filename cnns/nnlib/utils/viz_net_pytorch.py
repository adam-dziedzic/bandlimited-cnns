# https://gist.githubusercontent.com/wangg12/f11258583ffcc4728eb71adc0f38e832/raw/128ed13d84509c2fdbb2f7a43a21aa3caa523821/viz_net_pytorch.py
from graphviz import Digraph
import re
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Variable
import torchvision.models as models


def make_dot(var):
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if isinstance(var, Variable):
                value = '('+(', ').join(['%d'% v for v in var.size()])+')'
                dot.node(str(id(var)), str(value), fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'previous_functions'):
                for u in var.previous_functions:
                    dot.edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0])
    add_nodes(var)
    return dot


if __name__ == "__main__":
    # inputs = torch.randn(1,3,224,224)
    # imitate CIFAR-10
    inputs = torch.randn(1, 3, 224, 224)
    resnet18 = models.resnet18()
    y = resnet18(Variable(inputs))
    # print(y)

    g = make_dot(y)
    print("print g.source: ", g.source)
    g.view()