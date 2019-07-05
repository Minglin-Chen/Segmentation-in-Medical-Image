
from .evaluate_ICH210 import eval_op_ICH210
from .evaluate_BraTS19 import eval_op_BraTS19

eval_op_zoo = {
    'eval_op_ICH210': eval_op_ICH210,
    'eval_op_BraTS19': eval_op_BraTS19}


def eval_op_provider(name):

    return eval_op_zoo[name]