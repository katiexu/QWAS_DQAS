import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import numpy as np
from tensorcircuit.applications.graphdata import regular_graph_generator
import tensorflow as tf
from schemes import dqas_Scheme
from FusionModel import dqas_translator
import inspect
from collections import namedtuple
from matplotlib import pyplot as plt
from Arguments import Arguments
import random
import torch



def get_chosen_ops(chosen_idxs):
    chosen_ops = []
    for i, j in enumerate(chosen_idxs):
        if 'u' in Arguments.op_pool[j]:
            chosen_ops.append(Arguments.op_pool[j])
        else:
            chosen_ops.append(Arguments.op_pool[j])
    return chosen_ops
def make_pnnp(nnp,chosen_ops):
    nnp = nnp.numpy()
    pnnp = []
    for i, op in enumerate(chosen_ops):
        j=Arguments.op_pool.index(op)
        if 'u' in op:
            pnnp.append([nnp[i, j]])
        else:
            pnnp.append([nnp[i, j][0:1]])
    return pnnp
def set_seed(seed=42):
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def preset_byprob(prob):
    preset = []
    p = prob.shape[0]
    c = prob.shape[1]
    for i in range(p):
        j = np.random.choice(np.arange(c), p=np.array(prob[i]))
        #j = np.random.choice(np.arange(c))
        preset.append(j)
    return preset

def get_preset(stp):
    return tf.argmax(stp, axis=1)

def repr_op(element):
    if isinstance(element, str):
        return element
    if isinstance(element, list) or isinstance(element, tuple):
        return str(tuple([repr_op(e) for e in element]))
    if callable(element.__repr__):
        return element.__repr__()  # type: ignore
    else:
        return element.__repr__  # type: ignore

class display():
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'
