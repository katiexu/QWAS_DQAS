import os
import pickle
import copy
import numpy as np
from tensorcircuit.applications.graphdata import regular_graph_generator
import tensorflow as tf
from schemes import dqas_Scheme
from FusionModel import dqas_translator, dqas_translator2
import inspect
from collections import namedtuple
from matplotlib import pyplot as plt
from Arguments import Arguments
import random
import torch

seed = 42
torch.random.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

args = Arguments()
with open('step.history', 'rb') as f:
    result = namedtuple("result", ["epoch", "cand", "loss", "test_acc"])
    stp_initial_value, nnp_initial_value, history, edges = pickle.load(f)
    cur_acc = [h.test_acc for h in history]
    idx = cur_acc.index(max(cur_acc)) - len(cur_acc) -1
    his = history[idx]
    edges = edges[idx]
    nnp = tf.Variable(initial_value=nnp_initial_value, dtype=tf.float32)
    stp = tf.Variable(initial_value=stp_initial_value, dtype=tf.float32)
    print(his.cand * 6)
    chosen_ops = his.cand * 6
    print(edges)


def preset_byprob(prob):
    preset = []
    p = prob.shape[0]
    c = prob.shape[1]
    for i in range(p):
        j = np.random.choice(np.arange(c), p=np.array(prob[i]))
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


def get_var(name):
    """
    call in customized functions and grab variable within DQAF framework function by var name str

    :param name:
    :return:
    """
    return inspect.stack()[2][0].f_locals[name]


def qaoa_block_vag(edges, pnnp, chosen_ops, enable):
    design = dqas_translator2(chosen_ops, edges, repeat, 'full', enable)
    design['pnnp'] = tf.ragged.constant(pnnp, dtype=dtype)
    design['edges'] = edges

    val_loss, model_grads, test_acc = dqas_Scheme(design, 'MNIST', 'init', 10)
    return val_loss, test_acc


def DQAS_search(enable, edges):
    pnnp = make_pnnp(nnp,op_pool)
    loss, test_acc = qaoa_block_vag(edges, pnnp, chosen_ops, enable)
    return loss, test_acc

def make_pnnp(nnp,ops:list):
    nnp = nnp.numpy()
    pnnp = []
    for i, op in enumerate(chosen_ops):
        j=ops.index(op)
        if 'u' in op:
            pnnp.append([nnp[i, j]])
        else:
            pnnp.append([nnp[i, j][0:1]])
    return pnnp

def change(enable, edges):
    newenable = enable.copy()
    newedges = edges.copy()
    for r in range(enable.shape[0]):
        for l in range(enable.shape[1]):
            idx = random.randrange(-4, enable.shape[2])
            if idx >= 0:
                newenable[r, l, idx] = (not newenable[r, l, idx])

            idx = random.randrange(-4, enable.shape[2])
            if idx >= 0:
                newedges[r, l, idx, 0] = random.randrange(enable.shape[2])
                newedges[r, l, idx, 1] = random.randrange(enable.shape[2])
    assert not (enable == newenable).all()
    assert not (edges == newedges).all()
    return newenable, newedges


if __name__ == '__main__':
    args = Arguments()
    p = 20

    repeat = 6
    op_pool = ['rx', 'ry', 'rz', 'xx', 'yy', 'zz', 'u3', 'cu3']
    c = 8

    verbose = None
    dtype = tf.float32

    enable = np.ones((repeat, p, args.n_qubits), dtype=np.bool_)
    edges = np.array([[edges.copy() for l in range(enable.shape[1])] for r in range(repeat)])
    history = []

    min_loss = 5
    max_acc = 0
    if os.path.isfile('step2.history'):
        with open('step2.history', 'rb') as f:
            history = pickle.load(f)
            if len(history) > 0:
                enable, edges, min_loss, max_acc = history[-1]
    edges = np.array(edges)
    tmpenable, tmpedges = enable, edges
    dqas_epoch = 2000

    limit = 30
    curr = 0
    try:
        for epoch in range(len(history), dqas_epoch):
            try:
                print("Epoch: ", epoch)
                loss, acc = DQAS_search(tmpenable, tmpedges)
                print('\033[34m' + f'val_loss: {loss:.4f}\t acc: {acc:.4f} \tstep: {curr}/{limit}\033[0m')
                curr += 1
                if acc > max_acc:
                    curr = 0
                    max_acc = acc
                    enable, edges = tmpenable, tmpedges
                    history.append((enable, edges, loss, acc))
                if curr == limit:
                    raise Exception('stop iteration')
                tmpenable, tmpedges = change(enable, edges)
            finally:
                with open('step2.history', 'wb') as f:
                    pickle.dump((history), f)
    finally:
        with open('step2_result.csv', 'w') as f:
            print('epoch', 'loss', 'acc', sep=',\t', file=f)
            for i in range(len(history)):
                print(i, history[i][2], history[i][3], sep=',\t', file=f)
