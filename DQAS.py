import os
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

seed = 42
torch.random.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

args = Arguments()


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

def record():
    return result(
        get_var("epoch"), get_var("cand_preset_repr"), get_var("avcost1").numpy(), get_var("avtestacc").numpy()
        )


def qaoa_block_vag(gdata, ops, nnp, preset, repeat):
    nnp = nnp.numpy()
    pnnp = []
    chosen_ops = []
    repeated_preset = preset * repeat
    for i, j in enumerate(repeated_preset):
        # if '_' in ops[j]:
        #     pnnp.append([nnp[2 * i, j], nnp[2 * i + 1, j]])
        #     chosen_ops.append(ops[j][0:2])
        #     chosen_ops.append(ops[j][3:5])
        # else:
        #     pnnp.append([nnp[2 * i, j]])
        #     chosen_ops.append(ops[j])
        pnnp.append([nnp[i, j]])
        chosen_ops.append(ops[j])
    edges = []
    for e in gdata.edges:
        edges.append(e)
    design = dqas_translator(chosen_ops, edges, repeat,'full')
    # pnnp = array_to_tensor(np.array(pnnp))  # complex
    # pnnp = tf.ragged.constant(pnnp, dtype=getattr(tf, cons.dtypestr))
    design['pnnp'] = tf.ragged.constant(pnnp, dtype=dtype)
    design['preset'] = preset
    design['edges'] = edges

    val_loss, model_grads, test_acc = dqas_Scheme(design, 'MNIST', 'init', 1)
    val_loss = tf.constant(val_loss, dtype=dtype)
    gr = tf.constant(model_grads, dtype=dtype)
    gr = design['pnnp'].with_values(gr)

    gmatrix = np.zeros_like(nnp)
    for j in range(gr.shape[0]):
        # if gr[j].shape[0] == 2:
        #     gmatrix[2 * j, repeated_preset[j]] = gr[j][0]
        #     gmatrix[2 * j + 1, repeated_preset[j]] = gr[j][1]
        # else:  # 1
        #     gmatrix[2 * j, repeated_preset[j]] = gr[j][0]
        gmatrix[j, repeated_preset[j]] = gr[j][0]

    gmatrix = tf.constant(gmatrix)
    return val_loss, gmatrix, test_acc

def DQAS_search(stp, nnp, epoch):
    prob = tf.math.exp(stp) / tf.tile(
        tf.math.reduce_sum(tf.math.exp(stp), axis=1)[:, tf.newaxis], [1, c]
    )  # softmax categorical probability

    deri_stp = []
    deri_nnp = []
    avcost2 = 0
    costl = []
    test_acc_list = []

    if stp_regularization is not None:
        stp_penalty_gradient = stp_regularization(stp, nnp)
        if verbose:
            print("stp_penalty_gradient:", stp_penalty_gradient.numpy())
    else:
        stp_penalty_gradient = 0.0
    if nnp_regularization is not None:
        nnp_penalty_gradient = nnp_regularization(stp, nnp)
        if verbose:
            print("nnpp_penalty_gradient:", nnp_penalty_gradient.numpy())
    else:
        nnp_penalty_gradient = 0.0

    for _, gdata in zip(range(batch), g):
        preset = preset_byprob(prob)
        if noise is not None:
            loss, gnnp, test_acc = qaoa_block_vag(gdata, op_pool, nnp + noise, preset, repeat)
        else:
            loss, gnnp, test_acc = qaoa_block_vag(gdata, op_pool, nnp, preset, repeat)

        gs = tf.tensor_scatter_nd_add(
            tf.cast(-prob, dtype=dtype),
            tf.constant(list(zip(range(p), preset))),
            tf.ones([p], dtype=dtype),
        )  # \nabla lnp
        deri_stp.append(
            (tf.cast(loss, dtype=dtype) - tf.cast(avcost2, dtype=dtype))
            * tf.cast(gs, dtype=dtype)
        )
        deri_nnp.append(gnnp)
        costl.append(loss.numpy())
        test_acc_list.append(test_acc)

    avcost1 = tf.convert_to_tensor(np.mean(costl))
    avtestacc = tf.convert_to_tensor(np.mean(test_acc_list))

    print(
        "batched average loss: ",
        np.mean(costl),
        " batched loss std: ",
        np.std(costl),
        "\nnew baseline: ",
        avcost1.numpy(),  # type: ignore
    )

    batched_gs = tf.math.reduce_mean(
        tf.convert_to_tensor(deri_stp, dtype=dtype), axis=0
    )
    batched_gnnp = tf.math.reduce_mean(
        tf.convert_to_tensor(deri_nnp, dtype=dtype), axis=0
    )
    if verbose:
        print("batched gradient of stp: \n", batched_gs.numpy())
        print("batched gradient of nnp: \n", batched_gnnp.numpy())

    network_opt.apply_gradients(
        zip([batched_gnnp + nnp_penalty_gradient], [nnp])
    )
    structure_opt.apply_gradients(
        zip([batched_gs + stp_penalty_gradient], [stp])
    )
    if verbose:
        print(
            "strcuture parameter: \n",
            stp.numpy(),
            "\n network parameter: \n",
            nnp.numpy(),
        )

    cand_preset = get_preset(stp).numpy()
    cand_preset_repr = [repr_op(op_pool[f]) for f in cand_preset]
    print("best candidates so far:", cand_preset_repr)

    return stp, nnp, record()


if __name__ == '__main__':

    p = 4
    c = 6
    repeat = 20

    op_pool = ['rx', 'ry', 'rz', 'xx', 'yy', 'zz']
    g = regular_graph_generator(n=4, d=2,seed=seed)
    result = namedtuple("result", ["epoch", "cand", "loss", "test_acc"])

    verbose = None
    dtype = tf.float32
    batch = 8
    noise = np.random.normal(loc=0.0, scale=0.2, size=[repeat * p, c])
    noise = tf.constant(noise, dtype=tf.float32)
    network_opt = tf.keras.optimizers.Adam(learning_rate=0.1)  # network
    structure_opt = tf.keras.optimizers.Adam(
        learning_rate=0.1, beta_1=0.8, beta_2=0.99
    )  # structure
    stp_regularization = None
    nnp_regularization = None

    epoch_init = 0

    nnp_initial_value = np.random.normal(loc=0.23, scale=0.06, size=[repeat * p, c])
    stp_initial_value = np.zeros([p, c])
    history = []
    if os.path.isfile('step'):
        with open('step', 'rb') as f:
            stp_initial_value, nnp_initial_value, history = pickle.load(f)
        epoch_init = len(history)

    nnp = tf.Variable(initial_value=nnp_initial_value, dtype=dtype)
    stp = tf.Variable(initial_value=stp_initial_value, dtype=dtype)

    avcost1 = 0
    dqas_epoch = 2000

    try:
        for epoch in range(epoch_init, dqas_epoch):
            try:
                print("Epoch: ", epoch)
                stp, nnp, cur_history = DQAS_search(stp, nnp,epoch)
                history.append(cur_history)
            finally:
                with open('step', 'wb') as f:
                    pickle.dump((stp, nnp, history), f)
    finally:
        epochs = np.arange(len(history))
        data = np.array([r.loss for r in history])
        plt.figure()
        plt.plot(epochs, data)
        plt.xlabel("epoch")
        plt.ylabel("objective (loss)")
        plt.savefig("loss_plot.pdf")
        plt.close()

        test_acc_data = np.array([r.test_acc for r in history])
        plt.figure()
        plt.plot(epochs, test_acc_data)
        plt.xlabel("epoch")
        plt.ylabel("test acc")
        plt.savefig("test_acc_plot.pdf")
        plt.close()

        with open('history.csv','w') as f:
            print('epoch, loss, test_acc, cand',file=f)
            for h in history:
                print(h.epoch, h.loss, h.test_acc, ' '.join(h.cand), sep=',', file=f)






