import os
import pickle

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

if os.path.isfile('step2.history'):
    with open('step2.history', 'rb') as f:
        history = pickle.load(f)
        enable, edges, loss, acc = min(history, key=lambda x: x[2])

seed = 42
torch.random.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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


def qaoa_block_vag(ops, nnp, preset, repeat, enable, edges):
    nnp = nnp.numpy()
    pnnp = []
    chosen_ops = []
    repeated_preset = preset * repeat
    for i, j in enumerate(repeated_preset):
        if '_' in ops[j]:
            pnnp.append([nnp[2 * i, j], nnp[2 * i + 1, j]])
            underscore_index = ops[j].index('_')
            first_op = ops[j][:underscore_index]
            second_op = ops[j][underscore_index + 1:]
            chosen_ops.append(first_op)
            chosen_ops.append(second_op)
        else:
            pnnp.append([nnp[2 * i, j]])
            chosen_ops.append(ops[j])
        # pnnp.append([nnp[i, j]])
        # chosen_ops.append(ops[j])

    design = dqas_translator2(chosen_ops, edges, repeat, 'full', enable)
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
        if gr[j].shape[0] == 2:
            gmatrix[2 * j, repeated_preset[j]] = gr[j][0]
            gmatrix[2 * j + 1, repeated_preset[j]] = gr[j][1]
        else:  # 1
            gmatrix[2 * j, repeated_preset[j]] = gr[j][0]
        # gmatrix[j, repeated_preset[j]] = gr[j][0]

    gmatrix = tf.constant(gmatrix)
    return val_loss, gmatrix, test_acc


def DQAS_search(stp, nnp, epoch, enable, edges):
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

    for _ in range(batch):
        preset = preset_byprob(prob)
        if noise is not None:
            loss, gnnp, test_acc = qaoa_block_vag(op_pool, nnp + noise, preset, repeat, enable, edges)
        else:
            loss, gnnp, test_acc = qaoa_block_vag(op_pool, nnp, preset, repeat, enable, edges)
        print('\033[34m'+f'batch: {_}/{batch}\tval_loss: {loss:.4f}')
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
    args = Arguments()
    p = 20
    c = 36
    repeat = 2

    # op_pool = ['rx', 'ry', 'rz', 'xx', 'yy', 'zz']
    op_pool = ['rx_rx', 'rx_ry', 'rx_rz', 'rx_xx', 'rx_yy', 'rx_zz',
               'ry_rx', 'ry_ry', 'ry_rz', 'ry_xx', 'ry_yy', 'ry_zz',
               'rz_rx', 'rz_ry', 'rz_rz', 'rz_xx', 'rz_yy', 'rz_zz',
               'xx_rx', 'xx_ry', 'xx_rz', 'xx_xx', 'xx_yy', 'xx_zz',
               'yy_rx', 'yy_ry', 'yy_rz', 'yy_xx', 'yy_yy', 'yy_zz',
               'zz_rx', 'zz_ry', 'zz_rz', 'zz_xx', 'zz_yy', 'zz_zz'
               ]

    result = namedtuple("result", ["epoch", "cand", "loss", "test_acc"])

    verbose = None
    dtype = tf.float32
    batch = 8
    noise = np.random.normal(loc=0.0, scale=0.2, size=[2 * repeat * p, c])
    noise = tf.constant(noise, dtype=tf.float32)
    network_opt = tf.keras.optimizers.Adam(learning_rate=0.1)  # network
    structure_opt = tf.keras.optimizers.Adam(
        learning_rate=0.1, beta_1=0.8, beta_2=0.99
    )  # structure
    stp_regularization = None
    nnp_regularization = None

    epoch_init = 0

    nnp_initial_value = np.random.normal(loc=0.23, scale=0.06, size=[2 * repeat * p, c])
    stp_initial_value = np.zeros([p, c])
    history = []
    if os.path.isfile('step3.history'):
        with open('step3.history', 'rb') as f:
            stp_initial_value, nnp_initial_value, history = pickle.load(f)
        epoch_init = len(history)

    nnp = tf.Variable(initial_value=nnp_initial_value, dtype=dtype)
    stp = tf.Variable(initial_value=stp_initial_value, dtype=dtype)

    enable = np.ones((repeat, 80 // repeat, args.n_qubits), dtype=np.bool_)
    # enable[0,1,1]=False
    avcost1 = 0
    dqas_epoch = 2000

    try:
        for epoch in range(epoch_init, dqas_epoch):
            try:
                print("Epoch: ", epoch)
                stp, nnp, cur_history = DQAS_search(stp, nnp, epoch, enable, edges)
                history.append(cur_history)
                if len(history) > 50 and len(history) % 50 == 0:
                    cur_loss = [h.loss for h in history[-50:]]
                    last_loss = [h.loss for h in history[-100:-50]]
                    eta = abs(sum(cur_loss) / len(cur_loss) - sum(last_loss) / len(last_loss))
                    if eta < 0.001:
                        idx = cur_loss.index(min(cur_loss)) - len(cur_loss)
                        with open('best_net_3', 'wb') as f:
                            pickle.dump((history[idx],edges), f)
                        raise Exception("stop iteration.")
            finally:
                with open('step3.history', 'wb') as f:
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

        with open('history3.csv', 'w') as f:
            print('epoch, loss, test_acc, cand', file=f)
            for h in history:
                print(h.epoch, h.loss, h.test_acc, ' '.join(h.cand), sep=',', file=f)
