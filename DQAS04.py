import pickle

from FusionModel import dqas_translator
from schemes import dqas_Scheme
from utils import *
from datasets import MNISTDataLoaders


def load_state():
    with open('phase23.history', 'rb') as f:
        history = pickle.load(f)
        if len(history) > 0:
            _, nnp, chosen_ops, edges, enable, min_loss, max_acc = history[-1]
    return nnp, chosen_ops, edges, enable, history, min_loss, max_acc


def qaoa_block_vag(edges, pnnp, chosen_ops, enable, epochs):
    design = dqas_translator(chosen_ops, edges, Arguments.n_repeat, 'full', enable)
    design['pnnp'] = tf.ragged.constant(pnnp, dtype=tf.float32)
    design['edges'] = edges
    dataloader = MNISTDataLoaders(Arguments())
    val_loss, model_grads, test_acc = dqas_Scheme(design, dataloader, epochs, save=True)
    return val_loss, test_acc


def DQAS_search(enable, edges, chosen_ops, nnp, epochs):
    pnnp = make_pnnp(nnp, chosen_ops)
    loss, test_acc = qaoa_block_vag(edges, pnnp, chosen_ops, enable, epochs)
    return loss, test_acc


def main(epochs):
    set_seed(42)
    nnp, chosen_ops, edges, enable, history, min_loss, max_acc = load_state()
    loss, acc = DQAS_search(enable, edges, chosen_ops, nnp, epochs)
    with open('phase4.csv', 'w') as f:
        print('loss', 'acc', file=f)
        print(loss, acc, file=f)


if __name__ == '__main__':
    main(epochs=50)
