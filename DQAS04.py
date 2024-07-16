from utils import *
from datasets import MNISTDataLoaders

if os.path.isfile('phase3.history'):
    with open('phase3.history', 'rb') as f:
        nnp,history = pickle.load(f)
        if len(history) > 0:
            chosen_ops, loss, acc= history[-1]

if os.path.isfile('step2.history'):
    with open('step2.history', 'rb') as f:
        history = pickle.load(f)
        if len(history) > 0:
            enable, edges, min_loss, max_acc = history[-1]




def qaoa_block_vag(edges, pnnp, chosen_ops, enable):
    design = dqas_translator(chosen_ops, edges, Arguments.n_repeat, 'full', enable)
    design['pnnp'] = tf.ragged.constant(pnnp, dtype=tf.float32)
    design['edges'] = edges
    dataloader = MNISTDataLoaders(Arguments(), fashion=False)
    val_loss, model_grads, test_acc = dqas_Scheme(design, dataloader, 50)
    return val_loss, test_acc

def DQAS_search(enable, edges, chosen_ops):
    pnnp = make_pnnp(nnp, chosen_ops)
    loss, test_acc = qaoa_block_vag(edges, pnnp, chosen_ops, enable)
    return loss, test_acc


if __name__ == '__main__':
    verbose = None
    set_seed(42)
    history = []
    loss, acc = DQAS_search(enable, edges, chosen_ops)
