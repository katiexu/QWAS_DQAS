import os.path

from utils import *
from datasets import MNISTDataLoaders

def init_state():
    with open('phase1.history', 'rb') as f:
        history = pickle.load(f)
        cur_acc = [acc for _, _, _, acc, _, _ in history]
        idx = cur_acc.index(max(cur_acc)) - len(cur_acc)

        _, nnp_initial_value, _, _, edges, chosen_ops = history[idx]
        nnp = tf.Variable(initial_value=nnp_initial_value, dtype=tf.float32)

        print(display.BLUE,'load from phase1:')
        print(f'\tedges: {edges[0][0]}')
        print(f'\tchosen_ops: {chosen_ops[:Arguments.p]} * {Arguments.n_repeat}{display.RESET}')
    if os.path.isfile('phase3.history'):
        with open('phase3.history', 'rb') as f:
            _,history = pickle.load(f)
            his=max(history, key=lambda x:x[2])
            chosen_ops,loss,acc=his
            print(display.YELLOW, 'load form phase3:')
            print(f'\tchosen_ops: {chosen_ops[:Arguments.p]} * {Arguments.n_repeat}{display.RESET}')


    enable = np.ones((Arguments.n_repeat, Arguments.p, Arguments.n_qubits), dtype=np.bool_)
    history = []
    min_loss = 5
    max_acc = 0
    if os.path.isfile('phase2.history'):
        with open('phase2.history', 'rb') as f:
            history = pickle.load(f)
            if len(history) > 0:
                enable, edges, min_loss, max_acc = history[-1]
    return nnp, chosen_ops, edges, enable, history, min_loss, max_acc


def qaoa_block_vag(edges, pnnp, chosen_ops, enable,scheme_epochs):
    design = dqas_translator(chosen_ops, edges, Arguments.n_repeat, 'full', enable)
    design['pnnp'] = tf.ragged.constant(pnnp, dtype=tf.float32)
    design['edges'] = edges
    dataloader = MNISTDataLoaders(Arguments(), fashion=False)
    val_loss, model_grads, test_acc = dqas_Scheme(design, dataloader, scheme_epochs)
    return val_loss, test_acc


def DQAS_search(enable, edges,nnp,chosen_ops,scheme_epochs):
    pnnp = make_pnnp(nnp, chosen_ops)
    loss, test_acc = qaoa_block_vag(edges, pnnp, chosen_ops, enable,scheme_epochs)
    return loss, test_acc


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


def main(epochs = 2000, limit = 30,scheme_epochs=10):
    nnp, chosen_ops, edges, enable, history, min_loss, max_acc = init_state()

    curr = 0
    tmpenable, tmpedges = enable, edges
    try:
        for epoch in range(len(history), epochs):
            try:
                print("Epoch: ", epoch)
                loss, acc = DQAS_search(tmpenable, tmpedges,nnp,chosen_ops,scheme_epochs)
                curr += 1
                if acc > max_acc:
                    curr = 0
                    max_acc = acc
                    enable, edges = tmpenable, tmpedges
                    history.append((enable, edges, loss, acc))
                print('\033[34m' + f'val_loss: {loss:.4f}\t acc: {max_acc:.4f} \tstep: {curr}/{limit}\033[0m')
                if curr == limit:
                    print(display.RED + "stop iteration phase2." + display.RESET)
                    break
                tmpenable, tmpedges = change(enable, edges)
            finally:
                with open('phase2.history', 'wb') as f:
                    pickle.dump((history), f)
    finally:
        with open('phase2_result.csv', 'w') as f:
            print('epoch', 'loss', 'acc', sep=',\t', file=f)
            for i in range(len(history)):
                print(i, history[i][2], history[i][3], sep=',\t', file=f)
if __name__ == '__main__':
    main()
