from utils import *
from datasets import MNISTDataLoaders




def load_state():
    with open('phase23.history', 'rb') as f:
        history = pickle.load(f)
        if len(history) > 0:
            _, nnp, chosen_ops, edges, enable, min_loss, max_acc = history[-1]
    return nnp, enable, edges, chosen_ops, history, min_loss, max_acc


def qaoa_block_vag(edges, pnnp, chosen_ops, enable, scheme_epochs):
    design = dqas_translator(chosen_ops, edges, Arguments.n_repeat, 'full', enable)
    design['pnnp'] = tf.ragged.constant(pnnp, dtype=tf.float32)
    design['edges'] = edges
    dataloader = MNISTDataLoaders(Arguments())
    val_loss, model_grads, test_acc = dqas_Scheme(design, dataloader, scheme_epochs)
    return val_loss, test_acc


def DQAS_search(enable, edges, nnp, chosen_ops,scheme_epochs):
    pnnp = make_pnnp(nnp, chosen_ops)
    loss, test_acc = qaoa_block_vag(edges, pnnp, chosen_ops, enable,scheme_epochs)
    return loss, test_acc


def change_ops(chosen_ops):
    new_chosen_ops = []
    for op in chosen_ops[:Arguments.p]:
        if op in Arguments.op_pool_d:
            new_chosen_ops.append(random.choice(Arguments.op_pool_d))
        else:
            new_chosen_ops.append(random.choice(Arguments.op_pool_s))
    return new_chosen_ops * Arguments.n_repeat


def main(epochs = 2000,limit = 30, scheme_epochs=10):
    set_seed(42)
    nnp, enable, edges, chosen_ops, history, min_loss, max_acc = load_state()
    start = len([h for h in history[::-1] if h[0] == 'phase3'])
    curr = 0
    try:
        for epoch in range(start, epochs):
            try:
                print("Phase3 Epoch: ", epoch)
                chosen_ops = change_ops(chosen_ops)
                print(f'chosen_ops: {chosen_ops[:Arguments.p]} * {Arguments.n_repeat}')
                loss, acc = DQAS_search(enable, edges, nnp, chosen_ops,scheme_epochs)
                curr += 1
                if acc > max_acc:
                    curr = 0
                    max_acc = acc
                    history.append(('phase3', nnp, chosen_ops, edges, enable, history, min_loss, max_acc))
                print('\033[34m' + f'val_loss: {loss:.6f}\t max_acc: {max_acc:.6f} \tstep: {curr}/{limit}\033[0m')
                if curr == limit:
                    print(display.RED + "stop iteration phase3." + display.RESET)
                    break
            finally:
                with open('phase23.history', 'wb') as f:
                    pickle.dump((history), f)
    finally:
        with open('phase23_result.csv', 'w') as f:
            print('type','epoch','chosen ops', 'loss', 'acc', sep=',\t', file=f)
            for i in range(len(history)):
                print(history[i][0],i, ' '.join(history[i][2]),history[i][-2], history[i][-1], sep=',\t', file=f)
if __name__ == '__main__':
    main()
