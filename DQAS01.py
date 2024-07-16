from utils import *
from datasets import MNISTDataLoaders




def load_state():
    nnp_initial_value = np.random.normal(loc=0.23, scale=0.06, size=[Arguments.n_layers, len(Arguments.op_pool), 3])
    stp_initial_value = np.zeros([Arguments.p, len(Arguments.op_pool)])

    history = []
    if os.path.isfile('phase1.history'):
        with open('phase1.history', 'rb') as f:
            history = pickle.load(f)
            if len(history) > 0:
                stp_initial_value, nnp_initial_value, cost, acc, edges, chosen_ops = history[-1]

    nnp = tf.Variable(initial_value=nnp_initial_value, dtype=tf.float32)
    stp = tf.Variable(initial_value=stp_initial_value, dtype=tf.float32)
    return stp, nnp, history


def qaoa_block_vag(edges, nnp, preset, repeat, enable, dataloader,Scheme_epoch):
    repeated_preset = preset * repeat
    chosen_ops = get_chosen_ops(repeated_preset)
    pnnp = make_pnnp(nnp, chosen_ops)

    design = dqas_translator(chosen_ops, edges, repeat, 'full', enable)
    design['pnnp'] = tf.ragged.constant(pnnp, dtype=tf.float32)
    design['edges'] = edges
    val_loss, model_grads, test_acc = dqas_Scheme(design, dataloader, Scheme_epoch)
    val_loss = tf.constant(val_loss, dtype=tf.float32)

    gr = tf.constant(model_grads, dtype=tf.float32)
    gr = design['pnnp'].with_values(gr)
    g_nnp = np.zeros_like(nnp.numpy())
    for j in range(gr.shape[0]):
        g_nnp[j, repeated_preset[j]] = gr[j][0]
    g_nnp = tf.constant(g_nnp)

    return val_loss, g_nnp, test_acc, chosen_ops


def update_stp(stp, deri_stp):
    stp_penalty_gradient = 0
    newstp = tf.Variable(initial_value=stp.numpy().copy(), dtype=tf.float32)
    batched_gs = tf.math.reduce_mean(tf.convert_to_tensor(deri_stp, dtype=tf.float32), axis=0)
    structure_opt = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.8, beta_2=0.99)  # structure
    structure_opt.apply_gradients(
        zip([batched_gs + stp_penalty_gradient], [newstp])
    )
    return newstp


def update_nnp(nnp, deri_nnp):
    newnnp = tf.Variable(initial_value=nnp.numpy().copy(), dtype=tf.float32)
    nnp_penalty_gradient = 0
    batched_gnnp = tf.math.reduce_mean(tf.convert_to_tensor(deri_nnp, dtype=tf.float32), axis=0)
    network_opt = tf.keras.optimizers.Adam(learning_rate=0.1)  # network
    network_opt.apply_gradients(
        zip([batched_gnnp + nnp_penalty_gradient], [newnnp])
    )
    return newnnp


def generate_edges():
    edges_input = []
    for i in range(Arguments.n_qubits):
        edges_input.append(random.sample(range(Arguments.n_qubits), 2))
    print('\tedges:', edges_input)
    edges_input = np.array(
        [[edges_input.copy() for l in range(Arguments.p)] for r in range(Arguments.n_repeat)])
    return edges_input


def DQAS_search(stp, nnp,scheme_epochs):
    prob = tf.math.exp(stp) / tf.tile(tf.math.reduce_sum(tf.math.exp(stp), axis=1)[:, tf.newaxis],
                                      [1, len(Arguments.op_pool)])  # softmax categorical probability
    preset = preset_byprob(prob)
    print('chosen_ops: ', get_chosen_ops(preset))
    deri_stp, deri_nnp, costl, test_acc_list, edges, ops_list = [], [], [], [], [], []
    dataloader = MNISTDataLoaders(Arguments(), fashion=False)
    enable = np.ones((Arguments.n_repeat, Arguments.p, Arguments.n_qubits), dtype=np.bool_)
    for _ in range(8):
        edges_input = generate_edges()

        loss, gnnp, test_acc, chosen_ops = qaoa_block_vag(edges_input, nnp, preset, Arguments.n_repeat, enable,
                                                              dataloader,scheme_epochs)

        gs = tf.tensor_scatter_nd_add(
            tf.cast(-prob, dtype=tf.float32),
            tf.constant(list(zip(range(Arguments.p), preset))),
            tf.ones([Arguments.p], dtype=tf.float32),
        )
        deri_stp.append(
            (tf.cast(loss, dtype=tf.float32) - tf.cast(0, dtype=tf.float32)) * tf.cast(gs, dtype=tf.float32))
        deri_nnp.append(gnnp)

        costl.append(loss.numpy())
        test_acc_list.append(test_acc)
        edges.append(edges_input)
        ops_list.append(chosen_ops)

    newnnp = update_nnp(nnp, deri_nnp)
    newstp = update_stp(stp, deri_stp)

    # cand_preset = get_preset(stp).numpy()
    # cand_preset_repr = [repr_op(Arguments.op_pool[f]) for f in cand_preset]
    # print("best candidates so far:", cand_preset_repr)

    max_idx = np.argmax(test_acc_list)

    return newstp, newnnp, costl[max_idx], test_acc_list[max_idx], edges[max_idx], ops_list[max_idx]


def main(epochs = 200,threshold = 20, scheme_epochs = 1):
    set_seed(42)

    stp, nnp, history = load_state()

    epoch_init = len(history)
    newstp, newnnp = stp, nnp
    try:
        for epoch in range(epoch_init, epochs):
            try:
                stp = newstp
                nnp = newnnp
                print("DQAS Epoch: ", epoch)
                newstp, newnnp, cost, acc, edges, chosen_ops = DQAS_search(stp, nnp,scheme_epochs)
                history.append((stp, nnp, cost, acc, edges, chosen_ops))
                if len(history) > threshold and len(history) % threshold == 0:
                    cur_loss = [h.loss for h in history[-threshold:]]
                    last_loss = [h.loss for h in history[-threshold * 2:-threshold]]
                    eta = abs(sum(cur_loss) / len(cur_loss) - sum(last_loss) / len(last_loss))
                    if eta < 0.001:
                        print(display.RED+"stop iteration phase1"+display.RESET)
                        break
            finally:
                with open('phase1.history', 'wb') as f:
                    pickle.dump((history), f)

    finally:
        epochs = np.arange(len(history))
        data = np.array([r[2] for r in history])
        plt.figure()
        plt.plot(epochs, data)
        plt.xlabel("epoch")
        plt.ylabel("objective (loss)")
        plt.savefig("loss_plot.pdf")
        plt.close()

        test_acc_data = np.array([r[3] for r in history])
        plt.figure()
        plt.plot(epochs, test_acc_data)
        plt.xlabel("epoch")
        plt.ylabel("test acc")
        plt.savefig("test_acc_plot.pdf")
        plt.close()

        with open('phase1.csv', 'w') as f:
            print('epoch, loss, test_acc, edges', file=f)
            for epoch in range(len(history)):
                stp, nnp, cost, acc, edges, chosen_ops = history[epoch]
                print(epoch, cost, acc, f'"{edges[0][0].tolist()}"', sep=',', file=f)


if __name__ == '__main__':
    main()
