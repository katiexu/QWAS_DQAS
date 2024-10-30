import numpy as np

class Arguments:
    task = 'Ansatz Depth Classification'
    # task = 'MNIST-4'
    seed = 42
    device = 'cpu'
    backend = 'tq'
    op_pool_d = ['rx', 'ry', 'rz', 'u3']
    op_pool_s = ['xx', 'yy', 'zz', 'cu3']
    op_pool = op_pool_d + op_pool_s
    c = len(op_pool)

    if task == 'MNIST-4':
        fashion = False
        n_qubits = 4
        p = 20
        n_repeat = 1
        n_layers = p * n_repeat
        qlr = 0.01
        kernel = 6
        digits_of_interest = [0, 1, 2, 3]
        train_valid_split_ratio = [0.95, 0.05]
        center_crop = 24
        resize = 28
        batch_size = 256
    elif task == 'Fashion-4':
        fashion = True
        n_qubits = 4
        p = 20
        n_repeat = 1
        n_layers = p * n_repeat
        qlr = 0.01
        kernel = 6
        digits_of_interest = [0, 1, 2, 3]
        train_valid_split_ratio = [0.95, 0.05]
        center_crop = 24
        resize = 28
        batch_size = 256
    elif task == 'MNIST-10':
        fashion = False
        n_qubits = 10
        p = 20
        n_repeat = 1
        n_layers = p * n_repeat
        qlr = 0.01
        kernel = 4
        digits_of_interest = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        train_valid_split_ratio = [0.95, 0.05]
        center_crop = 24
        resize = 28
        batch_size = 256
    elif task == 'Fashion-10':
        fashion = True
        n_qubits = 10
        p = 20
        n_repeat = 1
        n_layers = p * n_repeat
        qlr = 0.01
        kernel = 4
        digits_of_interest = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        train_valid_split_ratio = [0.95, 0.05]
        center_crop = 24
        resize = 28
        batch_size = 256
    elif task == 'Ansatz Depth Classification':
        n_qubits = 4
        p = 20
        n_repeat = 1
        n_layers = p * n_repeat
        qlr = 0.01
        m = 2  # input m-copies of each state
        num_data0 = 300
        num_data1 = 300
        label0 = 1
        label1 = 6
        params0 = np.load('Hardware_Efficient/4_Qubits/Depth_1/hwe_4q_ps_5_1_weights.npy')
        params1 = np.load('Hardware_Efficient/4_Qubits/Depth_6/hwe_4q_ps_5_6_weights.npy')
    elif task == 'Entangled State Classification':
        n_qubits = 4
        p = 20
        n_repeat = 1
        n_layers = p * n_repeat
        qlr = 0.01
        m = 2  # input m-copies of each state
        num_data0 = 300
        num_data1 = 300
        label0 = 0.05
        label1 = 0.25
        params0 = np.load('Hardware_Efficient/4_Qubits/Depth_1/hwe_4q_ps_5_1_weights.npy')
        params1 = np.load('Hardware_Efficient/4_Qubits/Depth_1/hwe_4q_ps_25_1_weights.npy')
        depth = 1
    else:
        raise ValueError(f'Unknown task: {task}')
