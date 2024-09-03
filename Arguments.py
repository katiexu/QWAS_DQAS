class Arguments:
    p = 20
    n_repeat = 4
    n_qubits = 4
    # n_qubits = 10   # MNIST-10
    op_pool_d = ['rx', 'ry', 'rz', 'u3']
    op_pool_s = ['xx', 'yy', 'zz', 'cu3']
    op_pool = op_pool_d + op_pool_s
    c = len(op_pool)
    n_layers = p * n_repeat

    fashion = False   # MNIST or Fashion
    kernel = 6  # set to 6 for MNIST-4; set to 4 for MNIST=10

    def __init__(self):
        self.device = 'cpu'

        self.qlr = 0.01

        self.backend = 'tq'
        self.digits_of_interest = [0, 1, 2, 3]
        # self.digits_of_interest = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]    # MNIST-10
        self.train_valid_split_ratio = [0.95, 0.05]
        self.center_crop = 24
        self.resize = 28
        self.batch_size = 256
