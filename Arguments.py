class Arguments:
    p = 20
    n_repeat = 6
    n_qubits = 4
    op_pool_d=['rx', 'ry', 'rz','u3']
    op_pool_s=['xx', 'yy', 'zz', 'cu3']
    op_pool = op_pool_d+op_pool_s
    c = len(op_pool)
    n_layers = p * n_repeat

    def __init__(self):
        self.device='cpu'

        self.qlr = 0.01

        self.backend = 'tq'
        self.digits_of_interest = [0, 1, 2, 3]
        self.train_valid_split_ratio = [0.95, 0.05]
        self.center_crop = 24
        self.resize = 28
        self.batch_size = 256
