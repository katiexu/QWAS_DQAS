import copy
# import pennylane as qml
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from math import pi
import torch.nn.functional as F
from torchquantum.encoding import encoder_op_list_name_dict
from Arguments import Arguments
import numpy as np

args = Arguments()

def gen_arch(change_code, base_code):        # start from 1, not 0
    # arch_code = base_code[1:] * base_code[0]
    n_qubits = base_code[0]
    if n_qubits == 7:
        arch_code = [2, 3, 4, 5, 6, 7, 1] * base_code[1]   # qubits * layers
    else:
        arch_code = [2, 3, 4, 1] * base_code[1]
        # arch_code = [2, 3, 4, 5, 6, 7, 8, 9, 10, 1] * base_code[1]   # for MNIST 10
    if change_code != None:
        if type(change_code[0]) != type([]):
            change_code = [change_code]

        for i in range(len(change_code)):
            q = change_code[i][0]  # the qubit changed
            for id, t in enumerate(change_code[i][1:]):
                arch_code[q - 1 + id * n_qubits] = t
    return arch_code

def prune_single(change_code):
    single_dict = {}
    single_dict['current_qubit'] = []
    if change_code != None:
        if type(change_code[0]) != type([]):
            change_code = [change_code]
        length = len(change_code[0])
        change_code = np.array(change_code)
        change_qbit = change_code[:,0] - 1
        change_code = change_code.reshape(-1, length)    
        single_dict['current_qubit'] = change_qbit
        j = 0
        for i in change_qbit:            
            single_dict['qubit_{}'.format(i)] = change_code[:, 1:][j].reshape(-1, 2).transpose(1,0)
            j += 1
    return single_dict

def translator(single_code, enta_code, trainable, base_code):    
    updated_design = {}
    updated_design = prune_single(single_code)
    net = gen_arch(enta_code, base_code) 

    if trainable == 'full' or enta_code == None:
        updated_design['change_qubit'] = None
    else:
        if type(enta_code[0]) != type([]): enta_code = [enta_code]
        updated_design['change_qubit'] = enta_code[-1][0]

    # num of layers
    updated_design['n_layers'] = args.n_layers

    for layer in range(updated_design['n_layers']):
    # categories of single-qubit parametric gates
        for i in range(args.n_qubits):
            updated_design['rot' + str(layer) + str(i)] = 'U3'
        # categories and positions of entangled gates
        for j in range(args.n_qubits):
            if net[j + layer * args.n_qubits] > 0:
                updated_design['enta' + str(layer) + str(j)] = ('CU3', [j, net[j + layer * args.n_qubits]-1])
            else:
                updated_design['enta' + str(layer) + str(j)] = ('CU3', [abs(net[j + layer * args.n_qubits])-1, j])

    updated_design['total_gates'] = updated_design['n_layers'] * args.n_qubits * 2
    return updated_design


def dqas_translator(chosen_ops, edges, repeat, trainable):
    updated_design = {}
    # updated_design = prune_single(single_code)
    # net = gen_arch(enta_code, base_code)

    # if trainable == 'full' or enta_code == None:
    if trainable == 'full':
        updated_design['change_qubit'] = None
    # else:
    #     if type(enta_code[0]) != type([]): enta_code = [enta_code]
    #     updated_design['change_qubit'] = enta_code[-1][0]

    # num of layers
    # updated_design['n_layers'] = args.n_layers
    updated_design['n_layers'] = int(len(chosen_ops)/repeat)
    updated_design['repeat'] = repeat

    # for i in range(len(chosen_ops)):
    #     if len(chosen_ops[i]) != 1:
    for r in range(updated_design['repeat']):
        for layer in range(updated_design['n_layers']):
            # single-qubit gates
            for i in range(args.n_qubits):
                if chosen_ops[layer] == 'rx':
                    updated_design['rot' + str(r) + str(layer) + str(i)] = 'RX'
                elif chosen_ops[layer] == 'ry':
                    updated_design['rot' + str(r) + str(layer) + str(i)] = 'RY'
                elif chosen_ops[layer] == 'rz':
                    updated_design['rot' + str(r) + str(layer) + str(i)] = 'RZ'
                elif chosen_ops[layer] == 'H':
                    updated_design['rot' + str(r) + str(layer) + str(i)] = 'H'
                else:
                    updated_design['rot' + str(r) + str(layer) + str(i)] = 'N/A'

            # entangled gates
            for j in range(args.n_qubits):
                if chosen_ops[layer] == 'xx':
                    updated_design['enta' + str(r) + str(layer) + 'edge0'] = ('RXX', [edges[0][0], edges[0][1]])
                    updated_design['enta' + str(r) + str(layer) + 'edge1'] = ('RXX', [edges[1][0], edges[1][1]])
                    updated_design['enta' + str(r) + str(layer) + 'edge2'] = ('RXX', [edges[2][0], edges[2][1]])
                    updated_design['enta' + str(r) + str(layer) + 'edge3'] = ('RXX', [edges[3][0], edges[3][1]])
                elif chosen_ops[layer] == 'yy':
                    updated_design['enta' + str(r) + str(layer) + 'edge0'] = ('RYY', [edges[0][0], edges[0][1]])
                    updated_design['enta' + str(r) + str(layer) + 'edge1'] = ('RYY', [edges[1][0], edges[1][1]])
                    updated_design['enta' + str(r) + str(layer) + 'edge2'] = ('RYY', [edges[2][0], edges[2][1]])
                    updated_design['enta' + str(r) + str(layer) + 'edge3'] = ('RYY', [edges[3][0], edges[3][1]])
                elif chosen_ops[layer] == 'zz':
                    updated_design['enta' + str(r) + str(layer) + 'edge0'] = ('RZZ', [edges[0][0], edges[0][1]])
                    updated_design['enta' + str(r) + str(layer) + 'edge1'] = ('RZZ', [edges[1][0], edges[1][1]])
                    updated_design['enta' + str(r) + str(layer) + 'edge2'] = ('RZZ', [edges[2][0], edges[2][1]])
                    updated_design['enta' + str(r) + str(layer) + 'edge3'] = ('RZZ', [edges[3][0], edges[3][1]])
                else:
                    updated_design['enta' + str(r) + str(layer) + 'edge0'] = 'N/A'
                    updated_design['enta' + str(r) + str(layer) + 'edge1'] = 'N/A'
                    updated_design['enta' + str(r) + str(layer) + 'edge2'] = 'N/A'
                    updated_design['enta' + str(r) + str(layer) + 'edge3'] = 'N/A'


    return updated_design


def cir_to_matrix(x, y, arch_code):
    qubits = arch_code[0]
    layers = arch_code[1]
    entangle = gen_arch(y, arch_code)
    entangle = np.array([entangle]).reshape(layers, qubits).transpose(1,0)
    single = np.ones((qubits, 2*layers))
    # [[1,1,1,1]
    #  [2,2,2,2]
    #  [3,3,3,3]
    #  [0,0,0,0]]

    if x != None:
        if type(x[0]) != type([]):
            x = [x]    
        x = np.array(x)
        index = x[:, 0] - 1
        index = [int(index[i]) for i in range(len(index))]
        single[index] = x[:, 1:]    
    arch = np.insert(single, [(2 * i) for i in range(1, layers+1)], entangle, axis=1)    
    return arch.transpose(1, 0)


class TQLayer(tq.QuantumModule):
    def __init__(self, arguments, design):
        super().__init__()
        self.args = arguments
        self.design = design
        self.n_wires = self.args.n_qubits
        # self.encoder = tq.GeneralEncoder(encoder_op_list_name_dict['4x4_ryzxy'])
        # self.uploading = [tq.GeneralEncoder(encoder_op_list_name_dict['{}x4_ryzxy'.format(i)]) for i in range(4)]
        self.uploading = [tq.GeneralEncoder(self.data_uploading(i)) for i in range(4)]

        self.gates = tq.QuantumModuleList()
        # self.rots, self.entas = tq.QuantumModuleList(), tq.QuantumModuleList()
        # self.design['change_qubit'] = 3
        # self.q_params_rot, self.q_params_enta = [], []

        q_params = design['pnnp'].to_list()
        self.q_params = sum(q_params, [])
        # for i in range(self.args.n_qubits):
        #     self.q_params_rot.append(pi * torch.rand(self.design['n_layers'], 3))  # each U3 gate needs 3 parameters
        #     self.q_params_enta.append(pi * torch.rand(self.design['n_layers'], 3))  # each CU3 gate needs 3 parameters
        share_weights = [torch.nn.Parameter(torch.tensor([[q]])) for q in self.q_params]

        for r in range(self.design['repeat']):
            for layer in range(self.design['n_layers']):
                trainable  = True
                for q in range(self.n_wires):
                    # # 'trainable' option
                    # if self.design['change_qubit'] is None:
                    #     rot_trainable = True
                    #     enta_trainable = True
                    # elif q == self.design['change_qubit']:
                    #     rot_trainable = True
                    #     enta_trainable = True
                    # else:
                    #     rot_trainable = False
                    #     enta_trainable = False

                    # single-qubit gates
                    if self.design['rot' + str(r) + str(layer) + str(q)] == 'RX':
                        tmp = tq.RX(has_params=True, trainable=trainable)
                        tmp.params = share_weights[layer + (self.design['n_layers'] * r)]
                        self.gates.append(tmp)
                    elif self.design['rot' + str(r) + str(layer) + str(q)] == 'RY':
                        tmp = tq.RY(has_params=True, trainable=trainable)
                        tmp.params = share_weights[layer + (self.design['n_layers'] * r)]
                        self.gates.append(tmp)
                    elif self.design['rot' + str(r) + str(layer) + str(q)] == 'RZ':
                        tmp = tq.RZ(has_params=True, trainable=trainable)
                        tmp.params = share_weights[layer + (self.design['n_layers'] * r)]
                        self.gates.append(tmp)
                    elif self.design['rot' + str(r) + str(layer) + str(q)] == 'H':
                        tmp = tq.U1(has_params=True, trainable=trainable)
                        tmp.params = share_weights[layer + (self.design['n_layers'] * r)]
                        self.gates.append(tmp)
                    else:
                        pass


                # entangled gates
                for edge_idx in range(len(self.design['edges'])):
                    if self.design['enta' + str(r) + str(layer) + 'edge' + str(edge_idx)][0] == 'RXX':
                        tmp = tq.RXX(has_params=True, trainable=trainable)
                        tmp.params = share_weights[layer + (self.design['n_layers'] * r)]
                        self.gates.append(tmp)
                    elif self.design['enta' + str(r) + str(layer) + 'edge' + str(edge_idx)][0] == 'RYY':
                        tmp = tq.RYY(has_params=True, trainable=trainable)
                        tmp.params = share_weights[layer + (self.design['n_layers'] * r)]
                        self.gates.append(tmp)
                    elif self.design['enta' + str(r) + str(layer) + 'edge' + str(edge_idx)][0] == 'RZZ':
                        tmp = tq.RZZ(has_params=True, trainable=trainable)
                        tmp.params = share_weights[layer + (self.design['n_layers'] * r)]
                        self.gates.append(tmp)
                    else:
                        pass

        self.measure = tq.MeasureAll(tq.PauliZ)
    
    def data_uploading(self, qubit):
        input = [      
        {"input_idx": [0], "func": "ry", "wires": [qubit]},        
        {"input_idx": [1], "func": "rz", "wires": [qubit]},        
        {"input_idx": [2], "func": "rx", "wires": [qubit]},        
        {"input_idx": [3], "func": "ry", "wires": [qubit]},  
        ]
        return input

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6)  # 'down_sample_kernel_size' = 6
        x = x.view(bsz, 4, 4)
        # tmp = torch.cat((x.view(bsz, -1), torch.zeros(bsz, 4)), dim=-1)
        # x = tmp.reshape(bsz, -1, 10).transpose(1,2)

        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)

        # encode input image with '4x4_ryzxy' gates
        for j in range(self.n_wires):
            self.uploading[j](qdev, x[:,j])

        for r in range(self.design['repeat']):
            for layer in range(self.design['n_layers']):
                for j in range(self.n_wires):
                    if self.design['rot' + str(r) + str(layer) + str(j)] != 'N/A':
                        self.gates[j + layer * self.n_wires + (self.design['n_layers'] * self.n_wires * r)](qdev, wires=j)
                for edge_idx in range(len(self.design['edges'])):
                    if self.design['enta' + str(r) + str(layer) + 'edge' + str(edge_idx)] != 'N/A':
                        self.gates[edge_idx + layer * self.n_wires + (self.design['n_layers'] * self.n_wires * r)](qdev, wires=self.design['enta' + str(r) + str(layer) + 'edge' + str(edge_idx)][1])

            #     if not (j in self.design['current_qubit'] and self.design['qubit_{}'.format(j)][0][layer] == 0):
            #         self.uploading[j](qdev, x[:,j])
            #     if not (j in self.design['current_qubit'] and self.design['qubit_{}'.format(j)][1][layer] == 0):
            #         self.rots[j + layer * self.n_wires](qdev, wires=j)
            #
            # for j in range(self.n_wires):
            #     if self.design['enta' + str(layer) + str(j)][1][0] != self.design['enta' + str(layer) + str(j)][1][1]:
            #         self.entas[j + layer * self.n_wires](qdev, wires=self.design['enta' + str(layer) + str(j)][1])
        return self.measure(qdev)


class QNet(nn.Module):
    def __init__(self, arguments, design):
        super(QNet, self).__init__()
        self.args = arguments
        self.design = design
        self.QuantumLayer = TQLayer(self.args, self.design)

    def forward(self, x_image):
        exp_val = self.QuantumLayer(x_image)
        output = F.log_softmax(exp_val, dim=1)        
        return output
