import copy
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, f1_score
from datasets import MNISTDataLoaders, MOSIDataLoaders
from FusionModel import QNet

from Arguments import Arguments
import random

class display():
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'

def get_param_num(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total:', total_num, 'trainable:', trainable_num)





def train(model, data_loader, optimizer, criterion, args):
    model.train()
    for feed_dict in data_loader:
        images = feed_dict['image'].to(args.device)
        targets = feed_dict['digit'].to(args.device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()


def test(model, data_loader, criterion, args):
    model.eval()
    total_loss = 0
    target_all = torch.Tensor()
    output_all = torch.Tensor()
    with torch.no_grad():
        for feed_dict in data_loader:
            images = feed_dict['image'].to(args.device)
            targets = feed_dict['digit'].to(args.device)
            output = model(images)
            instant_loss = criterion(output, targets).item()
            total_loss += instant_loss
            target_all = torch.cat((target_all, targets), dim=0)
            output_all = torch.cat((output_all, output), dim=0)
    total_loss /= len(data_loader)
    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size

    return total_loss, accuracy


def evaluate(model, data_loader, args):
    model.eval()
    metrics = {}

    with torch.no_grad():
        for feed_dict in data_loader:
            images = feed_dict['image'].to(args.device)
            targets = feed_dict['digit'].to(args.device)
            output = model(images)

    _, indices = output.topk(1, dim=1)
    masks = indices.eq(targets.view(-1, 1).expand_as(indices))
    size = targets.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    metrics = accuracy
    return metrics


# def Scheme(design, task, weight='base', epochs=None, verbs=None, save=None):
#     random.seed(42)
#     np.random.seed(42)
#     torch.random.manual_seed(42)
#
#     args = Arguments(task)
#     if epochs == None:
#         epochs = args.epochs
#
#     if task == 'MOSI':
#         dataloader = MOSIDataLoaders(args)
#     else:
#         dataloader = MNISTDataLoaders(args, task)
#
#     train_loader, val_loader, test_loader = dataloader
#     model = QNet(args, design).to(args.device)
#     if weight != 'init':
#         if weight != 'base':
#             model.load_state_dict(weight, strict= False)
#         else:
#             model.load_state_dict(torch.load('weights/base_fashion'))
#             # model.load_state_dict(torch.load('weights/mnist_best_3'))
#     criterion = nn.NLLLoss()
#
#     optimizer = optim.Adam(model.QuantumLayer.parameters(), lr=args.qlr)
#     train_loss_list, val_loss_list = [], []
#     best_val_loss = 0
#
#     start = time.time()
#     for epoch in range(epochs):
#         try:
#             train(model, train_loader, optimizer, criterion, args)
#         except Exception as e:
#             print('No parameter gate exists')
#         train_loss = test(model, train_loader, criterion, args)
#         train_loss_list.append(train_loss)
#         val_loss = evaluate(model, val_loader, args)
#         val_loss_list.append(val_loss)
#         metrics = evaluate(model, test_loader, args)
#         val_loss = 0.5 *(val_loss+train_loss[-1])
#         if val_loss > best_val_loss:
#             best_val_loss = val_loss
#             if not verbs: print(epoch, train_loss, val_loss_list[-1], metrics, 'saving model')
#             best_model = copy.deepcopy(model)
#         else:
#             if not verbs: print(epoch, train_loss, val_loss_list[-1], metrics)
#     end = time.time()
#     # best_model = model
#     metrics = evaluate(best_model, test_loader, args)
#     display(metrics)
#     print("Running time: %s seconds" % (end - start))
#     report = {'train_loss_list': train_loss_list, 'val_loss_list': val_loss_list,
#               'best_val_loss': best_val_loss, 'mae': metrics}
#
#     if save:
#         torch.save(best_model.state_dict(), 'weights/init_weight')
#     return best_model, report


def dqas_Scheme(design, dataloader, epochs=None, verbs=None, save=None):
    args = Arguments()

    train_loader, val_loader, test_loader = dataloader
    model = QNet(args, design).to(args.device)
    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.QuantumLayer.parameters(), lr=args.qlr)
    train_loss_list, val_acc_list = [], []
    best_test_acc = 0
    model_grads = None
    print(
        f'\t{"epoch":>5s}\t{"train_loss":>12s}\t{"train_acc":>12s}\t{"val_acc":>12s}\t{"test_acc":>12s}\t{"best_test_acc":>15s}')
    for epoch in range(epochs):
        try:
            train(model, train_loader, optimizer, criterion, args)
        except Exception as e:
            traceback.print_exc()

        train_loss = test(model, train_loader, criterion, args)
        train_loss_list.append(train_loss)
        val_loss = test(model, val_loader, criterion, args)
        val_acc = evaluate(model, val_loader, args)

        val_acc_list.append(val_acc)
        test_acc = evaluate(model, test_loader, args)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model = copy.deepcopy(model)
            print(f'\t{epoch:5d}\t'
                  + f'{train_loss[0]:12.6f}\t'
                  + f'{train_loss[1]:12.6f}\t'
                  + f'{val_acc_list[-1]:12.6f}\t'
                  + display.YELLOW + f'{test_acc:12.6f}\t'
                  + f'{best_test_acc:15.6f}\t'
                  + display.RESET)
        else:
            print(f'\t{epoch:5d}\t'
                  + f'{train_loss[0]:12.6f}\t'
                  + f'{train_loss[1]:12.6f}\t'
                  + f'{val_acc_list[-1]:12.6f}\t'
                  + f'{test_acc:12.6f}\t'
                  + f'{best_test_acc:15.6f}\t')

        if model_grads is None:
            model_grads = []
            for param in model.parameters():
                if param.grad is not None:
                    grads = param.grad.tolist()
                    if len(grads[0]) == 1:
                        grads[0] = grads[0] * 3
                    model_grads.append(grads)

    if save:
        torch.save(best_model.state_dict(), 'weights/init_weight')
    return val_loss[0], model_grads, best_test_acc


if __name__ == '__main__':
    single = None
    enta = None

    # single = [[i] + [0]*8 for i in range(1,5)]
    enta = [[i] + [i] * 4 for i in range(1, 5)]

    # enta = [[4, 1, 1, 3, 1]]  #83.738
    # enta = [[3, 3, 3, 2, 2]]  #83.3
    # enta = [[3, 3, 3, 1, 2]]
    # import pickle
    # with open('search_space_mnist_single', 'rb') as file:
    #     search_space = pickle.load(file)

    # change_code = random.sample(search_space, 10)

    # for i in range(10):
    #     print(change_code[i])
    #     design = translator([change_code[i]])
    #     best_model, report = Scheme(design, 'base', 10)

    # design = translator(single, enta, 'full')
    # best_model, report = Scheme(design, 'init', 30)

    # arch_code = [4, 4]  # qubits, layer
    #
    # design = translator(single, enta, 'full', arch_code)
    # best_model, report = Scheme(design, 'MNIST', 'init' , 30)

    # torch.save(best_model.state_dict(), 'weights/base_fashion')
