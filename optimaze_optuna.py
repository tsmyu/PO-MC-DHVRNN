import torch
import torch.nn as nn
import torch.nn.functional as F

import optuna
optuna.logging.disable_default_handler()
import torch.optim as optim

import vrnn.models.vrnn_bats as Net


def get_optimizer(trial, model):
  optimizer_names = ['Adam', 'MomentumSGD', 'rmsprop']
  optimizer_name = trial.suggest_categorical('optimizer', optimizer_names)
  weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
  if optimizer_name == optimizer_names[0]: 
    adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
    optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
  elif optimizer_name == optimizer_names[1]:
    momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
    optimizer = optim.SGD(model.parameters(), lr=momentum_sgd_lr, momentum=0.9, weight_decay=weight_decay)
  else:
    optimizer = optim.RMSprop(model.parameters())
  return optimizer

def get_activation(trial):
    activation_names = ['ReLU', 'ELU']
    activation_name = trial.suggest_categorical('activation', activation_names)
    if activation_name == activation_names[0]:
        activation = F.relu
    else:
        activation = F.elu
    return activation

EPOCH = 10
def objective(trial):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #FC層のユニット数
    mid_units_for_enc_prior = int(trial.suggest_discrete_uniform("mid_units_for_enc_prior", 30, 200, 10))
    mid_units_for_dec = int(trial.suggest_discrete_uniform("mid_units_for_dec", 30, 200, 10))

    #GRUのユニット数
    gru_units = int(trial.suggest_discrete_uniform("gru_units", 30, 200, 10))

    #GRUの隠れ状態数
    GRU_micro = int(trial.suggest_discrete_uniform("gru_micro", 30, 200, 10))

    model = Net(trial, mid_units, num_filters).to(device)
    optimizer = get_optimizer(trial, model)

    dropout_num = trial.suggest_int("dropout_num", 1, 10) // 10

    for step in range(EPOCH):
        train(model, device, train_loader, optimizer)
        error_rate = test(model, device, test_loader)

    return error_rate