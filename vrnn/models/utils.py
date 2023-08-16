import torch
import math


######################################################################
############################ MODEL UTILS #############################
######################################################################


def num_trainable_params(model):
    total = 0
    for p in model.parameters():
        count = 1
        for s in p.size():
            count *= s
        total += count
    return total


def parse_model_params(model_args, params, parser):
    if parser is None:
        return params

    for arg in model_args:
        parser.add_argument("--" + arg, type=int, required=True)
    args, _ = parser.parse_known_args()

    for arg in model_args:
        params[arg] = getattr(args, arg)

    return params


def get_params_str(model_args, params):
    ret = ""
    for arg in model_args:
        ret += " {} {} |".format(arg, params[arg])
    return ret[1:-2]


def cudafy_list(states):
    for i in range(len(states)):
        states[i] = states[i].cuda()
    return states


def index_by_agent(states, n_agents):
    x = states[1:, :, : 2 * n_agents].clone()
    x = x.view(x.size(0), x.size(1), n_agents, -1).transpose(1, 2)
    return x


def get_macro_ohe(macro, n_agents, M):
    macro_ohe = torch.zeros(macro.size(0), n_agents, macro.size(1), M)
    for i in range(n_agents):
        macro_ohe[:, i, :, :] = one_hot_encode(macro[:, :, i].data, M)
    if macro.is_cuda:
        macro_ohe = macro_ohe.cuda()

    return macro_ohe


######################################################################
############################## GAUSSIAN ##############################
######################################################################


def sample_gauss(mean, std):
    eps = torch.FloatTensor(std.size()).normal_()
    if mean.is_cuda:
        eps = eps.cuda()
    return eps.mul(std).add_(mean)


def nll_gauss(mean, std, x, pow=False, Sum=True):
    pi = torch.FloatTensor([math.pi])
    if mean.is_cuda:
        pi = pi.cuda()
    if not pow:
        nll_element = (
            (x - mean).pow(2) / std.pow(2) + 2 * torch.log(std) + torch.log(2 * pi)
        )
    else:
        nll_element = (x - mean).pow(2) / std + torch.log(std) + torch.log(2 * pi)

    nll = 0.5 * torch.sum(nll_element) if Sum else 0.5 * torch.sum(nll_element, 1)
    return nll


def kld_gauss(mean_1, std_1, mean_2, std_2, Sum=True):
    kld_element = (
        2 * torch.log(std_2)
        - 2 * torch.log(std_1)
        + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2)
        - 1
    )
    kld = 0.5 * torch.sum(kld_element) if Sum else 0.5 * torch.sum(kld_element, 1)
    return kld


def entropy_gauss(std, scale=1):
    """Computes gaussian differential entropy."""
    pi, e = torch.FloatTensor([math.pi]), torch.FloatTensor([math.e])
    if std.is_cuda:
        pi, e = pi.cuda(), e.cuda()
    return 0.5 * torch.sum(scale * torch.log(2 * pi * e * std))


def batch_error(predict, true, Sum=True, sqrt=True, diff=True):
    # error = torch.sum(torch.sum((predict[:,:2] - true[:,:2]),1))
    if diff:
        error = torch.sum((predict[:, :2] - true[:, :2]).pow(2), 1)
    else:
        error = torch.sum(predict[:, :2].pow(2), 1)
    if sqrt:
        error = torch.sqrt(error)
    if Sum:
        error = torch.sum(error)
    return error


######################################################################
############### sample_gumbel_softmax ################################
######################################################################


def sample_gumbel(shape, eps=1e-20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unif = torch.rand(*shape).to(device)
    g = -torch.log(-torch.log(unif + eps))
    return g


def sample_gumbel_softmax(logits, temperature):
    """
    Input:
    logits: Tensor of log probs, shape = BS x k
    temperature = scalar

    Output: Tensor of values sampled from Gumbel softmax.
            These will tend towards a one-hot representation in the limit of temp -> 0
            shape = BS x k
    """
    g = sample_gumbel(logits.shape)
    h = (g + logits) / temperature
    h_max = h.max(dim=-1, keepdim=True)[0]
    h = h - h_max
    cache = torch.exp(h)
    y = cache / cache.sum(dim=-1, keepdim=True)
    return y


######################################################################
############################## BODY ##################################
######################################################################


def vel_cost(next_vel, vel_bound):
    # ReLu-like velocity cost function
    diff_vel = next_vel - vel_bound
    loc = (diff_vel > 0).nonzero()
    return torch.sum(diff_vel[loc])


def acc_cost(next_vel, current_abs_vel, fs, acc_bound):
    # ReLu-like acceleration cost function
    next_abs_vel = torch.sqrt(torch.sum((next_vel).pow(2), 1))
    diff_vel = next_abs_vel - (current_abs_vel + acc_bound * fs)
    loc = (diff_vel > 0).nonzero()
    return torch.sum(diff_vel[loc])


######################################################################
############################## ROLE OUT ##############################
######################################################################


def calc_dist_cos_sin(rolePos, refPos, batchSize):
    if torch.cuda.is_available():  # rolePos.is_cuda:
        rolePos = rolePos.cuda()
        refPos = refPos.cuda()
        rolefeat = torch.zeros((batchSize, 3)).cuda()
    else:
        rolefeat = torch.zeros((batchSize, 3))

    rolefeat[:, 0] = torch.sqrt(
        (rolePos[:, 0] - refPos[:, 0]).pow(2) + (rolePos[:, 1] - refPos[:, 1]).pow(2)
    )  # dist
    loc0 = (rolefeat[:, 0] == 0).nonzero()
    loc1 = (rolefeat[:, 0] != 0).nonzero()

    rolefeat[loc1, 1] = (rolePos[loc1, 0] - refPos[loc1, 0]) / rolefeat[loc1, 0]  # cos
    rolefeat[loc1, 2] = (rolePos[loc1, 1] - refPos[loc1, 1]) / rolefeat[loc1, 0]  # sin

    rolefeat[loc0, 1] = 0.0
    rolefeat[loc0, 2] = 0.0

    return rolefeat


def roll_out(
    y_t,
    y_t_1,
    prediction_all,
    acc,
    normalize,
    n_roles,
    n_feat,
    ball_dim,
    fs,
    batchSize,
    i,
):  # update feature vector using next_prediction
    """
    input:
        prev_feature & next_feature: see get_sequences_Le in sequencing.py
        next_prediction: xy position or velocity (n_pl*2,0)
        roleOrder: scalar (1,0)
    output:
        new_feature_vector
    """
    prev_feature = y_t
    next_feature = y_t_1
    dim_x = prediction_all.shape[2]
    if acc == 0:  # vel
        next_vel = prediction_all
        dim = dim_x
    elif acc == 1 or acc == -1 or acc == 3:  # pos,vel,(acc)
        error("not modified yet. dim should be defined")  # TBD
        next_pos = prediction_all[:, :, :2]
        if acc >= 1:
            next_vel = prediction_all[:, :, 2:4]
        if acc == 3:
            next_acc = prediction_all[:, :, 4:]
    elif acc == 2:  # vel,acc
        dim = int(dim_x / 2)
        next_vel = prediction_all[:, :, :dim]
        next_acc = prediction_all[:, :, dim:]
    elif acc == 4:  # acc
        next_acc = prediction_all
        dim = dim_x

    roleOrder = i
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # batchSize = prev_feature.shape[0]
    n_feat_all = prev_feature.shape[1]

    n_all_agents = n_roles
    n_feat_player = n_feat * n_all_agents

    next_current = next_feature[:, 0:n_feat_player]

    legacy_next = next_current.reshape(batchSize, n_all_agents, n_feat)
    new_matrix = torch.zeros((batchSize, n_all_agents, n_feat)).to(device)
    teammateList = list(range(n_all_agents))

    roleOrderList = [role for role in range(n_roles)]
    role_long = torch.zeros((batchSize, n_feat)).to(device)
    teammateList.remove(roleOrder)

    # fix role vector
    if acc >= 2:
        role_long[:, dim * 2 :] = next_acc[:, roleOrder, :]
    if acc >= 0 and acc < 4:
        role_long[:, dim : dim * 2] = next_vel[:, roleOrder, :]
    if acc == 1 or acc == 3 or acc == -1:
        error("not modified yet")  # TBD
        role_long[:, :dim] = next_pos[:, roleOrder, :]
    elif acc == 0 or acc == 2:
        role_long[:, 0:dim] = (
            prev_feature[:, roleOrder * n_feat : (roleOrder * n_feat + dim)]
            + prev_feature[:, roleOrder * n_feat + dim : (roleOrder * n_feat + dim * 2)]
            * fs
        )
    elif acc == 4:
        role_long[:, dim : dim * 2] = (
            prev_feature[:, roleOrder * n_feat + 2 : (roleOrder * n_feat + dim * 2)]
            + prev_feature[
                :, roleOrder * n_feat + dim * 2 : (roleOrder * n_feat + dim * 3)
            ]
            * fs
        )
        role_long[:, 0:dim] = (
            prev_feature[:, roleOrder * n_feat : (roleOrder * n_feat + dim)]
            + prev_feature[:, roleOrder * n_feat + dim : (roleOrder * n_feat + dim * 2)]
            * fs
        )

    new_matrix[:, roleOrder, :] = role_long

    # fix all teammates vector
    for teammate in teammateList:
        player = legacy_next[:, teammate, :]
        if (
            teammate in roleOrderList
        ):  # if the teammate is one of the active players: e.g. eliminate goalkeepers
            teamRoleInd = roleOrderList.index(teammate)

            if acc >= 2:  # vel,acc
                player[:, dim * 2 :] = next_acc[:, teamRoleInd, :]

            if acc >= 0 and acc < 4:
                player[:, dim : dim * 2] = next_vel[:, teamRoleInd, :]
            elif acc == 4:
                player[:, dim : dim * 2] = (
                    prev_feature[
                        :, teamRoleInd * n_feat + dim : (teamRoleInd * n_feat + dim * 2)
                    ]
                    + prev_feature[
                        :,
                        teamRoleInd * n_feat
                        + dim : 2 : (teamRoleInd * n_feat + dim * 3),
                    ]
                    * fs
                )

            if acc == 1 or acc == 3 or acc == -1:
                player[:, 0:dim] = next_pos[:, teamRoleInd, :]
            else:
                player[:, 0:dim] = (
                    prev_feature[:, teamRoleInd * n_feat : (teamRoleInd * n_feat + dim)]
                    + prev_feature[
                        :, teamRoleInd * n_feat + dim : (teamRoleInd * n_feat + dim * 2)
                    ]
                    * fs
                )

        new_matrix[:, teammate, :] = player

    # output
    new_feature_vector = torch.reshape(new_matrix, (batchSize, n_all_agents * n_feat))
    # import pdb; pdb.set_trace()
    return new_feature_vector


######################################################################
########################## MISCELLANEOUS #############################
######################################################################


def one_hot_encode(inds, N):
    dims = [inds.size(i) for i in range(len(inds.size()))]
    inds = inds.unsqueeze(-1).cpu().long()
    dims.append(N)
    ret = torch.zeros(dims)
    ret.scatter_(-1, inds, 1)
    return ret


def sample_multinomial(probs):
    inds = torch.multinomial(probs, 1).data.cpu().long().squeeze()
    ret = one_hot_encode(inds, probs.size(-1))
    if probs.is_cuda:
        ret = ret.cuda()
    return ret
