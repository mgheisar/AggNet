import torch
import torch.nn as nn
import torch.nn.functional as F
import loss_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loss_bc(similarity, N, m_set):
    loss_outputs = 0
    acc = 0
    N_group = N // m_set
    similarity = F.sigmoid(similarity)
    for i in range(N_group):
        y_f = torch.zeros(N).to(device)
        y_f[i * m_set:(i + 1) * m_set] = 1
        # weight = y_f * ((N - m_set) / m_set-1) + 1
        # loss = F.binary_cross_entropy_with_logits(similarity[i], y_f, weight=weight.to(device), reduce=False)
        # loss_outputs += torch.mean(loss)
        weight = torch.FloatTensor([1, (N - m_set)/m_set])
        weight_ = weight[y_f.data.view(-1).long()].view_as(y_f).to(device)
        loss = F.binary_cross_entropy(similarity[i], y_f, weight=weight_, reduce=False)
        loss_outputs += torch.mean(loss)
        y_pred = (similarity[i] > 0.5).float()
        acc += torch.sum(weight_*(y_f == y_pred).float()) / torch.sum(weight_)
    return loss_outputs / N_group, acc / N_group


def loss_bc_fb(similarity, N, m_set):
    loss_outputs = 0
    acc = 0
    N_group = N // m_set
    similarity = F.sigmoid(similarity)
    for i in range(N_group):
        y_f = torch.zeros(N).to(device)
        y_f[i * m_set:(i + 1) * m_set] = 1
        # weight = y_f * ((N - m_set) / m_set-1) + 1
        # loss = F.binary_cross_entropy(similarity[i], y_f, weight=weight.to(device), reduce=False)
        # loss_outputs += torch.mean(loss)
        weight = torch.FloatTensor([1, (N - m_set)/m_set]).to(device)
        weight_ = weight[y_f.data.view(-1).long()].view_as(y_f).to(device)
        loss = F.binary_cross_entropy(similarity[i], y_f, weight=weight_, reduce=False)
        loss_ce = torch.mean(loss)
        TP = weight[1] * torch.sum(similarity[i] * y_f)
        FN = weight[1] * torch.sum((1-similarity[i]) * y_f)
        FP = weight[0] * torch.sum(similarity[i] * (1 - y_f))
        eps = 1e-9
        alpha = 0.5  # 0.5
        Betta = 1
        Betta_sq = Betta ** 2
        loss_fb_score = 1 - ((1 + Betta_sq) * TP) / ((1 + Betta_sq) * TP + Betta_sq * FN + FP + eps)
        # loss_fb_score = - ((1 + Betta_sq) * TP) / ((1 + Betta_sq) * TP + Betta_sq * FN + FP + eps)

        loss_outputs += alpha * loss_ce + (1-alpha) * loss_fb_score

        y_pred = (similarity[i] > 0.5).float()
        acc += torch.sum(weight_*(y_f == y_pred).float()) / torch.sum(weight_)
    return loss_outputs / N_group, acc / N_group


def loss_auc_max_v1(similarity, N, m_set):
    # Reference:: Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic
    loss_outputs = 0
    acc = 0
    N_group = N // m_set
    for i in range(N_group):
        y_f = torch.zeros(N).to(device)
        y_f[i * m_set:(i + 1) * m_set] = 1
        weight = torch.FloatTensor([1, (N - m_set)/m_set]).to(device)
        weight_ = weight[y_f.data.view(-1).long()].view_as(y_f).to(device)  # ????

        y_pred = F.sigmoid(similarity[i])
        eps = 1e-4
        pos = y_pred[y_f > 0]
        neg = y_pred[y_f == 0]
        pos = pos.unsqueeze(0)
        neg = neg.unsqueeze(1)
        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.2  # best(0.2,3)choose a value between 0.1 and 0.7 for γ. p = 2 or 3 (0.6,3)(0.2,2)(0.6,2)(0.3,3)(0.1,3)(0.1,2)(0.3,2) has been tested(not recommended)
        p = 3
        difference = torch.zeros_like(pos * neg) + pos - neg - gamma
        masked = difference[difference < 0.0]
        loss = torch.sum(torch.pow(-masked, p))

        loss_outputs += loss

        y_pred = (F.sigmoid(similarity[i]) > 0.5).float()
        acc += torch.sum(weight_*(y_f == y_pred).float()) / torch.sum(weight_)
    return loss_outputs / N_group, acc / N_group
# def loss_auc_max_v2(similarity, N, m_set):
#     # Reference: : AUC Optimization vs. Error Rate Minimization
#     loss_outputs = 0
#     acc = 0
#     N_group = N // m_set
#     for i in range(N_group):
#         y_f = torch.zeros(N, 1).to(device)
#         y_f[i * m_set:(i + 1) * m_set] = 1
#         weight = torch.FloatTensor([1, (N - m_set)/m_set]).to(device)
#         weight_ = weight[y_f.data.view(-1).long()].view_as(y_f).to(device)  # ????
#         activation = similarity[i].unsqueeze(-1)
#         prediction = F.sigmoid(similarity[i]).unsqueeze(-1)
#         one_vec = torch.ones(y_f.shape).to(device)
#         temp = prediction.repeat(1, N)
#         loss = -torch.sum(F.relu(temp - temp.t()) *
#                              F.relu(y_f @ one_vec.t() - one_vec @ y_f.t()))
#         # loss = -torch.mean(F.sigmoid(activation @ activation.t()) *
#         #                     F.relu(y_f @ one_vec.t() - one_vec @ y_f.t()))
#         # from sklearn.metrics import roc_curve, auc
#         # fpr, tpr, thresholds = roc_curve(y_f.cpu(), prediction.cpu())
#         # a1 = auc(fpr, tpr)
#
#         loss_outputs += loss
#
#         y_pred = (F.sigmoid(similarity[i]) > 0.5).float()
#         acc += torch.sum(weight_.squeeze()*(y_f.squeeze() == y_pred).float()) / torch.sum(weight_.squeeze())
#     return loss_outputs / N_group, acc / N_group
#
#
# def loss_auc_max_v22(similarity, N, m_set):
#     # Reference: : AUC Optimization vs. Error Rate Minimization
#     loss_outputs = 0
#     acc = 0
#     N_group = N // m_set
#     for i in range(N_group):
#         y_f = torch.zeros(N, 1).to(device)
#         y_f[i * m_set:(i + 1) * m_set] = 1
#         weight = torch.FloatTensor([1, (N - m_set)/m_set]).to(device)
#         weight_ = weight[y_f.data.view(-1).long()].view_as(y_f).to(device)  # ????
#         activation = similarity[i].unsqueeze(-1)
#         prediction = F.sigmoid(similarity[i]).unsqueeze(-1)
#         one_vec = torch.ones(y_f.shape).to(device)
#         temp = activation.repeat(1, N)
#         loss = -torch.sum(F.logsigmoid((temp - temp.t()) *
#                                        F.relu(y_f @ one_vec.t() - one_vec @ y_f.t()))) / ((N - m_set)*m_set)
#         # loss = -torch.mean(F.sigmoid(activation @ activation.t()) *
#         #                     F.relu(y_f @ one_vec.t() - one_vec @ y_f.t()))
#         # from sklearn.metrics import roc_curve, auc
#         # fpr, tpr, thresholds = roc_curve(y_f.cpu(), prediction.cpu())
#         # a1 = auc(fpr, tpr)
#
#         loss_outputs += loss
#
#         y_pred = (F.sigmoid(similarity[i]) > 0.5).float()
#         acc += torch.sum(weight_.squeeze()*(y_f.squeeze() == y_pred).float()) / torch.sum(weight_.squeeze())
#     return loss_outputs / N_group, acc / N_group


def FloatTensor(*args):
    if device.type == 'cuda':
        return torch.cuda.FloatTensor(*args)
    else:
        return torch.FloatTensor(*args)


def loss_AUCPRHingeLoss(similarity, N, m_set):
    """
        precision_range_lower (float): the lower range of precision values over
            which to compute AUC. Must be nonnegative, `\leq precision_range_upper`,
            and `leq 1.0`.
        precision_range_upper (float): the upper range of precision values over
            which to compute AUC. Must be nonnegative, `\geq precision_range_lower`,
            and `leq 1.0`.
        num_classes (int): number of classes(aka labels)
        num_anchors (int): The number of grid points used to approximate the
            Riemann sum.

        logits: Variable :math:`(N, C)` where `C = number of classes`
        targets: Variable :math:`(N)` where each value is
            `0 <= targets[i] <= C-1`
        weights: Coefficients for the loss. Must be a `Tensor` of shape
            [N] or [N, C], where `N = batch_size`, `C = number of classes`.
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                sizeAverage is set to False, the losses are instead summed
                for each minibatch. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on size_average. When reduce
            is False, returns a loss per input/target element instead and ignores
            size_average. Default: True
    """
    precision_range_lower: float = 0.0
    precision_range_upper: float = 1.0
    num_classes: int = 1
    num_anchors: int = 20
    precision_range = (precision_range_lower, precision_range_upper,)
    precision_values, delta = loss_utils.range_to_anchors_and_delta(precision_range, num_anchors)
    biases = nn.Parameter(FloatTensor(num_classes, num_anchors).zero_())
    lambdas = nn.Parameter(FloatTensor(num_classes, num_anchors).data.fill_(1.0))

    loss_outputs = 0
    acc = 0
    N_group = N // m_set
    for i in range(N_group):
        logits = similarity[i].unsqueeze(-1)
        y_f = torch.zeros(N).to(device)
        y_f[i * m_set:(i + 1) * m_set] = 1
        targets = y_f.unsqueeze(-1)
        weight = torch.FloatTensor([1, (N - m_set) / m_set]).to(device)
        weights = weight[y_f.data.view(-1).long()].view_as(y_f).to(device)  # ????

        labels, weights = prepare_labels_weights(logits, targets, weights=weights)
        lambdas = loss_utils.lagrange_multiplier(lambdas)
        hinge_loss = loss_utils.weighted_hinge_loss(
            labels.unsqueeze(-1),
            logits.unsqueeze(-1) - biases,
            positive_weights=1.0 + lambdas * (1.0 - precision_values),
            negative_weights=lambdas * precision_values,
        )
        class_priors = loss_utils.build_class_priors(labels, weights=weights)
        lambda_term = class_priors.unsqueeze(-1) * (
            lambdas * (1.0 - precision_values)
        )
        per_anchor_loss = weights.unsqueeze(-1) * hinge_loss - lambda_term

        loss = per_anchor_loss.sum(2) * delta
        loss /= precision_range[1] - precision_range[0]

        loss_outputs += loss.mean()

        y_pred = (F.sigmoid(similarity[i]) > 0.5).float()
        acc += torch.sum(weights.squeeze() * (y_f == y_pred).float()) / torch.sum(weights)
    return loss_outputs / N_group, acc / N_group


def prepare_labels_weights(logits, targets, weights=None):
    """
    Args:
        logits: Variable :math:`(N, C)` where `C = number of classes`
        targets: Variable :math:`(N)` where each value is
            `0 <= targets[i] <= C-1`
        weights: Coefficients for the loss. Must be a `Tensor` of shape
            [N] or [N, C], where `N = batch_size`, `C = number of classes`.
    Returns:
        labels: Tensor of shape [N, C], one-hot representation
        weights: Tensor of shape broadcastable to labels
    """
    N, C = logits.size()
    # labels = FloatTensor(N, C).zero_().scatter(1, targets.unsqueeze(1).data, 1)
    labels = targets
    if weights is None:
        weights = FloatTensor(N).data.fill_(1.0)
    if weights.dim() == 1:
        weights.unsqueeze_(-1)
    return labels, weights
