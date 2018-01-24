# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import lib.transformations.tvm as minimize_tv
import torchvision.transforms as trans
from enum import Enum
import lib.util as util
from lib.transformations.transforms import Crop


class FGSMode(Enum):
    CARLINI = 'carlini'
    LOGIT = 'logit'

    @classmethod
    def has_value(cls, value):
        return (any(value == item.value for item in cls))

    def __str__(self):
        return str(self.value)


# computes robustness, MSSIM, and success rate
# commented out MSSIM since it is quite slow...
def compute_stats(all_inputs, all_outputs, status):
    # computing ssim takes too long...
    # ssim = MSSIM(all_inputs, all_outputs)
    ssim = None
    all_inputs = all_inputs.view(all_inputs.size(0), -1)
    all_outputs = all_outputs.view(all_outputs.size(0), -1)
    diff = (all_inputs - all_outputs).norm(2, 1).squeeze()
    diff = diff.div(all_inputs.norm(2, 1).squeeze())
    n_succ = status.eq(1).sum()
    n_fail = status.eq(-1).sum()
    return (diff.mean(), ssim, float(n_succ) / float(n_succ + n_fail))


# Implementing fast gradient sign
# Goodfellow et al. - Explaining and harnessing adversarial examples
# Use logit mode if computing loss w.r.t. output scores rather than softmax
def fgs(model, input, target, step_size=0.1, train_mode=False, mode=None, verbose=True):
    is_gpu = next(model.parameters()).is_cuda
    if mode:
        assert FGSMode.has_value(mode)
    if train_mode:
        model.train()
    else:
        model.eval()
    model.zero_grad()
    input_var = torch.autograd.Variable(input, requires_grad=True)
    output = model(input_var)
    if is_gpu:
        cpu_targets = target.clone()
        target = target.cuda(async=True)
    else:
        cpu_targets = target
    target_var = torch.autograd.Variable(target)
    _, pred = output.data.cpu().max(1)
    pred = pred.squeeze()
    corr = pred.eq(cpu_targets)
    if mode == str(FGSMode.CARLINI):
        output = output.mul(-1).add(1).log()
        criterion = torch.nn.NLLLoss()
    elif mode == str(FGSMode.LOGIT):
        criterion = torch.nn.NLLLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    if is_gpu:
        criterion = criterion.cuda()
    loss = criterion(output, target_var)
    loss.backward()
    grad_sign = input_var.grad.sign()
    input_var2 = input_var + step_size * grad_sign
    output2 = model(input_var2)
    _, pred2 = output2.data.cpu().max(1)
    pred2 = pred2.squeeze()
    status = torch.zeros(input_var.size(0)).long()
    status[corr] = 2 * pred[corr].ne(pred2[corr]).long() - 1
    return (status, step_size * grad_sign.data.cpu())


# uses line search to find the first step size that reaches desired robustness
# model should have perfect accuracy on input
# assumes epsilon that achieves desired robustness is in range 2^[a,b]
def fgs_search(model, input, target, r, rb=0.9, precision=2,
               a=-10.0, b=0.0, batch_size=25, verbose=True):
    opt_exp = b
    for i in range(precision):
        # search through predefined range on first iteration
        if i == 0:
            lower = a
        else:
            lower = opt_exp - pow(10, 1 - i)
        exponents = torch.arange(lower, opt_exp, pow(10, -i))
        for exponent in exponents:
            step_size = pow(2, exponent)
            succ = torch.zeros(input.size(0)).byte()
            dataset = torch.utils.data.TensorDataset(input + step_size * r, target)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            count = 0
            for x, y in dataloader:
                x_batch = torch.autograd.Variable(x).cuda()
                output = model.forward(x_batch)
                _, pred = output.data.max(1)
                pred = pred.squeeze().cpu()
                succ[count:(count + x.size(0))] = pred.ne(y)
                count = count + x.size(0)
            success_rate = succ.float().mean()
            if verbose:
                print('step size = %1.4f,  success rate = %1.4f'
                      % (step_size, success_rate))
            if success_rate >= rb:
                opt_exp = exponent
                break
    return (succ, pow(2, opt_exp))


def fgs_compute_status(model, inputs, outputs, targets, status,
                       batch_size=25, threshold=0.9, verbose=True):
    all_idx = torch.arange(0, status.size(0)).long()
    corr = all_idx[status.ne(0)]
    r = outputs - inputs
    succ, eps = fgs_search(
        model, inputs[corr], targets[corr], r[corr], rb=threshold,
        batch_size=batch_size, verbose=verbose)
    succ = succ.long()
    status[corr] = 2 * succ - 1
    return (status, eps)


# Implements iterative fast gradient sign
# Kurakin et al. - Adversarial examples in the physical world
def ifgs(model, input, target, max_iter=10, step_size=0.01, train_mode=False,
         mode=None, verbose=True):
    if train_mode:
        model.train()
    else:
        model.eval()
    pred = util.get_labels(model, input)
    corr = pred.eq(target)
    r = torch.zeros(input.size())
    for _ in range(max_iter):
        _, ri = fgs(
            model, input, target, step_size, train_mode, mode, verbose=verbose)
        r = r + ri
        input = input + ri
    pred_xp = util.get_labels(model, input + r)
    status = torch.zeros(input.size(0)).long()
    status[corr] = 2 * pred[corr].ne(pred_xp[corr]).long() - 1
    return (status, r)


# computes DeepFool for a single input image
def deepfool_single(model, imgs, target, n_classes, train_mode, max_iter=10,
                    step_size=0.1, batch_size=25, labels=None, verbose=True):
    is_gpu = next(model.parameters()).is_cuda
    if train_mode:
        model.train()
    else:
        model.eval()
    cpu_targets = target
    imgs_var = torch.autograd.Variable(imgs)
    imgs_var2 = imgs_var.clone()
    r = torch.zeros(imgs_var.size())
    criterion = torch.nn.NLLLoss()
    if is_gpu:
        criterion = criterion.cuda()
    for m in range(max_iter):
        imgs_var_in = imgs_var2.expand(1, imgs_var2.size(0), imgs_var2.size(1),
                                        imgs_var2.size(2))
        grad_input = imgs_var_in.repeat(n_classes, 1, 1, 1)
        output = model(imgs_var_in).clone()
        for j in range(int(n_classes / batch_size)):
            model.zero_grad()
            idx = torch.arange(j * batch_size,
                               (j + 1) * batch_size).long()
            imgs_var_batch = torch.autograd.Variable(
                imgs_var_in.data.repeat(batch_size, 1, 1, 1), requires_grad=True)
            output_batch = model(imgs_var_batch)
            if is_gpu:
                _idx = idx.clone().cuda()
            else:
                _idx = idx.clone()
            loss_batch = criterion(output_batch, torch.autograd.Variable(_idx))
            loss_batch.backward()
            grad_input.index_copy_(0, torch.autograd.Variable(idx),
                                   -imgs_var_batch.grad)
        f = (output - output[0][target].expand_as(output)).cpu()
        w = grad_input - grad_input[target].expand_as(grad_input)
        w_norm = w.view(n_classes, -1).norm(2, 1)
        ratio = torch.abs(f).div(w_norm).data
        ratio[0][target] = float('inf')
        min_ratio, min_idx = ratio.min(1)
        min_w = w[min_idx[0]]
        min_norm = w_norm[min_idx[0]].data
        min_ratio = min_ratio[0]
        min_norm = min_norm[0]
        ri = min_ratio / min_norm * step_size * min_w
        imgs_var2 = imgs_var2.add(ri)
        r = r.add(ri.data)
        imgs_var_in = imgs_var2.clone().expand(1, imgs_var2.size(0),
                                                imgs_var2.size(1), imgs_var2.size(2))
        output2 = model.forward(imgs_var_in).clone()
        _, pred2 = output2.data.cpu().max(1)
        pred2 = pred2.squeeze()[0]
        diff = torch.norm(imgs_var - imgs_var2) / torch.norm(imgs_var)
        diff = diff.data[0]
        if verbose:
            print('iteration ' + str(m + 1) +
                  ': perturbation norm ratio = ' + str(diff))
        if pred2 != cpu_targets:
            if verbose:
                if labels:
                    print('old label = %s, new label = %s' % (labels[cpu_targets],
                                                                labels[pred2]))
                else:
                    print('old label = %d, new label = %d' % (cpu_targets, pred2))
            break
    return (pred2 != target, r)


# Implements DeepFool for a batch of examples
def deepfool(model, input, target, n_classes, train_mode=False, max_iter=5,
             step_size=0.1, batch_size=25, labels=None):
    pred = util.get_labels(model, input, batch_size)
    status = torch.zeros(input.size(0)).long()
    r = torch.zeros(input.size())
    for i in range(input.size(0)):
        status[i], r[i] = deepfool_single(
            model, input[i], target[i], n_classes, train_mode,
            max_iter, step_size, batch_size, labels)
    status = 2 * status - 1
    status[pred.ne(target)] = 0
    return (status, r)


# Implements universal adversarial perturbations
# does not really work...
def universal(model, input, target, n_classes, max_val=0.1, train_mode=False,
              max_iter=10, step_size=0.1, batch_size=25, data_dir=None, r=None,
              verbose=True):
    pred = util.get_labels(model, input, batch_size)
    if r is None:
        r = torch.zeros(input[0].size())
    perm = torch.randperm(input.size(0))
    for i in range(input.size(0)):
        idx = perm[i]
        if verbose:
            print('sample %d: index %d' % (i + 1, idx))
        x_adv = torch.autograd.Variable((input[idx] + r))
        x_adv = x_adv.expand(1, input.size(1), input.size(2), input.size(3))
        output = model.forward(x_adv)
        _, pred_adv = output.max(1)
        pred_adv = pred_adv.data.cpu()[0][0]
        if pred[idx] == pred_adv:
            succ, ri = deepfool_single(
                model, input[idx] + r, pred[idx], n_classes, train_mode, max_iter,
                step_size, batch_size, data_dir)
            if succ:
                r = (r + ri).clamp(-max_val, max_val)
    x = input + r.expand_as(input)
    pred_xp = util.get_labels(model, x)
    status = 2 * pred_xp.ne(target).long() - 1
    status[pred.ne(target)] = 0
    return (status, r)


# Implements Carlini-Wagner's L2 and Linf attacks
# Carlini and Wagner - Towards evaluating the robustness of neural networks
# Modified with TV minimization, random cropping, and random pixel dropping
def cw(model, input, target, weight, loss_str, bound=0, tv_weight=0,
       max_iter=100, step_size=0.01, kappa=0, p=2, crop_frac=1.0, drop_rate=0.0,
       train_mode=False, verbose=True):
    is_gpu = next(model.parameters()).is_cuda
    if train_mode:
        model.train()
    else:
        model.eval()
    pred = util.get_labels(model, input)
    corr = pred.eq(target)
    w = torch.autograd.Variable(input, requires_grad=True)
    best_w = input.clone()
    best_loss = float('inf')
    optimizer = torch.optim.Adam([w], lr=step_size)
    input_var = torch.autograd.Variable(input)
    input_vec = input.view(input.size(0), -1)
    to_pil = trans.ToPILImage()
    scale_up = trans.Resize((w.size(2), w.size(3)))
    scale_down = trans.Resize((int(crop_frac * w.size(2)), int(crop_frac * w.size(3))))
    to_tensor = trans.ToTensor()
    probs = util.get_probs(model, input)
    _, top2 = probs.topk(2, 1)
    argmax = top2[:, 0]
    for j in range(top2.size(0)):
        if argmax[j] == target[j]:
            argmax[j] = top2[j, 1]
    for i in range(max_iter):
        if i > 0:
            w.grad.data.fill_(0)
        model.zero_grad()
        if loss_str == 'l2':
            loss = torch.pow(w - input_var, 2).sum()
        elif loss_str == 'linf':
            loss = torch.clamp((w - input_var).abs() - bound, min=0).sum()
        else:
            raise ValueError('Unsupported loss: %s' % loss_str)
        recons_loss = loss.data[0]
        w_data = w.data
        if crop_frac < 1 and i % 3 == 1:
            w_cropped = torch.zeros(
                w.size(0), w.size(1), int(crop_frac * w.size(2)),
                int(crop_frac * w.size(3)))
            locs = torch.zeros(w.size(0), 4).long()
            w_in = torch.zeros(w.size())
            for m in range(w.size(0)):
                locs[m] = torch.LongTensor(Crop('random', crop_frac)(w_data[m]))
                w_cropped = w_data[m, :, locs[m][0]:(locs[m][0] + locs[m][2]),
                                   locs[m][1]:(locs[m][1] + locs[m][3])]
                minimum = w_cropped.min()
                maximum = w_cropped.max() - minimum
                w_in[m] = to_tensor(scale_up(to_pil((w_cropped - minimum) / maximum)))
                w_in[m] = w_in[m] * maximum + minimum
            w_in = torch.autograd.Variable(w_in, requires_grad=True)
        else:
            w_in = torch.autograd.Variable(w_data, requires_grad=True)
        if drop_rate == 0 and i % 3 == 2:
            output = model.forward(w_in)
        else:
            output = model.forward(torch.nn.Dropout(p=drop_rate).forward(w_in))
        for j in range(output.size(0)):
            loss += weight * torch.clamp(
                output[j][target[j]] - output[j][argmax[j]] + kappa, min=0).cpu()
        adv_loss = loss.data[0] - recons_loss
        if is_gpu:
            loss = loss.cuda()
        loss.backward()
        if crop_frac < 1 and i % 3 == 1:
            grad_full = torch.zeros(w.size())
            grad_cpu = w_in.grad.data
            for m in range(w.size(0)):
                minimum = grad_cpu[m].min()
                maximum = grad_cpu[m].max() - minimum
                grad_m = to_tensor(scale_down(
                    to_pil((grad_cpu[m] - minimum) / maximum)))
                grad_m = grad_m * maximum + minimum
                grad_full[m, :, locs[m][0]:(locs[m][0] + locs[m][2]),
                          locs[m][1]:(locs[m][1] + locs[m][3])] = grad_m
            w.grad.data.add_(grad_full)
        else:
            w.grad.data.add_(w_in.grad.data)
        w_cpu = w.data.numpy()
        input_np = input.numpy()
        tv_loss = 0
        if tv_weight > 0:
            for j in range(output.size(0)):
                for k in range(3):
                    tv_loss += tv_weight * minimize_tv.tv(
                        w_cpu[j, k] - input_np[j, k], p)
                    grad = tv_weight * torch.from_numpy(
                        minimize_tv.tv_dx(w_cpu[j, k] - input_np[j, k], p))
                    w.grad.data[j, k].add_(grad.float())
        optimizer.step()
        total_loss = loss.data.cpu()[0] + tv_loss
        # w.data = utils.img_to_tensor(utils.transform_img(w.data), scale=False)
        output_vec = w.data
        preds = util.get_labels(model, output_vec)
        output_vec = output_vec.view(output_vec.size(0), -1)
        diff = (input_vec - output_vec).norm(2, 1).squeeze()
        diff = diff.div(input_vec.norm(2, 1).squeeze())
        rb = diff.mean()
        sr = float(preds.ne(target).sum()) / target.size(0)
        if verbose:
            print('iteration %d: loss = %f, %s_loss = %f, '
                  'adv_loss = %f, tv_loss = %f' % (
                      i + 1, total_loss, loss_str, recons_loss, adv_loss, tv_loss))
            print('robustness = %f, success rate = %f' % (rb, sr))
        if total_loss < best_loss:
            best_loss = total_loss
            best_w = w.data.clone()
    pred_xp = util.get_labels(model, best_w)
    status = torch.zeros(input.size(0)).long()
    status[corr] = 2 * pred[corr].ne(pred_xp[corr]).long() - 1
    return (status, best_w - input)


# random signs
def rand_sign(model, input, target, step_size, num_bins=100):
    x = torch.autograd.Variable(input, requires_grad=True)
    output = model.forward(x)
    _, pred = output.data.max(1)
    pred = pred.squeeze().cpu()
    corr = pred.eq(target)
    target = torch.autograd.Variable(target)
    P = torch.ones(input.size(0), num_bins)
    sign = 2 * torch.bernoulli(P) - 1
    H = torch.rand(input.size())
    H = (H * num_bins).floor().int()
    r = torch.zeros(input.size())
    for i in range(input.size(0)):
        for j in range(num_bins):
            r[i][H[i].eq(j)] = sign[i, j]
    xp = x + step_size * torch.autograd.Variable(r.cuda())
    output_xp = model.forward(xp).clone()
    _, pred_xp = output_xp.data.max(1)
    pred_xp = pred_xp.squeeze().cpu()
    status = torch.zeros(x.size(0)).long()
    status[corr] = 2 * pred[corr].ne(pred_xp[corr]).long() - 1
    return (status, step_size * r)
