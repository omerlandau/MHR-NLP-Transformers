# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def get_conf_inc_loss_self_driven(x):
    """
    :param x: features tensor
    :return:

    This function implements stepwise version of conf growth.
    """
    radius = x.detach()
    assert radius.requires_grad == False
    radius = radius + 0.2
    l = ((x - radius) ** 2).mean()
    return l


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True, gamma_conf=None, batch_num=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """


        net_output = model(**sample['net_input'])

        #l_alpha_enc = 0
        #l_alpha_dec =0
        #l_alpha_dec_e = 0

        #if(batch_num>1):

        #    for i in range(len(model.encoder.layers)):
        #        model.encoder.layers[i].self_attn.alphas.requires_grad = False

        #    for i in range(len(model.decoder.layers)):
        #        model.decoder.layers[i].self_attn.alphas.requires_grad = False
        #        model.decoder.layers[i].encoder_attn.alphas.requires_grad = False

        #    for i in range(len(model.encoder.layers)):
        #            l_alpha_enc += 3*torch.norm(model.encoder.layers[i].self_attn.alphas)

        #    for i in range(len(model.decoder.layers)):
        #            l_alpha_dec += 1.5*torch.norm(model.decoder.layers[i].self_attn.alphas)
        #            l_alpha_dec_e += 2*torch.norm(model.decoder.layers[i].encoder_attn.alphas)

        #    print(l_alpha_enc)

        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        if gamma_conf is not None and (batch_num<0.65):

            l_conf_enc = 0
            l_conf_dec = 0
            l_growth_enc = 0
            l_growth_dec = 0
            l_conf_dec_e = 0
            l_growth_dec_e = 0
            for i in range(len(model.encoder.layers)):
                #l_conf_enc += model.encoder.layers[i].self_attn.head_conf.var()
                l_growth_enc += get_conf_inc_loss_self_driven(model.encoder.layers[i].self_attn.head_conf)
            for i in range(len(model.decoder.layers)):
                #l_conf_dec += model.decoder.layers[i].self_attn.head_conf.var()
                l_growth_dec += get_conf_inc_loss_self_driven(model.decoder.layers[i].self_attn.head_conf)
                #l_conf_dec_e += model.decoder.layers[i].encoder_attn.head_conf.var()
                l_growth_dec_e += get_conf_inc_loss_self_driven(model.decoder.layers[i].encoder_attn.head_conf)



            loss += (batch_num+0.3)*l_growth_enc + l_growth_dec*gamma_conf*(batch_num +0.3)\
                    + l_growth_dec_e*gamma_conf*(batch_num +0.3)

        #loss += l_alpha_enc + l_alpha_dec_e + l_alpha_dec

        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
