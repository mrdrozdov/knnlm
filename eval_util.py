import torch
import numpy as np


class EvalUtil:
    @staticmethod
    def get_knn_probmass(tgts, dists, knn_tgts):
        tgts = torch.from_numpy(tgts).long().view(-1)
        dists = torch.from_numpy(dists).float().squeeze(-1)
        probs = torch.log_softmax(dists, dim=-1)
        mass = torch.exp(probs)
        return probs, mass

    @staticmethod
    def get_knn_log_prob(tgts, dists, knn_tgts):

        tgts = torch.from_numpy(tgts).long().view(-1)
        dists = torch.from_numpy(dists).float().squeeze(-1)
        #dists = -dists
        probs = torch.log_softmax(dists, dim=-1)

        index_mask = torch.eq(torch.from_numpy(knn_tgts).long().squeeze(-1), tgts.unsqueeze(-1)).float()
        index_mask[index_mask == 0] = -10000 # for stability
        index_mask[index_mask == 1] = 0

        # (T_reducedxB)
        yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1).clone().numpy()

        # Bx1
        return yhat_knn_prob.reshape(-1, 1)

    @staticmethod
    def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff):
        combine_probs = torch.stack([vocab_p, knn_p], dim=0)
        coeffs = torch.ones_like(combine_probs)
        coeffs[0] = np.log(1 - coeff)
        coeffs[1] = np.log(coeff)
        curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

        return curr_prob

    @staticmethod
    def combine_many_probs(p_list, vocab_p, coeff_list):
        combine_probs = torch.stack([vocab_p] + p_list, dim=0)
        coeffs = torch.ones_like(combine_probs)
        coeffs[0] = np.log(1 - sum(coeff_list))
        for i, c in enumerate(coeff_list):
            coeffs[i+1] = np.log(c)
        curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

        return curr_prob

    @staticmethod
    def eval_ppl(p):
        avg_nll = -p.mean() / np.log(2)
        ppl = 2**avg_nll
        return ppl
