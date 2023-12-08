# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import logging

import einops
import numpy as np
from tslearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import random

from misc.dist_utils import gather_from_all
from model.crn.trn import BertMLMHead
from model.head import MlpHead
from cluster.Group import Cluster_GPU
from PIL import ImageFilter

class SimclrLoss(nn.Module):
    def __init__(self, cfg, is_bassl):
        nn.Module.__init__(self)

        self.cfg = cfg
        self.num_pos = 2  # fixed

        if is_bassl:
            ssm_name = cfg.LOSS.shot_scene_matching.name
            nce_cfg = cfg.LOSS.shot_scene_matching.params[ssm_name]
        else:
            nce_cfg = cfg.LOSS.simclr
        self.T = nce_cfg["temperature"]

        # create head for nce loss
        self.head_nce = MlpHead(**nce_cfg["head"])

        # parameters for mask generation
        self.total_instances = (
            self.cfg.TRAIN.BATCH_SIZE.effective_batch_size * self.num_pos
        )
        self.world_size = self.cfg.DISTRIBUTED.WORLD_SIZE
        self.batch_size = self.total_instances // self.world_size
        self.orig_instances = self.batch_size // self.num_pos

    def on_train_start(self, dist_rank, device):
        self.dist_rank = dist_rank
        self.device = device
        logging.info(f"Creating Info-NCE loss on Rank: {self.dist_rank}")
        self.precompute_pos_neg_mask()

    def precompute_pos_neg_mask(self):
        """ we precompute the positive and negative masks to speed up the loss calculation
        """
        # computed once at the begining of training
        pos_mask = torch.zeros(
            self.batch_size, self.total_instances, device=self.device
        )
        neg_mask = torch.zeros(
            self.batch_size, self.total_instances, device=self.device
        )
        all_indices = np.arange(self.total_instances)
        pos_members = self.orig_instances * np.arange(self.num_pos)
        orig_members = torch.arange(self.orig_instances)
        for anchor in np.arange(self.num_pos):
            for img_idx in range(self.orig_instances):
                delete_inds = self.batch_size * self.dist_rank + img_idx + pos_members
                neg_inds = torch.tensor(np.delete(all_indices, delete_inds)).long()
                neg_mask[anchor * self.orig_instances + img_idx, neg_inds] = 1
            for pos in np.delete(np.arange(self.num_pos), anchor):
                pos_inds = (
                    self.batch_size * self.dist_rank
                    + pos * self.orig_instances
                    + orig_members
                )
                pos_mask[
                    torch.arange(
                        anchor * self.orig_instances, (anchor + 1) * self.orig_instances
                    ).long(),
                    pos_inds.long(),
                ] = 1
        self.pos_mask = pos_mask
        self.neg_mask = neg_mask

    def _compute_ssm_loss(self, s_emb, d_emb, dtw_path):
        b, n_sparse, _ = s_emb.shape
        # compute scene-level embeddings (average of dense shot features)
        scene_emb = []
        for bi in range(b):
            for si in range(n_sparse):
                aligned_dense_mask = dtw_path[bi][:, 0] == si
                aligned_dense_idx = dtw_path[bi][:, 1][aligned_dense_mask]
                cur_scene_emb = d_emb[bi, aligned_dense_idx].mean(dim=0)
                scene_emb.append(cur_scene_emb)
        scene_emb = torch.stack(scene_emb, dim=0)  # [b*n_sparse,d]
        scene_emb = F.normalize(scene_emb, dim=-1)
        scene_emb = einops.rearrange(scene_emb, "(b nscene) d -> b nscene d", b=b)

        # compute contrastive loss for individual aligned pairs
        ssm_loss = 0
        for si in range(n_sparse):
            sparse_shot = s_emb[:, si]
            scene_shot = scene_emb[:, si]
            paired_emb = torch.cat([sparse_shot, scene_shot], dim=0)  # [b*2 d]
            ssm_loss += self._compute_nce_loss(paired_emb)

        ssm_loss = ssm_loss / n_sparse
        return ssm_loss

    def _compute_nce_loss(self, embedding):
        # Step 1: gather all the embeddings. Shape example: 4096 x 128
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            embeddings_buffer = gather_from_all(embedding)
        else:
            embeddings_buffer = embedding

        # Step 2: matrix multiply: 64 x 128 with 4096 x 128 = 64 x 4096
        # and divide by temperature.
        similarity = torch.exp(torch.mm(embedding, embeddings_buffer.t()) / self.T)
        pos = torch.sum(similarity * self.pos_mask, 1)
        neg = torch.sum(similarity * self.neg_mask, 1)
        loss = -(torch.mean(torch.log(pos / (pos + neg))))
        return loss

    def forward(self, shot_repr, **kwargs):
        # shot_repr shape: [b nview d] -> [(nview b) d]
        shot_repr = torch.cat(torch.unbind(shot_repr, dim=1), dim=0)
        shot_repr = self.head_nce(shot_repr)  # [(nview b) d_head]
        return {"simclr_loss": self._compute_nce_loss(shot_repr)}


class PretextTaskWrapper(SimclrLoss):
    def __init__(self, cfg):
        SimclrLoss.__init__(self, cfg=cfg, is_bassl=True)

        self.use_crn = cfg.MODEL.contextual_relation_network.enabled
        self.use_msm_loss = cfg.LOSS.masked_shot_modeling.get("enabled", False)
        self.use_pp_loss = cfg.LOSS.pseudo_boundary_prediction.get("enabled", False)
        self.use_cgm_loss = cfg.LOSS.contextual_group_matching.get("enabled", False)
        self.use_sc_loss = cfg.LOSS.scene_consistency.get("enabled", False)

        # we are replacing CGM with SCRL's Scene Consistency
        assert(self.use_cgm_loss == False)
        assert(self.use_sc_loss and self.use_msm_loss and self.use_pp_loss)

        if self.use_crn:
            # if we use CRN, one of following losses should be used (set to True)
            assert self.use_msm_loss or self.use_pp_loss or self.use_cgm_loss or self.use_sc_loss
            crn_name = cfg.MODEL.contextual_relation_network.name
        else:
            # if we do not use TM, all following losses should not be used (set to False)
            assert (
                (not self.use_msm_loss)
                and (not self.use_pp_loss)
                and (not self.use_cgm_loss)
            )

        # masked shot modeling loss
        if self.use_msm_loss:
            msm_params = cfg.MODEL.contextual_relation_network.params[crn_name]
            msm_params["vocab_size"] = msm_params.input_dim
            self.head_msm = BertMLMHead(msm_params)

        # boundary prediction loss
        if self.use_pp_loss:
            crn_odim = cfg.MODEL.contextual_relation_network.params[crn_name][
                "hidden_size"
            ]
            self.head_pp = nn.Linear(crn_odim, 2)

            # loss params
            self.num_neg_sample = cfg.LOSS.pseudo_boundary_prediction.num_neg_sample

        # group alignment loss
        if self.use_cgm_loss:
            crn_odim = cfg.MODEL.contextual_relation_network.params[crn_name][
                "hidden_size"
            ]
            self.head_cgm = nn.Linear(crn_odim * 2, 2)

        # scene consistency loss
        if self.use_sc_loss:
            # crn_odim = cfg.MODEL.contextual_relation_network.params[crn_name][
            #     "hidden_size"
            # ]
            # self.head_sc = nn.Linear(crn_odim * 2, 2)
            self.soft_gamma = cfg.LOSS.scene_consistency.soft_gamma
            self.multi_positive = cfg.LOSS.scene_consistency.multi_positive
            self.cluster_num = cfg.LOSS.scene_consistency.cluster_num
            self.cluster_obj = Cluster_GPU(self.cluster_num)
            self.dim = cfg.LOSS.scene_consistency.dim
            self.K = cfg.LOSS.scene_consistency.K
            self.head_sc = nn.Linear(self.dim * 2, 2)

            # create the queue
            self.register_buffer("queue", torch.randn(self.dim, self.K))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            # we must augment the img ourselves when computing k embeddings - Tommy
            # while the data is already augmented, workaround is to augment again
            # https://github.com/TencentYoutuResearch/SceneSegmentation-SCRL/blob/main/data/movienet_data.py#L97
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            augmentation_color_k = nn.Sequential(#[
                # transforms.ToPILImage(),
                # transforms.RandomResizedCrop(224, scale=(0.2, 1.)), # done by Bassl, but we dont want to resize and crop a batched input
                # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5), # done by Bassl with diff params, cannot work with already batched inputs
                # transforms.RandomGrayscale(p=0.2), # cannot work with already batched inputs
                # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5), # done by Bassl, same params
                transforms.GaussianBlur(5, [.1, 2.]), # done by Bassl, same radius params, kernel size is arbitrary
                transforms.RandomHorizontalFlip(), # done by BaSSL
                # transforms.ToTensor(),
                # normalize # done by Bassl with diff params, we skip re-normalizing since our batched input is no longer 3-dimensional image
            )#]
            # self.transform_k = transforms.Compose(augmentation_color_k)
            self.transform_k = augmentation_color_k


    @torch.no_grad()
    def get_q_and_k_index_cluster(self, embeddings, return_group=False) -> tuple:

        B = embeddings.size(0)
        target_index = list(range(0, B))
        q_index = target_index
        # print(q_index)
        choice_cluster, choice_points = self.cluster_obj(embeddings)
        k_index = []
        # print(choice_cluster)
        # print(choice_points)
        for c in choice_cluster: # runs B times
            # print(choice_points[c])
            mean_cluster = np.mean(c)
            mean_points = np.mean(choice_points[int(mean_cluster)])
            # k_index.append(int(choice_points[c]))
            k_index.append(int(mean_points))
        # print(k_index)
        if return_group:
            return (q_index, k_index, choice_cluster, choice_points)
        else:
            return (q_index, k_index)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle
    

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # print("ptr ", self.queue_ptr)
        # print("keys ", keys.shape)
        # not entirely sure how this is supposed to work... but leaving out for now - Tommy
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        
        # not entirely sure how this is supposed to work... but leaving out for now - Tommy
        # assert self.K % batch_size == 0  # for simplicity

        # not entirely sure how this is supposed to work... but hacking for now - Tommy
        # replace the keys at ptr (dequeue and enqueue)
        # self.queue[:, ptr:ptr + batch_size] = keys.T
        # print("queue ", self.queue.shape)
        # print("keys ", keys.shape)
        self.queue[:, :self.K] = keys.T

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    # def _compute_sc_loss(self, crn, dense_shot_repr):
    def _compute_sc_loss(self, dense_shot_repr, crn=None, enc=None):
        """taken from SCRL Github file SCRL_MoCo.py"""
        # compute query features
        # we use CRN instead of separate encoder for query - Tommy
        # embeddings = self.encoder_q(img_q, self.mlp)
        # print("dense shot repr ", dense_shot_repr.shape)
        if crn:
            embeddings, _ = crn(dense_shot_repr)  # infer CRN without masking
            embeddings = embeddings[:, 1:].contiguous()  # exclude [CLS] token
        if enc:
            embeddings = enc(dense_shot_repr)
        # print(embeddings.shape)

        B, nshot, _ = embeddings.shape
        # print("nshot ", nshot)
        embeddings = F.normalize(embeddings, dim=1) # may or may not be necessary - Tommy

        # get q and k index
        index_q, index_k = self.get_q_and_k_index_cluster(embeddings)
        
        # features of q
        q = embeddings[index_q]

        # compute key features
        with torch.no_grad():  
            # we don't need to do this for BaSSL as we are not using contrastive loss in this pretext - Tommy
            # update the key encoder
            # self._momentum_update_key_encoder()

            # we augment the query image again, to obtain k embeddings
            # augmented_shot = torch.stack(self.transform_k(dense_shot_repr), dim=0)
            augmented_shot = self.transform_k(dense_shot_repr).cuda()
            # print("augmented shot ", augmented_shot.shape)

            # shuffle for making use of BN
            augmented_shot, idx_unshuffle = self._batch_shuffle_ddp(augmented_shot)

            # k = self.encoder_k(img_k, self.mlp)   # img_k is an augmented version of img_q - Tommy
            # we encode the augmented img
            if crn:
                k, _ = crn(augmented_shot)
                k = k[:, 1:].contiguous()  # exclude [CLS] token
            if enc:
                k = enc(augmented_shot)

            k = F.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        k_ori = k # self-augmented selection (img_k is an augmented version of img_q) - Tommy
        k = k[index_k] # scene consistency selection, using index_k to find cluster mean - Tommy

        if self.multi_positive:
            # SCRL Soft-SC positive sample selection - Tommy
            k = (k + k_ori) * self.soft_gamma

        # negative shot selection from SCRL
        neg_shot_repr = self.queue.clone().detach()
        # print("neg shot repr ", neg_shot_repr.shape)
        # print("q ", q.shape)
        # print("k ", k.shape)
        
        # pick a random shot to calculate loss over - Tommy
        group_idx = np.arange(0, nshot)
        sampled_idx = np.random.choice(group_idx, size=1)[0]

        q = q[:, sampled_idx].contiguous()
        k = k[:, sampled_idx].contiguous()
        # print("neg shot repr ", neg_shot_repr.shape)
        # print("q ", q.shape)
        # print("k ", k.shape)


        # From this point on, we refer back to BaSSL's CGM logit calculation for CGM cross-entropy loss - Tommy
        logits = self.head_sc(
            torch.cat(
                [
                    torch.cat([q, k], dim=1),
                    torch.cat([q, neg_shot_repr.T], dim=1),
                ],
                dim=0,
            )
        )  # [2*B 2]
        assert(logits.shape[0] == 2 * B)
        labels = torch.cat(
            [
                torch.ones(B, dtype=torch.long, device=embeddings.device),
                torch.zeros(B, dtype=torch.long, device=embeddings.device),
            ],
            dim=0,
        )  # [2*B]
        assert(labels.shape[0] == 2 * B)

        ### SCRL Logit and Label computation ###
        # # compute logits
        # # positive logits: Nx1
        # l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # # negative logits: NxK
        # l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # # logits: Nx(1+K)
        # logits = torch.cat([l_pos, l_neg], dim=1)

        # # apply temperature
        # logits /= self.T

        # # labels: positive key indicators
        # labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # # dequeue and enqueue
        # self._dequeue_and_enqueue(k)

        # return logits, labels

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        # we use same cross-entropy loss fxn as CGM for control, only changing the shot selection method - Tommy
        scm_loss = F.cross_entropy(logits, labels) 

        return scm_loss

    @torch.no_grad()
    def _compute_dtw_path(self, s_emb, d_emb):
        """ compute alignment between two sequences using DTW """
        cost = (
            (1 - torch.bmm(s_emb, d_emb.transpose(1, 2)))
            .cpu()
            .numpy()
            .astype(np.float32)
        )  # shape: [b n_sparse n_dense]
        dtw_path = []
        for bi in range(cost.shape[0]):
            _path, _ = metrics.dtw_path_from_metric(cost[bi], metric="precomputed")
            dtw_path.append(np.asarray(_path))  # [n_dense 2]

        return dtw_path

    def _compute_boundary(self, dtw_path, nshot):
        """ get indices of boundary shots
        return:
            bd_idx: list of size B each of which means index of boundary shot
        """
        # dtw_path: list of B * [ndense 2]
        # find boundary location where the last index of first group (0)
        np_path = np.asarray(dtw_path)
        bd_idx = [np.where(path[:, 0] == 0)[0][-1] for path in np_path]

        return bd_idx

    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don"t compute unnecessary hiddens) """
        mask = mask.unsqueeze(-1).expand_as(hidden).bool()
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def _compute_msm_loss(self, crn, shot_repr, masking_mask):
        """ compute Masked Shot Modeling loss """
        # infer CRN with masking
        crn_repr_w_mask, _ = crn(
            shot_repr, masking_mask
        )  # [B S+1 D]; S means # of shots

        # compute masked shot modeling loss
        crn_repr_wo_cls = crn_repr_w_mask[
            :, 1:
        ].contiguous()  # exclude [CLS] token; [B S D]
        crn_repr_at_masked = self._compute_masked_hidden(
            crn_repr_wo_cls, masking_mask
        )  # [M D]
        logit_at_masked = self.head_msm(crn_repr_at_masked)  # [M D]
        shot_repr_at_masked = self._compute_masked_hidden(
            shot_repr.detach(), masking_mask
        )  # [M D]
        masked_shot_loss = F.mse_loss(
            logit_at_masked, shot_repr_at_masked
        )  # l2 distance

        return masked_shot_loss

    def _compute_pp_loss(self, crn_repr_wo_mask, bd_idx):
        """ compute pseudo-boundary prediction loss """
        # bd_idx: list of B elements
        B, nshot, _ = crn_repr_wo_mask.shape  # nshot == ndense

        # sample non-boundary shots
        nobd_idx = []
        for bi in range(B):
            cand = np.delete(np.arange(nshot), bd_idx[bi])
            nobd_idx.append(
                np.random.choice(cand, size=self.num_neg_sample, replace=False)
            )
        nobd_idx = np.asarray(nobd_idx)

        # get representations of boundary and non-boundary shots
        # shape of shot_repr: [B*(num_neg_sample+1) D]
        # where first B elements correspond to boundary shots
        b_idx = torch.arange(0, B, device=crn_repr_wo_mask.device)
        bd_shot_repr = crn_repr_wo_mask[b_idx, bd_idx]  # [B D]
        nobd_shot_repr = [
            crn_repr_wo_mask[b_idx, nobd_idx[:, ni]]
            for ni in range(self.num_neg_sample)
        ]  # [B num_neg_sample D]
        shot_repr = torch.cat([bd_shot_repr, torch.cat(nobd_shot_repr, dim=0)], dim=0)

        # compute boundaryness loss
        bd_pred = self.head_pp(shot_repr)  # [B*(num_neg_sample+1) D]
        bd_label = torch.ones(
            (bd_pred.shape[0]), dtype=torch.long, device=crn_repr_wo_mask.device
        )
        bd_label[B:] = 0
        pp_loss = F.cross_entropy(bd_pred, bd_label)

        return pp_loss

    def _compute_cgm_loss(self, crn_repr_wo_mask, dtw_path, bd_idx):
        """ contextual group mathcing loss
            where we sample two pairs of (center shot, pos_shot), (center shot, neg_shot)
            and predict whether the pairs belong to the same group or not
        """
        assert (dtw_path is not None) and (bd_idx is not None)
        B, nshot, _ = crn_repr_wo_mask.shape
        center_idx = nshot // 2

        # sample shot indices from group 0 and 1
        matched_idx, no_matched_idx = [], []
        for bi in range(B):
            center_group = int(center_idx > bd_idx[bi].item())
            for si in range(2):
                if si == 0:
                    group_idx = np.arange(0, bd_idx[bi].item() + 1)
                else:
                    group_idx = np.arange(bd_idx[bi].item() + 1, nshot)

                group_cand = np.delete(group_idx, group_idx == center_idx)
                sampled_idx = np.random.choice(group_cand, size=1)[0]
                if int(sampled_idx > bd_idx[bi].item()) == center_group:
                    matched_idx.append(sampled_idx)
                else:
                    no_matched_idx.append(sampled_idx)

        # obtain representations
        b_idx = torch.arange(0, B, device=crn_repr_wo_mask.device)
        center_shot_repr = F.normalize(crn_repr_wo_mask[:, center_idx], dim=1)  # [B D]
        pos_shot_repr = F.normalize(
            crn_repr_wo_mask[b_idx, matched_idx], dim=1
        )  # [B D]
        neg_shot_repr = F.normalize(
            crn_repr_wo_mask[b_idx, no_matched_idx], dim=1
        )  # [B D]

        logit = self.head_cgm(
            torch.cat(
                [
                    torch.cat([center_shot_repr, pos_shot_repr], dim=1),
                    torch.cat([center_shot_repr, neg_shot_repr], dim=1),
                ],
                dim=0,
            )
        )  # [2*B 2]
        label = torch.cat(
            [
                torch.ones(B, dtype=torch.long, device=crn_repr_wo_mask.device),
                torch.zeros(B, dtype=torch.long, device=crn_repr_wo_mask.device),
            ],
            dim=0,
        )  # [2*B]
        cgm_loss = F.cross_entropy(logit, label)

        return cgm_loss

# utils for Scene Consistency loss - Tommy
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
