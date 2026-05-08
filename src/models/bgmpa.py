# coding: utf-8
"""Clean full implementation of BGMPA."""

import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_knn_normalized_graph, build_sim


class BGMPA(GeneralRecommender):
    """Behavior-guided Modality Graph Propagation with Preference Aggregation."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.sparse = True
        self.embedding_dim = config['embedding_size']
        self.n_ui_layers = config['n_ui_layers']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.cl_loss = config['cl_loss']
        self.dual_graph_cl_weight = float(config['dual_graph_cl_weight'])
        self.image_knn_k = config['image_knn_k']
        self.text_knn_k = config['text_knn_k']
        self.behavior_graph_alpha = float(config['behavior_graph_alpha'])
        self.graph_selection_temperature = 2.0
        self.graph_confidence_bias_scale = 0.4
        lower_bound = float(config['preference_gate_lower_bound'] if 'preference_gate_lower_bound' in config else 0.5)
        lower_bound_logit = self._inverse_sigmoid(lower_bound)

        self.dropout = nn.Dropout(p=config['dropout_rate'])
        self.rng_align_dropout = nn.Dropout(p=config['dropout_rate'])
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)
        self.behavior_item_similarity = self._build_behavior_item_similarity()

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.image_embedding, self.image_plain_adj, self.image_original_adj = self._init_modality_graphs(
            self.v_feat, dataset_path, 'image', self.image_knn_k
        )
        self.text_embedding, self.text_plain_adj, self.text_original_adj = self._init_modality_graphs(
            self.t_feat, dataset_path, 'text', self.text_knn_k
        )
        self.use_image_modality = self.v_feat is not None
        self.use_text_modality = self.t_feat is not None
        if self._active_modality_count() == 0:
            raise ValueError('At least one available modality is required.')

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        self.gate_v = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.gate_t = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.gate_image_prefer = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.gate_text_prefer = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.image_gate_lower_bound_logit = nn.Parameter(torch.tensor(lower_bound_logit, dtype=torch.float32))
        self.text_gate_lower_bound_logit = nn.Parameter(torch.tensor(lower_bound_logit, dtype=torch.float32))
        self.graph_confidence_gate_image = self._mlp(self.embedding_dim + 1, self.embedding_dim)
        self.graph_confidence_gate_text = self._mlp(self.embedding_dim + 1, self.embedding_dim)
        self.modality_attn = nn.MultiheadAttention(self.embedding_dim, num_heads=4, batch_first=True)
        self.graph_selection_gate_image = self._mlp(self.embedding_dim * 3, 1)
        self.graph_selection_gate_text = self._mlp(self.embedding_dim * 3, 1)
        self.dual_graph_projector_image = self._mlp(self.embedding_dim, self.embedding_dim)
        self.dual_graph_projector_text = self._mlp(self.embedding_dim, self.embedding_dim)
        self._eval_state_cache = None

    def _mlp(self, in_dim, out_dim):
        return nn.Sequential(nn.Linear(in_dim, self.embedding_dim), nn.ReLU(), nn.Linear(self.embedding_dim, out_dim))

    def pre_epoch_processing(self):
        self._eval_state_cache = None

    def _init_modality_graphs(self, features, dataset_path, name, topk):
        if features is None:
            return None, None, None
        embedding = nn.Embedding.from_pretrained(features, freeze=False)
        alpha_tag = f"{self.behavior_graph_alpha:.2f}".replace(".", "p")
        graph_tag = f"bgmix_cosine_linear_a{alpha_tag}"
        cache_tag = "mask_img0p00_txt0p00"
        plain_path = os.path.join(dataset_path, f'{name}_adj_{topk}_{self.sparse}_{cache_tag}_plain_graph.pt')
        behavior_path = os.path.join(dataset_path, f'{name}_adj_{topk}_{self.sparse}_{cache_tag}_{graph_tag}.pt')
        plain_adj = self._rebuild_and_cache_graph(
            plain_path,
            lambda: build_knn_normalized_graph(build_sim(embedding.weight.detach()), topk, self.sparse, 'sym'),
        )
        behavior_adj = self._rebuild_and_cache_graph(
            behavior_path,
            lambda: build_knn_normalized_graph(
                self._blend_with_behavior(build_sim(embedding.weight.detach())), topk, self.sparse, 'sym'
            ),
        )
        return embedding, plain_adj.to(self.device), behavior_adj.to(self.device)

    def _rebuild_and_cache_graph(self, graph_path, build_fn):
        graph = build_fn()
        torch.save(graph, graph_path)
        return graph.coalesce() if isinstance(graph, torch.Tensor) and graph.is_sparse else graph

    def get_adj_mat(self):
        R = self.interaction_matrix.tocsr()
        adj_mat = sp.bmat(
            [
                [sp.csr_matrix((self.n_users, self.n_users), dtype=np.float32), R],
                [R.transpose().tocsr(), sp.csr_matrix((self.n_items, self.n_items), dtype=np.float32)],
            ],
            format='csr',
            dtype=np.float32,
        )
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum + 1e-8, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        norm_adj = sp.diags(d_inv).dot(adj_mat).dot(sp.diags(d_inv)).tocoo().tocsr()
        self.R = norm_adj[:self.n_users, self.n_users:].tocsr()
        return norm_adj

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        return torch.sparse_coo_tensor(indices, torch.from_numpy(sparse_mx.data), torch.Size(sparse_mx.shape)).coalesce()

    def _active_modality_count(self):
        return int(self.use_image_modality) + int(self.use_text_modality)

    def _preserve_training_rng_alignment(self, reference_tensor):
        if self.training:
            _ = self.rng_align_dropout(torch.zeros(
                (self.n_users, self.embedding_dim),
                device=reference_tensor.device,
                dtype=reference_tensor.dtype,
            ))

    def _build_behavior_item_similarity(self):
        item_user = self.interaction_matrix.transpose().tocsr().astype(np.float32)
        row_norms = np.sqrt(item_user.multiply(item_user).sum(axis=1)).A1 + 1e-8
        normalized_item_user = sp.diags(1.0 / row_norms).dot(item_user)
        return torch.from_numpy(normalized_item_user.dot(normalized_item_user.transpose()).toarray().astype(np.float32))

    def _inverse_sigmoid(self, value):
        value = float(np.clip(value, 1e-6, 1.0 - 1e-6))
        return np.log(value / (1.0 - value))

    def _preference_gate(self, logits, lower_bound_logit):
        lower = torch.sigmoid(lower_bound_logit).to(logits.device, dtype=logits.dtype)
        return lower + (1.0 - lower) * torch.sigmoid(logits)

    def _blend_with_behavior(self, modality_similarity):
        behavior_similarity = self.behavior_item_similarity.to(modality_similarity.device, dtype=modality_similarity.dtype)
        return self.behavior_graph_alpha * modality_similarity + (1.0 - self.behavior_graph_alpha) * behavior_similarity

    def _encode_content_view(self):
        ego = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
        all_embeddings = [ego]
        for _ in range(self.n_ui_layers):
            ego = torch.sparse.mm(self.norm_adj, ego)
            all_embeddings.append(ego)
        return torch.stack(all_embeddings, dim=1).mean(dim=1)

    def _modality_seed(self, embedding, projector, gate):
        if embedding is None:
            return None
        projected = projector(embedding.weight)
        return self.item_id_embedding.weight * gate(projected)

    def _propagate_item_branch(self, adj, item_embeds):
        if item_embeds is None or adj is None:
            return torch.zeros(
                (self.n_users + self.n_items, self.embedding_dim),
                device=self.item_id_embedding.weight.device,
                dtype=self.item_id_embedding.weight.dtype,
            )
        for _ in range(self.n_layers):
            item_embeds = torch.sparse.mm(adj, item_embeds) if self.sparse else torch.mm(adj, item_embeds)
        return torch.cat([torch.sparse.mm(self.R, item_embeds), item_embeds], dim=0)

    def _build_modality_views(self, image_seed, text_seed, image_adj, text_adj):
        return (
            self._propagate_item_branch(image_adj, image_seed),
            self._propagate_item_branch(text_adj, text_seed),
        )

    def _select_graph_view(self, content, behavior_view, plain_view, gate_layer):
        gate = torch.sigmoid(self.graph_selection_temperature * gate_layer(torch.cat([content, behavior_view, plain_view], dim=-1)))
        return plain_view + gate * (behavior_view - plain_view), gate

    def _aggregate_modalities(self, content, image_view, text_view, image_gate, text_gate):
        if self._active_modality_count() == 1:
            image_side, text_side = image_view, text_view
        else:
            _, weights = self.modality_attn(
                content.unsqueeze(1),
                torch.stack([image_view, text_view], dim=1),
                torch.stack([image_view, text_view], dim=1),
                need_weights=True,
                average_attn_weights=False,
            )
            weights = weights.mean(dim=1).squeeze(1)
            weights = 0.95 * weights + 0.05 * torch.full_like(weights, 0.5)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            image_logits = self.gate_image_prefer(content)
            text_logits = self.gate_text_prefer(content)
            if image_gate is not None:
                image_logits = image_logits + self.graph_confidence_bias_scale * self.graph_confidence_gate_image(
                    torch.cat([content, image_gate], dim=-1)
                )
            if text_gate is not None:
                text_logits = text_logits + self.graph_confidence_bias_scale * self.graph_confidence_gate_text(
                    torch.cat([content, text_gate], dim=-1)
                )
            image_side = self.dropout(
                self._preference_gate(image_logits, self.image_gate_lower_bound_logit) * weights[:, 0:1] * image_view
            )
            text_side = self.dropout(
                self._preference_gate(text_logits, self.text_gate_lower_bound_logit) * weights[:, 1:2] * text_view
            )
        side = (image_side + text_side) / float(self._active_modality_count())
        side = 0.6 * side + 0.4 * self.dropout(side)
        return image_side, text_side, side, content + side

    def _split_user_item(self, embeddings):
        return torch.split(embeddings, [self.n_users, self.n_items], dim=0)

    def _compute_joint_representations(self):
        image_seed = self._modality_seed(self.image_embedding, self.image_trs, self.gate_v) if self.use_image_modality else None
        text_seed = self._modality_seed(self.text_embedding, self.text_trs, self.gate_t) if self.use_text_modality else None
        content = self._encode_content_view()
        image_bg, text_bg = self._build_modality_views(image_seed, text_seed, self.image_original_adj, self.text_original_adj)
        image_plain, text_plain = self._build_modality_views(image_seed, text_seed, self.image_plain_adj, self.text_plain_adj)
        image_view, image_gate = self._select_graph_view(content, image_bg, image_plain, self.graph_selection_gate_image)
        text_view, text_gate = self._select_graph_view(content, text_bg, text_plain, self.graph_selection_gate_text)
        image_side, text_side, side, final = self._aggregate_modalities(content, image_view, text_view, image_gate, text_gate)
        content_users, content_items = self._split_user_item(content)
        side_users, side_items = self._split_user_item(side)
        final_users, final_items = self._split_user_item(final)
        image_users, image_items = self._split_user_item(image_side)
        text_users, text_items = self._split_user_item(text_side)
        image_plain_users, image_plain_items = self._split_user_item(image_plain)
        text_plain_users, text_plain_items = self._split_user_item(text_plain)
        return {
            'content_embeds': content, 'side_embeds': side, 'all_users': final_users, 'all_items': final_items,
            'content_users': content_users, 'content_items': content_items, 'side_users': side_users, 'side_items': side_items,
            'image_users': image_users, 'image_items': image_items, 'text_users': text_users, 'text_items': text_items,
            'image_plain_users': image_plain_users, 'image_plain_items': image_plain_items,
            'text_plain_users': text_plain_users, 'text_plain_items': text_plain_items,
        }

    def forward(self, adj=None):
        reps = self._compute_joint_representations()
        self._preserve_training_rng_alignment(reps['content_embeds'])
        return reps['all_users'], reps['all_items'], reps['side_embeds'], reps['content_embeds']

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))

    def get_l2_regularization(self, users, pos_items, neg_items):
        reg_loss = (
            self.user_embedding(users).norm(2).pow(2) +
            self.item_id_embedding(pos_items).norm(2).pow(2) +
            self.item_id_embedding(neg_items).norm(2).pow(2)
        ) / 2.0
        return self.reg_weight * reg_loss / users.size(0)

    def contrastive_alignment_loss(self, view1, view2, temperature):
        view1 = F.normalize(view1, p=2, dim=-1)
        view2 = F.normalize(view2, p=2, dim=-1)
        logits = torch.matmul(view1, view2.transpose(0, 1)) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))

    def calculate_loss(self, interaction):
        self._eval_state_cache = None
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]
        reps = self._compute_joint_representations()
        self._preserve_training_rng_alignment(reps['content_embeds'])
        loss = self.bpr_loss(reps['all_users'][users], reps['all_items'][pos_items], reps['all_items'][neg_items])
        loss = loss + self.get_l2_regularization(users, pos_items, neg_items)

        unique_users = torch.unique(users)
        unique_items = torch.unique(pos_items)
        cross_view_loss = self.contrastive_alignment_loss(
            reps['side_items'][unique_items], reps['content_items'][unique_items], 0.2
        ) + self.contrastive_alignment_loss(
            reps['side_users'][unique_users], reps['content_users'][unique_users], 0.2
        )
        dual_graph_loss = reps['all_items'].new_tensor(0.0)
        if self.use_image_modality:
            dual_graph_loss = dual_graph_loss + self._dual_graph_loss(
                self.dual_graph_projector_image, reps, unique_users, unique_items, 'image'
            )
        if self.use_text_modality:
            dual_graph_loss = dual_graph_loss + self._dual_graph_loss(
                self.dual_graph_projector_text, reps, unique_users, unique_items, 'text'
            )
        return loss + self.cl_loss * cross_view_loss + self.dual_graph_cl_weight * dual_graph_loss

    def _dual_graph_loss(self, projector, reps, users, items, prefix):
        return self.contrastive_alignment_loss(
            projector(reps[f'{prefix}_plain_items'][items]), projector(reps[f'{prefix}_items'][items]), 0.2
        ) + self.contrastive_alignment_loss(
            projector(reps[f'{prefix}_plain_users'][users]), projector(reps[f'{prefix}_users'][users]), 0.2
        )

    def full_sort_predict(self, interaction):
        user = interaction[0]
        if self._eval_state_cache is None:
            restore_mode = self.training
            self.eval()
            with torch.no_grad():
                all_users, all_items, _, _ = self.forward()
                self._eval_state_cache = (all_users, all_items)
            self.train(restore_mode)
        all_users, all_items = self._eval_state_cache
        return torch.matmul(all_users[user], all_items.transpose(0, 1))
