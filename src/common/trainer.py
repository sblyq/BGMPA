# coding: utf-8
"""Minimal trainer for BGMPA."""

import itertools
from logging import getLogger
from time import time

import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_

from utils.topk_evaluator import TopKEvaluator
from utils.utils import dict2str, early_stopping


class Trainer(object):
    def __init__(self, config, model, mg=False):
        self.config = config
        self.model = model
        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.weight_decay = 0.0 if config['weight_decay'] is None else config['weight_decay']
        self.req_training = config['req_training']

        self.cur_step = 0
        self.best_valid_score = -1
        self.best_valid_result = self._empty_metric_dict()
        self.best_test_upon_valid = self._empty_metric_dict()
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_scheduler()
        self.evaluator = TopKEvaluator(config)

    def _empty_metric_dict(self):
        result = {}
        for metric, k in itertools.product(self.config['metrics'], self.config['topk']):
            result[f'{metric.lower()}@{k}'] = 0.0
        return result

    def _build_optimizer(self):
        learner = self.learner.lower()
        if learner == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if learner == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if learner == 'adagrad':
            return optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if learner == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.logger.warning('Unknown optimizer %s, using Adam instead.', self.learner)
        return optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def _build_scheduler(self):
        scheduler_cfg = self.config['learning_rate_scheduler']
        decay, interval = scheduler_cfg[0], scheduler_cfg[1]
        return optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: decay ** (epoch / interval),
        )

    def _train_epoch(self, train_data):
        if not self.req_training:
            return 0.0

        self.model.train()
        total_loss = 0.0
        for interaction in train_data:
            self.optimizer.zero_grad()
            loss = self.model.calculate_loss(interaction)
            if torch.isnan(loss):
                raise ValueError('Training loss became NaN.')
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        for epoch_idx in range(self.epochs):
            train_start = time()
            self.model.pre_epoch_processing()
            train_loss = self._train_epoch(train_data)
            self.lr_scheduler.step()
            train_time = time() - train_start
            post_info = self.model.post_epoch_processing()

            if verbose:
                self.logger.info('epoch %d training [time: %.2fs, train loss: %.4f]', epoch_idx, train_time, train_loss)
                if post_info is not None:
                    self.logger.info(post_info)

            if (epoch_idx + 1) % self.eval_step != 0:
                continue

            valid_start = time()
            valid_score, valid_result = self._valid_epoch(valid_data)
            valid_time = time() - valid_start
            _, test_result = self._valid_epoch(test_data)

            self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                valid_score,
                self.best_valid_score,
                self.cur_step,
                max_step=self.stopping_step,
                bigger=self.valid_metric_bigger,
            )

            if verbose:
                self.logger.info('epoch %d evaluating [time: %.2fs, valid_score: %.6f]', epoch_idx, valid_time, valid_score)
                self.logger.info('valid result: \n%s', dict2str(valid_result))
                self.logger.info('test result: \n%s', dict2str(test_result))

            if update_flag:
                self.best_valid_result = valid_result
                self.best_test_upon_valid = test_result
                if verbose:
                    self.logger.info('██ %s--Best validation results updated!!!', self.config['model'])

            if stop_flag:
                if verbose:
                    self.logger.info('Finished training with early stopping.')
                break

        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid

    def _valid_epoch(self, valid_data):
        result = self.evaluate(valid_data)
        score = result[self.valid_metric] if self.valid_metric else result['ndcg@20']
        return score, result

    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        self.model.eval()
        batch_matrix_list = []
        for batched_data in eval_data:
            scores = self.model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            scores[masked_items[0], masked_items[1]] = -1e10
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)
            batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)
