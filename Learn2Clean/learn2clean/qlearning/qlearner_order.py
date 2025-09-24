#!/usr/bin/env python3
# coding: utf-8

import warnings
import time
import numpy as np
import random
from random import randint
import re

from ..normalization.normalizer import Normalizer
from ..deduplication.deduplicator import Deduplicator
from ..punctualion_remove.punctuation_remover import PunctuationRemover
from ..spellcheck.spellchecker import SpellChecker
from ..stopword_remove.stopword_remover import StopwordRemover
from ..tokenize.tokenizer import Tokenizer
from ..unit_convert.unit_converter import UnitConverter
from ..lowercase.lowercaser import Lowercaser
from ..language_translator.translator import LanguageTranslator
from ..outlier_detection.outlier_detector import Outlier_detector
from ..consistency_checking.consistency_checker import Consistency_checker
from ..imputation.imputer import Imputer
from ..feature_selection.feature_selector import Feature_selector
from ..regression.regressor import Regressor
from ..clustering.clusterer import Clusterer
from ..classification.classifier import Classifier


def remove_adjacent(nums):
    previous = object()
    out = []
    for i in nums:
        if i != previous:
            out.append(i)
        previous = i
    return out


class Qlearner():
    """
    Learn2Clean Q-learning over ordered preprocessing blocks, then model (MARS for regression, LR for classification).
    """

    def __init__(self, dataset, goal, target_goal, target_prepare,
                 verbose=False, file_name=None, threshold=None, f_goal=0.8,
                 randomize_blocks=True):
        self.dataset = dataset
        self.goal = goal
        self.target_goal = target_goal
        self.target_prepare = target_prepare
        self.verbose = verbose
        self.file_name = file_name
        self.threshold = threshold
        self.f_goal = f_goal

        self.randomize_blocks = randomize_blocks
        self._rng = None  
        self._blocks = None
        self._goals = None
        self._actions = None
        self._block_id_of = None
        self._goal_block_idx = None
        self._R_full = None
        self._learn_row_mask = None
        self._learn_rows_idx = None
        self._global_to_learnrow = None

    def get_params(self, deep=True):
        return {
            'goal': self.goal,
            'target_goal': self.target_goal,
            'target_prepare': self.target_prepare,
            'verbose': self.verbose,
            'file_name': self.file_name,
            'threshold': self.threshold,
            'f_goal': self.f_goal,
            'randomize_blocks': self.randomize_blocks,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for Qlearner. Parameter(s) IGNORED. "
                              "Check with `qlearner.get_params().keys()`")
            else:
                setattr(self, k, v)

    # -------- Build blocks (unordered) -------- #
    def _build_blocks(self, check_missing):
        blocks = []
        if check_missing:
            blocks.append([
                ("DROP",   Imputer, {}),
                ("MEAN",   Imputer, {}),
                ("MEDIAN", Imputer, {}),
                ("MF",     Imputer, {}),
                ("KNN_1",  Imputer, {}),
                ("KNN_5",  Imputer, {}),
                ("KNN_10", Imputer, {}),
                ("KNN_20", Imputer, {}),
                ("KNN_30", Imputer, {}),
            ])
        blocks.append([
            ("none_dedup", Deduplicator, {}),
            ("first",      Deduplicator, {}),
        ])

        blocks.append([
            ("NONE_lowercase", Lowercaser, {}),
            ("LC",             Lowercaser, {}),
        ])
        blocks.append([
            ("NONE_translate", LanguageTranslator, {}),
            ("GT",             LanguageTranslator, {}),
        ])
        blocks.append([
            ("NONE_punct", PunctuationRemover, {}),
            ("PR",         PunctuationRemover, {}),
        ])
        blocks.append([
            ("NONE_stopword", StopwordRemover, {}),
            ("SW",            StopwordRemover, {}),
        ])
        blocks.append([
            ("NONE_spell", SpellChecker, {}),
            ("SC",         SpellChecker, {}),
        ])
        blocks.append([
            ("NONE_tokenize", Tokenizer, {}),
            ("WS",            Tokenizer, {}),
            ("NLTK",          Tokenizer, {}),
        ])
        blocks.append([
            ("NONE_convert", UnitConverter, {}),
            ("UC",           UnitConverter, {}),
        ])
        blocks.append([
            ("NONE_normalization", Normalizer, {}),
            ("SS",                 Normalizer, {}),
            ("RS",                 Normalizer, {}),
            ("MA",                 Normalizer, {}),
            ("MM",                 Normalizer, {}),
        ])
        blocks.append([
            ("NONE_outlier", Outlier_detector, {}),
            ("IF",           Outlier_detector, {}),
            ("LOF_1",        Outlier_detector, {}),
            ("LOF_5",        Outlier_detector, {}),
            ("LOF_10",       Outlier_detector, {}),
            ("LOF_20",       Outlier_detector, {}),
            ("LOF_30",       Outlier_detector, {}),
        ])

        goals = [("MARS", Regressor, {}), ("LR", Classifier, {})]
        return blocks, goals

    def _flatten_actions(self, blocks, goals):
        actions = []
        block_id_of = []
        for b_idx, block in enumerate(blocks):
            for name, cls, extra in block:
                actions.append((name, cls, extra))
                block_id_of.append(b_idx)
        goal_block_idx = len(blocks)
        for name, cls, extra in goals:
            actions.append((name, cls, extra))
            block_id_of.append(goal_block_idx)
        return actions, block_id_of, goal_block_idx

    def _build_reward_matrix(self, blocks, goals):
        actions, block_id_of, goal_block_idx = self._flatten_actions(blocks, goals)
        n = len(actions)
        R_full = -1.0 * np.ones((n, n), dtype="float32")

        for i in range(n):
            bi = block_id_of[i]
            if bi < goal_block_idx - 1:
                # From block i -> any action in next block
                for j in range(n):
                    if block_id_of[j] == bi + 1:
                        R_full[i, j] = 0.0
            elif bi == goal_block_idx - 1:
                # Last preprocessing block -> goals
                for j in range(n):
                    if block_id_of[j] == goal_block_idx:
                        R_full[i, j] = 100.0

        learn_row_mask = ~np.all(R_full == -1, axis=1)
        R_learn = R_full[learn_row_mask, :]
        Q_learn = np.zeros_like(R_learn, dtype="float32")

        learn_rows_idx = np.where(learn_row_mask)[0]
        global_to_learnrow = {g: i for i, g in enumerate(learn_rows_idx)}

        return (Q_learn, R_learn, R_full, actions, block_id_of, goal_block_idx,
                learn_row_mask, learn_rows_idx, global_to_learnrow)

    def Initialization_Reward_Matrix(self, dataset):
        check_missing = dataset['train'].copy().isnull().sum().sum() > 0
        blocks, goals = self._build_blocks(check_missing)
        if self.randomize_blocks:
            if self._rng is None:
                self._rng = np.random.RandomState()
            idx = np.arange(len(blocks))
            self._rng.shuffle(idx)
            blocks = [blocks[i] for i in idx]

        (Q_learn, R_learn, R_full, actions, block_id_of, goal_block_idx,
         learn_row_mask, learn_rows_idx, global_to_learnrow) = self._build_reward_matrix(blocks, goals)

        # cache for later
        self._blocks = blocks
        self._goals = goals
        self._actions = actions
        self._block_id_of = block_id_of
        self._goal_block_idx = goal_block_idx
        self._R_full = R_full
        self._learn_row_mask = learn_row_mask
        self._learn_rows_idx = learn_rows_idx
        self._global_to_learnrow = global_to_learnrow

        n_actions = R_learn.shape[0]
        n_states = n_actions
        return Q_learn, R_learn, n_actions, n_states, check_missing

    def _instantiate_and_run(self, a_idx, dataset, target_goal, target_prepare):
        name, cls, extra = self._actions[a_idx]

        if self._block_id_of[a_idx] < self._goal_block_idx:
            if cls is Normalizer:
                return cls(dataset=dataset, strategy=name,
                           exclude=target_prepare, verbose=self.verbose, **extra).transform()
            elif cls in (Tokenizer, Lowercaser, PunctuationRemover, StopwordRemover,
                         SpellChecker, LanguageTranslator, Deduplicator, UnitConverter):
                return cls(dataset=dataset, strategy=name,
                           verbose=self.verbose, **extra).transform()
            elif cls is Imputer:
                return cls(dataset=dataset, strategy=name,
                           verbose=self.verbose, **extra).transform()
            elif cls is Outlier_detector:
                return cls(dataset=dataset, strategy=name,
                           verbose=self.verbose, **extra).transform()
            else:
                return cls(dataset=dataset, strategy=name,
                           verbose=self.verbose, **extra).transform()
        else:
            if name == "MARS":
                return Regressor(dataset=dataset, strategy="MARS",
                                 target=target_goal, verbose=self.verbose, **extra).transform()
            elif name == "LR":
                return Classifier(dataset=dataset, strategy="LR",
                                  target=target_goal, verbose=self.verbose, **extra).transform()
            else:
                raise ValueError("Unknown goal: %s" % name)

    def pipeline(self, dataset, actions_list, target_goal, target_prepare, check_missing):
        dataset = dataset.copy()
        print("\nStart pipeline")
        print("-------------")
        start_time = time.time()

        res = None
        n = None

        for a_global in actions_list:
            n = self._instantiate_and_run(a_global, dataset, target_goal, target_prepare)
            if isinstance(n, dict) and 'train' in n:
                dataset = n
            if self._block_id_of[a_global] == self._goal_block_idx:
                res = n

        t = time.time() - start_time
        print("End Pipeline CPU time: %s seconds" % t)
        return n, res, t

    def show_traverse(self, dataset, q_learn, g, target1, target2, check_missing):
        actions_names = [name for (name, _cls, _extra) in self._actions]
        learn_rows_idx = self._learn_rows_idx

        results = []
        for i_learnrow, start_global in enumerate(learn_rows_idx):
            actions_list = []
            traverse_name = "%s -> " % actions_names[start_global]
            current_learn_row = i_learnrow
            n_steps = 0

            while current_learn_row < q_learn.shape[0] and n_steps < 64:
                actions_list.append(learn_rows_idx[current_learn_row])
                next_col = int(np.argmax(q_learn[current_learn_row]))
                traverse_name += "%s -> " % actions_names[next_col]
                actions_list.append(next_col)
                n_steps += 1

                if next_col not in self._global_to_learnrow:
                    break
                current_learn_row = int(self._global_to_learnrow[next_col])

            actions_list = remove_adjacent(actions_list)
            traverse_name = traverse_name[:-4]

            print("\n\nStrategy#", i_learnrow, ": Greedy traversal for start", actions_names[start_global])
            print(traverse_name)

            goal_global_idx = len(self._actions) - len(self._goals) + g
            if actions_list[-1] != goal_global_idx:
                actions_list.append(goal_global_idx)
                traverse_name += " -> %s" % self._goals[g][0]

            pipeline_result = self.pipeline(dataset, actions_list, target1, target2, check_missing)
            metrics = pipeline_result[1]

            if metrics is not None and 'quality_metric' in metrics:
                print("Quality metric ", metrics['quality_metric'])
                results.append(metrics)
                if self._goals[g][0] == "LR" and metrics['quality_metric'] >= self.f_goal:
                    print("Pipeline : ", traverse_name)
                    print(metrics)
                    print("Achieved in ", i_learnrow + 1)
                    return True, i_learnrow + 1
                if self._goals[g][0] == "MARS" and metrics['quality_metric'] <= self.f_goal:
                    print("Pipeline : ", traverse_name)
                    print(metrics)
                    print("Achieved in ", i_learnrow + 1)
                    return True, i_learnrow + 1

        goal_global_idx = len(self._actions) - len(self._goals) + g
        pipeline_result = self.pipeline(dataset, [goal_global_idx], target1, target2, check_missing)
        results.append(pipeline_result[1])
        return False, i_learnrow + 1 if 'i_learnrow' in locals() else 1

    def learn2clean(self, r_state):
        goals_names = ["MARS", "LR"]
        if self.goal not in goals_names:
            raise ValueError("Goal invalid. Choose 'MARS' (regression) or 'LR' (classification).")
        g = goals_names.index(self.goal)

        if self.target_goal != self.dataset['target'].name:
            raise ValueError("Target variable invalid (must equal dataset['target'].name).")

        start_l2c = time.time()
        print("Start Learn2Clean")

        gamma = 0.8
        beta = 1.0
        n_episodes = int(1e3)
        epsilon = 0.05

        # Seeded RNG for both Q-learning and block shuffling
        rng = np.random.RandomState(r_state)
        self._rng = rng  # <-- used in Initialization_Reward_Matrix

        q_learn, r_learn, n_actions, n_states, check_missing = self.Initialization_Reward_Matrix(self.dataset)

        for e in range(n_episodes):
            learn_row_indices = np.arange(n_actions)
            rng.shuffle(learn_row_indices)
            current_row = int(learn_row_indices[0])
            goal_reached = False

            while not goal_reached:
                valid_moves = np.where(r_learn[current_row] >= 0)[0]
                if valid_moves.size == 0:
                    break

                if rng.rand() < epsilon:
                    action_col = int(random.choice(valid_moves.tolist()))
                else:
                    if np.sum(q_learn[current_row]) > 0:
                        action_col = int(np.argmax(q_learn[current_row]))
                    else:
                        action_col = int(random.choice(valid_moves.tolist()))

                reward = float(r_learn[current_row, action_col])

                if action_col in self._global_to_learnrow:
                    next_row = int(self._global_to_learnrow[action_col])
                    max_future = float(np.max(q_learn[next_row])) if q_learn.shape[0] > 0 else 0.0
                else:
                    next_row = None
                    max_future = 0.0

                qsa = q_learn[current_row, action_col]
                target = reward + gamma * max_future
                new_q = qsa + beta * (target - qsa)
                q_learn[current_row, action_col] = new_q

                row = q_learn[current_row]
                pos = row > 0
                if np.any(pos):
                    q_learn[current_row, pos] = row[pos] / np.sum(row[pos])

                if reward > 1:
                    goal_reached = True
                elif next_row is not None:
                    current_row = next_row
                else:
                    break

        if self.verbose:
            print("Q-value matrix (learning rows)\n", q_learn)

        print("Learn2Clean - Pipeline construction -- CPU time: %s seconds"
              % (time.time() - start_l2c))

        print("=== Start Pipeline Execution ===")
        start_exec = time.time()
        result_list = self.show_traverse(self.dataset, q_learn, g,
                                         self.target_goal, self.target_prepare,
                                         check_missing)
        t = time.time() - start_exec
        print("=== End of Learn2Clean - Pipeline execution -- CPU time: %s seconds" % t)
        print()
        return result_list

    def random_cleaning(self, dataset_name):
        random.seed(time.time())
        check_missing = self.dataset['train'].isnull().sum().sum() > 0
        blocks, goals = self._build_blocks(check_missing)

        # randomize block order here too for consistency (optional)
        if self.randomize_blocks:
            if self._rng is None:
                self._rng = np.random.RandomState()
            idx = np.arange(len(blocks))
            self._rng.shuffle(idx)
            blocks = [blocks[i] for i in idx]

        actions, block_id_of, goal_block_idx = self._flatten_actions(blocks, goals)

        chosen = []
        names = []
        for b_idx, block in enumerate(blocks):
            a_local = randint(0, len(block) - 1)
            global_idx = sum(len(b) for b in blocks[:b_idx]) + a_local
            chosen.append(global_idx)
            names.append(block[a_local][0])

        goal_name = self.goal
        goal_global_idx = len(actions) - len(goals) + [g[0] for g in goals].index(goal_name)
        chosen.append(goal_global_idx)
        names.append(goal_name)

        traverse_name = " -> ".join(names)

        print("\n--------------------------")
        print("Random cleaning strategy:\n", traverse_name)
        print("--------------------------")

        p = self.pipeline(self.dataset, chosen, self.target_goal, self.target_prepare, check_missing)
        rr = (dataset_name, "random", goal_name, self.target_goal,
              self.target_prepare, traverse_name, 'quality_metric', p[1:])
        print(rr)
        return p[1]

    def no_prep(self, dataset_name):
        goals = ["MARS", "LR"]
        if self.goal not in goals:
            raise ValueError("Goal invalid. Choose 'MARS' or 'LR'.")
        check_missing = self.dataset['train'].isnull().sum().sum() > 0
        blocks, _goals = self._build_blocks(check_missing)
        if self.randomize_blocks:
            if self._rng is None:
                self._rng = np.random.RandomState()
            idx = np.arange(len(blocks))
            self._rng.shuffle(idx)
            blocks = [blocks[i] for i in idx]
        goal_global_idx = sum(len(b) for b in blocks) + goals.index(self.goal)
        p = self.pipeline(self.dataset, [goal_global_idx],
                          self.target_goal, self.target_prepare, check_missing)
        rr = (dataset_name, "no-prep", self.goal, self.target_goal,
              self.target_prepare, self.goal, 'quality_metric', p[1:])
        print(rr)
        return p[1]
