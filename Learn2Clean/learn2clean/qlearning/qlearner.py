#!/usr/bin/env python3
# coding: utf-8
# Author:Laure Berti-Equille <laure.berti@ird.fr>

# RP - replaced NB by LR for comparison

import warnings
import time
import numpy as np
import re
import random
from random import randint

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


# ---------------- Q-learning helpers ---------------- #

def remove_adjacent(nums):
    """Remove immediate duplicates in a list (e.g., [2,2,3] -> [2,3])."""
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

    Parameters
    ----------
    dataset : dict {'train': DataFrame, 'test': DataFrame, 'target': Series}
    goal : str in {'MARS','LR'}
    target_goal : str, target column name (must equal dataset['target'].name)
    target_prepare : str, column to exclude from normalizers
    verbose : bool
    file_name : str or None
    threshold : any (unused, for API parity)
    f_goal : float, success threshold (accuracy >= f_goal for LR; MSE <= f_goal for MARS)
    """

    # ------------------ Init & params ------------------ #
    def __init__(self, dataset, goal, target_goal, target_prepare,
                 verbose=False, file_name=None, threshold=None, f_goal=0.8):

        self.dataset = dataset
        self.goal = goal
        self.target_goal = target_goal
        self.target_prepare = target_prepare
        self.verbose = verbose
        self.file_name = file_name
        self.threshold = threshold
        self.f_goal = f_goal

        # will be set in Initialization_Reward_Matrix
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
            'threshold': self.threshold
        }
    
    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for Qlearner. Parameter(s) IGNORED. "
                              "Check with `qlearner.get_params().keys()`")
            else:
                setattr(self, k, v)

    # ------------------ Block construction ------------------ #
    def _build_blocks(self, check_missing):
        """
        Ordered blocks. Each entry: (strategy_name, Class, extra_kwargs_dict).
        Agent can 'skip' by picking the NONE_* strategy inside each block.
        """
        blocks = []

        # (0) Imputation – only if there are missing values
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

        # (1) Deduplication
        blocks.append([
            ("none_dedup", Deduplicator, {}),
            ("first",      Deduplicator, {}),
        ])

        # (2) Lowercasing
        blocks.append([
            ("NONE_lowercase", Lowercaser, {}),
            ("LC",             Lowercaser, {}),
        ])

        # (3) Translation
        blocks.append([
            ("NONE_translate", LanguageTranslator, {}),
            ("GT",             LanguageTranslator, {}),
        ])

        # (4) Punctuation removal
        blocks.append([
            ("NONE_punct", PunctuationRemover, {}),
            ("PR",         PunctuationRemover, {}),
        ])

        # (5) Stopword removal
        blocks.append([
            ("NONE_stopword", StopwordRemover, {}),
            ("SW",            StopwordRemover, {}),
        ])

        # (6) Spell checking
        blocks.append([
            ("NONE_spell", SpellChecker, {}),
            ("SC",         SpellChecker, {}),
        ])

        # (7) Tokenization
        blocks.append([
            ("NONE_tokenize", Tokenizer, {}),
            ("WS",            Tokenizer, {}),
            ("NLTK",          Tokenizer, {}),
        ])

        # (8) Unit conversion
        blocks.append([
            ("NONE_convert", UnitConverter, {}),
            ("UC",           UnitConverter, {}),
        ])

        # (9) Normalization
        blocks.append([
            ("NONE_normalization", Normalizer, {}),
            ("SS",                 Normalizer, {}),
            ("RS",                 Normalizer, {}),
            ("MA",                 Normalizer, {}),
            ("MM",                 Normalizer, {}),
        ])

        # (10) Outliers
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
        """Flatten blocks + append goals; also track block indices per action."""
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
        """
        Build R (full) and R_learn (rows excluding terminal rows).
        -1 = forbidden, 0 = allowed to next block, 100 = into goal.
        """
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
            else:
                # goal rows remain -1
                pass

        # Non-terminal learning rows: those with at least one valid move (>=0)
        learn_row_mask = ~np.all(R_full == -1, axis=1)
        R_learn = R_full[learn_row_mask, :]

        Q_learn = np.zeros_like(R_learn, dtype="float32")

        if self.verbose:
            print("Reward matrix shape (learn):", R_learn.shape)
            print(R_learn)

        # Build mapping between learning rows and global indices
        learn_rows_idx = np.where(learn_row_mask)[0]
        global_to_learnrow = {g: i for i, g in enumerate(learn_rows_idx)}

        return (Q_learn, R_learn, R_full, actions, block_id_of, goal_block_idx,
                learn_row_mask, learn_rows_idx, global_to_learnrow)

    # ------------------ Public: build matrices ------------------ #
    def Initialization_Reward_Matrix(self, dataset):
        """Create reward/Q matrices, store mappings, return learning shapes & flags."""
        check_missing = dataset['train'].copy().isnull().sum().sum() > 0
        blocks, goals = self._build_blocks(check_missing)

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

        n_actions = R_learn.shape[0]   # number of learning rows
        n_states = n_actions           # symmetric here

        return Q_learn, R_learn, n_actions, n_states, check_missing

    # ------------------ Pipeline execution ------------------ #
    def _instantiate_and_run(self, a_idx, dataset, target_goal, target_prepare):
        """
        Given global action index, instantiate & run component.
        Returns dataset dict for preprocessing steps; dict with metrics for goals.
        """
        name, cls, extra = self._actions[a_idx]

        # preprocessing block
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

        # goals
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
        """Execute a global-index pipeline. Returns (last_dataset, metrics_dict, elapsed_seconds)."""
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

    # ------------------ Traversal / execution ------------------ #
    def show_traverse(self, dataset, q_learn, g, target1, target2, check_missing):
        """
        Build greedy traversals from learned Q (rows = learning rows),
        map back to global indices, run pipelines, and early stop on success.
        """
        actions_names = [name for (name, _cls, _extra) in self._actions]
        learn_rows_idx = self._learn_rows_idx
        n_states = len(learn_rows_idx) + 1  # conceptual terminal

        results = []
        for i_learnrow, start_global in enumerate(learn_rows_idx):
            actions_list = []
            traverse_name = "%s -> " % actions_names[start_global]
            current_learn_row = i_learnrow
            n_steps = 0

            while current_learn_row < q_learn.shape[0] and n_steps < 64:
                # add current global index
                actions_list.append(learn_rows_idx[current_learn_row])

                # choose best column (global action index)
                next_col = int(np.argmax(q_learn[current_learn_row]))
                traverse_name += "%s -> " % actions_names[next_col]
                actions_list.append(next_col)

                n_steps += 1

                # if next_col is terminal (not a learning row), stop
                if next_col not in self._global_to_learnrow:
                    break

                # otherwise move to that learning row
                current_learn_row = int(self._global_to_learnrow[next_col])

            actions_list = remove_adjacent(actions_list)
            traverse_name = traverse_name[:-4]

            print("\n\nStrategy#", i_learnrow, ": Greedy traversal for start", actions_names[start_global])
            print(traverse_name)

            # append the chosen goal (0=MARS,1=LR)
            goal_global_idx = len(self._actions) - len(self._goals) + g
            if actions_list[-1] != goal_global_idx:
                actions_list.append(goal_global_idx)
                traverse_name += " -> %s" % self._goals[g][0]

            pipeline_result = self.pipeline(dataset, actions_list, target1, target2, check_missing)
            metrics = pipeline_result[1]

            if metrics is not None and 'quality_metric' in metrics:
                print("Quality metric ", metrics['quality_metric'])
                results.append(metrics)

                # early stop if meets f_goal
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

        # fallback: run only the goal
        goal_global_idx = len(self._actions) - len(self._goals) + g
        pipeline_result = self.pipeline(dataset, [goal_global_idx], target1, target2, check_missing)
        results.append(pipeline_result[1])
        return False, i_learnrow + 1 if 'i_learnrow' in locals() else 1

    # ------------------ Main learning loop (terminal-safe) ------------------ #
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
        rng = np.random.RandomState(r_state)

        q_learn, r_learn, n_actions, n_states, check_missing = self.Initialization_Reward_Matrix(self.dataset)

        # rows used for learning: 0..n_actions-1 map to self._learn_rows_idx (global)
        for e in range(n_episodes):
            learn_row_indices = np.arange(n_actions)
            rng.shuffle(learn_row_indices)
            current_row = int(learn_row_indices[0])
            goal_reached = False

            while not goal_reached:
                valid_moves = np.where(r_learn[current_row] >= 0)[0]
                if valid_moves.size == 0:
                    break

                # choose column (global action index)
                if rng.rand() < epsilon:
                    action_col = int(random.choice(valid_moves.tolist()))
                else:
                    if np.sum(q_learn[current_row]) > 0:
                        action_col = int(np.argmax(q_learn[current_row]))
                    else:
                        action_col = int(random.choice(valid_moves.tolist()))

                reward = float(r_learn[current_row, action_col])

                # terminal if chosen column is not a learning row (e.g., goal)
                if action_col in self._global_to_learnrow:
                    next_row = int(self._global_to_learnrow[action_col])
                    max_future = float(np.max(q_learn[next_row])) if q_learn.shape[0] > 0 else 0.0
                else:
                    next_row = None
                    max_future = 0.0

                # Q-update (terminal-safe)
                qsa = q_learn[current_row, action_col]
                target = reward + gamma * max_future
                new_q = qsa + beta * (target - qsa)
                q_learn[current_row, action_col] = new_q

                # renormalize positives on current_row
                row = q_learn[current_row]
                pos = row > 0
                if np.any(pos):
                    q_learn[current_row, pos] = row[pos] / np.sum(row[pos])

                if reward > 1:  # 100 when stepping into a goal
                    goal_reached = True
                elif next_row is not None:
                    current_row = next_row
                else:
                    break  # terminal without explicit 100 reward (shouldn't happen under our R)

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

    # ------------------ Baselines ------------------ #
    def random_cleaning(self, dataset_name):
        random.seed(time.time())
        check_missing = self.dataset['train'].isnull().sum().sum() > 0
        blocks, goals = self._build_blocks(check_missing)
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
        goal_global_idx = sum(len(b) for b in blocks) + goals.index(self.goal)
        p = self.pipeline(self.dataset, [goal_global_idx],
                          self.target_goal, self.target_prepare, check_missing)
        rr = (dataset_name, "no-prep", self.goal, self.target_goal,
              self.target_prepare, self.goal, 'quality_metric', p[1:])
        print(rr)
        return p[1]

