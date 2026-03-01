#!/usr/bin/env python3
# coding: utf-8

import warnings
import time
import numpy as np
import random
from random import randint

from ..normalization.normalizer import Normalizer
from ..deduplication.deduplicator import Deduplicator

# NOTE: keep your repo's actual path (you used "punctualion_remove" earlier)
from ..punctualion_remove.punctuation_remover import PunctuationRemover

from ..spellcheck.spellchecker import SpellChecker
from ..stopword_remove.stopword_remover import StopwordRemover
from ..tokenize.tokenizer import Tokenizer
from ..lowercase.lowercaser import Lowercaser
from ..language_translator.translator import LanguageTranslator
from ..outlier_detection.outlier_detector import Outlier_detector
from ..consistency_checking.consistency_checker import Consistency_checker
from ..imputation.imputer import Imputer
from ..clustering.clusterer import Clusterer
from ..regression.regressor import Regressor
from ..classification.classifier import Classifier

# ---------------------- converted / additional modules ---------------------- #
from ..unit_convert.unit_converter import UnitConverter
from ..whitespace.whitespace import WhitespaceCleaner  # <-- adjust path
from ..feature_selection.feature_selector import FeatureSelector     # <-- your L2C-compatible FS class

from ..sampling.sampling import DataSampler            # <-- adjust to your actual path
from ..invalid_value.invalid_value import InvalidValueRepair  # <-- adjust path
from ..floating_point.floating_point import FloatingPointStabilizer  # <-- adjust path
from ..distribution_shape.distribution_shape import DistributionShapeCorrector  # <-- adjust path
from ..multicollinearity.multicollinearity import VIFMulticollinearityCleaner        # <-- adjust path
from ..poly_pca.poly_pca import PolyPCATransformer       


def remove_adjacent(nums):
    """Remove immediate duplicates in a list (e.g., [2,2,3] -> [2,3])."""
    previous = object()
    out = []
    for i in nums:
        if i != previous:
            out.append(i)
        previous = i
    return out


class Qlearner:
    """
    Learn2Clean Q-learning over ordered preprocessing blocks, then model.

    Objective:
      - If dataset_name == 'adult': quality_metric = 1 - |SP| (higher is better; set f_goal accordingly)
      - Else:                       quality_metric = accuracy (higher is better)
    """

    def __init__(
        self,
        dataset,
        goal,
        target_goal,
        target_prepare,
        verbose=False,
        file_name=None,
        threshold=None,
        f_goal=0.8,
        randomize_blocks=True,
        dataset_name="",
    ):
        self.dataset = dataset
        self.goal = goal
        self.target_goal = target_goal
        self.target_prepare = target_prepare
        self.verbose = verbose
        self.file_name = file_name
        self.threshold = threshold
        self.f_goal = f_goal

        self.randomize_blocks = randomize_blocks
        self.dataset_name = (dataset_name or "").lower()

        self._rng = None

        # set during Initialization_Reward_Matrix
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
            'dataset_name': self.dataset_name,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn(
                    "Invalid parameter(s) for Qlearner. Parameter(s) IGNORED. "
                    "Check with `qlearner.get_params().keys()`"
                )
            else:
                setattr(self, k, v)

    # -------- Build blocks -------- #
    def _build_blocks(self, check_missing):
        """
        Ordered blocks. Each entry in blocks is a LIST of ACTIONS.
        Each action: (strategy_name, Class, extra_kwargs_dict).
        """
        blocks = []

        # (0) Data sampling (MUST keep y/sensitive aligned)
        blocks.append([
            ("FULL",       DataSampler, {}),
            ("RANDOM",     DataSampler, {}),
            ("SNAPSHOT",   DataSampler, {}),
            ("STRATIFIED", DataSampler, {}),
        ])

        # (1) Invalid value repair
        blocks.append([
            ("NONE",     InvalidValueRepair, {}),
            ("SENTINEL", InvalidValueRepair, {}),
            ("REGEX",    InvalidValueRepair, {}),
            ("BOTH",     InvalidValueRepair, {}),
        ])

        # (2) Imputation – only if there are missing values
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

        # (3) Deduplication (optional; was commented in your original build)
        blocks.append([
            ("none_dedup", Deduplicator, {}),
            ("first",      Deduplicator, {}),
        ])

        # (4) Whitespace cleaning (text)
        blocks.append([
            ("none", WhitespaceCleaner, {}),
            ("wc",   WhitespaceCleaner, {}),
        ])

        # (5) Punctuation removal
        blocks.append([
            ("NONE_punct", PunctuationRemover, {}),
            ("PR",         PunctuationRemover, {}),
        ])

        # (6) Stopword removal
        blocks.append([
            ("NONE_stopword", StopwordRemover, {}),
            ("SW",            StopwordRemover, {}),
        ])

        # (7) Lowercase
        blocks.append([
            ("NONE_lowercase", Lowercaser, {}),
            ("LC",             Lowercaser, {}),
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

        # (10) Floating point stabilization
        blocks.append([
            ("NONE",  FloatingPointStabilizer, {}),
            ("SNAP",  FloatingPointStabilizer, {}),
            ("ROUND", FloatingPointStabilizer, {}),
            ("BOTH",  FloatingPointStabilizer, {}),
        ])

        # (11) Distribution shape correction
        blocks.append([
            ("NONE",       DistributionShapeCorrector, {}),
            ("LOG1P",      DistributionShapeCorrector, {}),
            ("SQRT",       DistributionShapeCorrector, {}),
            ("BOXCOX",     DistributionShapeCorrector, {}),
            ("YEOJOHNSON", DistributionShapeCorrector, {}),
        ])

        # (12) Outliers
        blocks.append([
            ("NONE_outlier", Outlier_detector, {}),
            ("IF",           Outlier_detector, {}),
            ("LOF_1",        Outlier_detector, {}),
            ("LOF_5",        Outlier_detector, {}),
            ("LOF_10",       Outlier_detector, {}),
            ("LOF_20",       Outlier_detector, {}),
            ("LOF_30",       Outlier_detector, {}),
        ])

        # (13) Consistency checking (if you actually use it; otherwise keep NONE only)
        blocks.append([
            ("NONE_consistency", Consistency_checker, {}),
            ("CC",               Consistency_checker, {}),
        ])

        # (14) VIF multicollinearity cleaning
        blocks.append([
            ("NONE",          VIFMulticollinearityCleaner, {}),
            ("DROP_HIGH_VIF", VIFMulticollinearityCleaner, {}),
        ])

        # (15) Feature selection
        blocks.append([
            ("NONE",       FeatureSelector, {}),
            ("VARIANCE",   FeatureSelector, {}),
            ("MUTUAL_INFO", FeatureSelector, {}),
        ])

        # (16) Polynomial expansion + reducer (PCA/SPCA/MBSPCA/KPCA)
        # PolyPCATransformer uses `reducer=` not `strategy=`.
        blocks.append([
            ("NONE",               PolyPCATransformer, {}),
            ("PCA",                PolyPCATransformer, {}),
            ("SPARSEPCA",          PolyPCATransformer, {}),
            ("MINIBATCHSPARSEPCA", PolyPCATransformer, {}),
            ("KERNELPCA",          PolyPCATransformer, {}),
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
                for j in range(n):
                    if block_id_of[j] == bi + 1:
                        R_full[i, j] = 0.0
            elif bi == goal_block_idx - 1:
                for j in range(n):
                    if block_id_of[j] == goal_block_idx:
                        R_full[i, j] = 100.0 if actions[j][0] == self.goal else -1.0

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

        # randomize block order (your original behavior) — WARNING: can break dependencies
        if self.randomize_blocks:
            if self._rng is None:
                self._rng = np.random.RandomState()
            idx = np.arange(len(blocks))
            self._rng.shuffle(idx)
            blocks = [blocks[i] for i in idx]

        (Q_learn, R_learn, R_full, actions, block_id_of, goal_block_idx,
         learn_row_mask, learn_rows_idx, global_to_learnrow) = self._build_reward_matrix(blocks, goals)

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

    # ------------------ Pipeline execution ------------------ #
    def _instantiate_and_run(self, a_idx, dataset, target_goal, target_prepare):
        name, cls, extra = self._actions[a_idx]

        # preprocessing step
        if self._block_id_of[a_idx] < self._goal_block_idx:

            # --- Sampler: special (must keep y/sensitive aligned) ---
            if cls is DataSampler:
                strat = str(name).lower()
                if strat == "full":
                    strategy = "full"
                elif strat == "random":
                    strategy = "random"
                elif strat == "snapshot":
                    strategy = "snapshot"
                elif strat == "stratified":
                    strategy = "stratified"
                else:
                    strategy = "full"

                sampler = DataSampler(
                    dataset=dataset["train"],
                    strategy=strategy,
                    random_state=42,
                    verbose=self.verbose,
                    **extra
                )
                X_new, y_new, s_new = sampler.transform(
                    y=dataset.get("target", None),
                    sensitive=dataset.get("sensitive", None),
                )
                out = dataset.copy()
                out["train"] = X_new
                if y_new is not None:
                    out["target"] = y_new
                if s_new is not None:
                    out["sensitive"] = s_new
                return out

            # --- Normalizer: exclude target_prepare ---
            if cls is Normalizer:
                return cls(
                    dataset=dataset,
                    strategy=name,
                    exclude=target_prepare,
                    verbose=self.verbose,
                    **extra
                ).transform()

            # --- PolyPCATransformer: uses reducer=... not strategy=... ---
            if cls is PolyPCATransformer:
                reducer = str(name).lower()
                # map block labels -> reducer enum
                if reducer == "none":
                    reducer = "none"
                elif reducer == "pca":
                    reducer = "pca"
                elif reducer == "sparsepca":
                    reducer = "sparsepca"
                elif reducer == "minibatchsparsepca":
                    reducer = "minibatchsparsepca"
                elif reducer == "kernelpca":
                    reducer = "kernelpca"
                return cls(
                    dataset=dataset,
                    reducer=reducer,
                    verbose=self.verbose,
                    **extra
                ).transform()

            # --- DistributionShapeCorrector: normalize strategy name ---
            if cls is DistributionShapeCorrector:
                s = str(name).lower()
                return cls(
                    dataset=dataset,
                    strategy=s,
                    exclude=target_prepare,
                    verbose=self.verbose,
                    **extra
                ).transform()

            # --- FeatureSelector: MI requires y_train ---
            if cls is FeatureSelector:
                y_train = dataset.get("target", None)
                return cls(
                    dataset=dataset,
                    strategy=str(name).lower(),
                    verbose=self.verbose,
                    **extra
                ).transform(y_train=y_train)

            # --- WhitespaceCleaner: expects 'wc'/'none' (your class lowercases) ---
            if cls is WhitespaceCleaner:
                return cls(
                    dataset=dataset,
                    strategy=str(name).lower(),
                    verbose=self.verbose,
                    **extra
                ).transform()

            # --- Standard modules (Learn2Clean-style) ---
            if cls in (
                Tokenizer, Lowercaser, PunctuationRemover, StopwordRemover,
                SpellChecker, LanguageTranslator, Deduplicator, UnitConverter,
                Consistency_checker, Clusterer,
                InvalidValueRepair, FloatingPointStabilizer,
                VIFMulticollinearityCleaner
            ):
                return cls(dataset=dataset, strategy=name, verbose=self.verbose, **extra).transform()

            if cls is Imputer:
                return cls(dataset=dataset, strategy=name, verbose=self.verbose, **extra).transform()

            if cls is Outlier_detector:
                return cls(dataset=dataset, strategy=name, verbose=self.verbose, **extra).transform()

            # fallback
            return cls(dataset=dataset, strategy=name, verbose=self.verbose, **extra).transform()

        # goal (terminal)
        if name != self.goal:
            return dataset  # safeguard

        if name == "MARS":
            return Regressor(
                dataset=dataset,
                strategy="MARS",
                target=target_goal,
                verbose=self.verbose,
                **extra
            ).transform()

        if name == "LR":
            return Classifier(
                dataset=dataset,
                strategy="LR",
                target=target_goal,
                verbose=self.verbose,
                dataset_name=self.dataset_name,
                **extra
            ).transform()

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

        goal_start = len(self._actions) - len(self._goals)
        desired_goal_global = goal_start + g

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
            actions_list = [
                a for a in actions_list
                if (self._block_id_of[a] != self._goal_block_idx) or (a == desired_goal_global)
            ]
            traverse_name = traverse_name[:-4]

            print("\n\nStrategy#", i_learnrow, ": Greedy traversal for start", actions_names[start_global])
            print(traverse_name)

            if not actions_list or actions_list[-1] != desired_goal_global:
                actions_list.append(desired_goal_global)
                traverse_name += " -> %s" % self._goals[g][0]

            pipeline_result = self.pipeline(dataset, actions_list, target1, target2, check_missing)
            metrics = pipeline_result[1]

            qm = None
            if isinstance(metrics, dict):
                qm = metrics.get('quality_metric', None)
                try:
                    if qm is not None:
                        qm = float(qm)
                except Exception:
                    qm = None
                metrics['quality_metric'] = qm
                print("Quality metric ", qm)

                results.append(metrics)

                if self._goals[g][0] == "LR" and qm is not None and qm >= self.f_goal:
                    print("Pipeline : ", traverse_name)
                    print(metrics)
                    print("Achieved in ", i_learnrow + 1)
                    return True, i_learnrow + 1

                if self._goals[g][0] == "MARS" and qm is not None and qm <= self.f_goal:
                    print("Pipeline : ", traverse_name)
                    print(metrics)
                    print("Achieved in ", i_learnrow + 1)
                    return True, i_learnrow + 1

        pipeline_result = self.pipeline(dataset, [desired_goal_global], target1, target2, check_missing)
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

        rng = np.random.RandomState(r_state)
        self._rng = rng

        q_learn, r_learn, n_actions, n_states, check_missing = self.Initialization_Reward_Matrix(self.dataset)

        for _e in range(n_episodes):
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
                q_learn[current_row, action_col] = qsa + beta * (target - qsa)

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

        if self.randomize_blocks:
            if self._rng is None:
                self._rng = np.random.RandomState()
            idx = np.arange(len(blocks))
            self._rng.shuffle(idx)
            blocks = [blocks[i] for i in idx]

        actions, block_id_of, goal_block_idx = self._flatten_actions(blocks, goals)

        chosen = []
        names = []
        offset = 0
        for block in blocks:
            a_local = randint(0, len(block) - 1)
            chosen.append(offset + a_local)
            names.append(block[a_local][0])
            offset += len(block)

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
