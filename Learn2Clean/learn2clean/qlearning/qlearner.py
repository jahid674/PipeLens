#!/usr/bin/env python3
# coding: utf-8

import warnings
import time
import numpy as np
import random
from random import randint

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter('ignore', category=ImportWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)

# ---------------------- existing Learn2Clean modules ---------------------- #
from ..normalization.normalizer import Normalizer
from ..deduplication.deduplicator import Deduplicator
from ..punctualion_remove.punctuation_remover import PunctuationRemover
from ..spellcheck.spellchecker import SpellChecker
from ..stopword_remove.stopword_remover import StopwordRemover
from ..tokenize.tokenizer import Tokenizer
from ..lowercase.lowercaser import Lowercaser
from ..language_translator.translator import LanguageTranslator
from ..outlier_detection.outlier_detector import Outlier_detector
from ..consistency_checking.consistency_checker import Consistency_checker  # optional
from ..imputation.imputer import Imputer
from ..clustering.clusterer import Clusterer
from ..regression.regressor import Regressor
from ..classification.classifier import Classifier

# ---------------------- converted / additional modules ---------------------- #
from ..unit_convert.unit_converter import UnitConverter

# DataFrame-based modules (your versions)
from ..whitespace.whitespace import WhitespaceCleaner
from ..fselection.fselection import FeatureSelector

# Dict-based modules (your integrated design)
from ..sampling.sampling import DataSampler
from ..invalid_value.invalid_value import InvalidValueRepair
from ..floating_point.floating_point import FloatingPointStabilizer
from ..distribution_shape.distribution_shape import DistributionShapeCorrector
from ..multicollinearity.multicollinearity import VIFMulticollinearityCleaner
from ..poly_pca.poly_pca import PolyPCATransformer


def remove_adjacent(nums):
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
      - If dataset_name == 'adult': quality_metric = 1 - |SP|  (higher is better)
      - Else:                       quality_metric = accuracy (higher is better)

    Policy:
      - BEFORE running ANY module (except Imputer), DROP NaN/inf in FEATURES ONLY
        (train/test DataFrames). Target and sensitive are NEVER altered.
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
        dataset_name: str = "",
        include_consistency_checker: bool = False,  # default OFF
        sensitive_col: str = "Sex",                 # adult default
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
        self.include_consistency_checker = include_consistency_checker
        self.sensitive_col = sensitive_col

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

    # ------------------------------------------------------------------ #
    #      DO NOT ALTER target/sensitive: keep immutable + separate        #
    # ------------------------------------------------------------------ #
    def _ensure_full_label_store(self, dataset: dict) -> dict:
        """
        Store full target/sensitive once; NEVER overwrite/trim them.

        KEY BEHAVIOR (per your request):
        - During preprocessing: keep train/test FEATURE-ONLY (drop Sex from features)
        - Right before training LR (Classifier): BRING Sex back into train (aligned by index)
            so Classifier can read it from dataset['train']['Sex'].
        """
        out = dataset if isinstance(dataset, dict) else {}
        out = out.copy()

        # Preserve full series
        if "_target_full" not in out:
            out["_target_full"] = out.get("target", None)
        if "_sensitive_full" not in out:
            # If sensitive is not provided separately, try to cache from train[Sex]
            s = out.get("sensitive", None)
            if s is None:
                Xtr0 = out.get("train", None)
                if Xtr0 is not None and self.sensitive_col in Xtr0.columns:
                    s = Xtr0[self.sensitive_col].copy()
            out["_sensitive_full"] = s

        # Force feature-only frames (do NOT mutate target/sensitive series)
        for split in ["train", "test"]:
            X = out.get(split, None)
            if X is None:
                continue

            # drop target column if it exists in features
            if self.target_prepare and self.target_prepare in X.columns:
                X = X.drop(columns=[self.target_prepare])

            # drop sensitive column if it exists in features (keep it in _sensitive_full)
            if self.sensitive_col and self.sensitive_col in X.columns:
                X = X.drop(columns=[self.sensitive_col])

            out[split] = X

        # Expose immutable series (not sliced here)
        out["target"] = out.get("_target_full", None)
        out["sensitive"] = out.get("_sensitive_full", None)

        return out


    def _prepare_for_goal(self, dataset: dict) -> dict:
        """
        Before LR/MARS:
        - ensure features are ready
        - BRING BACK Sex column into train/test (aligned by index) so Classifier can use it
        """
        out = self._ensure_full_label_store(dataset)

        sfull = out.get("_sensitive_full", out.get("sensitive", None))
        if sfull is None:
            return out

        # Bring Sex back ONLY for goal evaluation (LR/MARS), aligned by index
        for split in ["train", "test"]:
            X = out.get(split, None)
            if X is None:
                continue

            # Align sensitive to the current feature index (important after row drops/outliers)
            try:
                s_aligned = sfull.reindex(X.index)
            except Exception:
                # fallback: if it isn't a Series with index
                s_aligned = None

            if s_aligned is not None:
                X2 = X.copy()
                X2[self.sensitive_col] = s_aligned
                out[split] = X2

        return out


    # ---------------------- REQUIRED missing/inf dropping ---------------------- #
    def _drop_nan_inf_rows_df(self, df):
        """
        Drop rows with NaN/inf in a FEATURES DataFrame, preserving index.
        """
        if df is None:
            return None
        df2 = df.replace([np.inf, -np.inf], np.nan)
        return df2.dropna(axis=0, how="any")

    def _drop_missing_features_only_except_imputer(self, dataset: dict) -> dict:
        """
        Your rule: drop missing values before any module EXCEPT Imputer.
        IMPORTANT: This applies ONLY to FEATURES (train/test DataFrames).
                   target/sensitive are never modified.
        """
        out = self._ensure_full_label_store(dataset)

        Xtr0 = out.get("train", None)
        if Xtr0 is not None:
            out["train"] = self._drop_nan_inf_rows_df(Xtr0)

        Xte0 = out.get("test", None)
        if Xte0 is not None:
            out["test"] = self._drop_nan_inf_rows_df(Xte0)

        return out

    # --------------------------- blocks --------------------------- #
    def _build_blocks(self, check_missing):
        blocks = []

        # # (0) Data sampling
        # blocks.append([
        #     ("FULL",       DataSampler, {}),
        #     ("RANDOM",     DataSampler, {}),
        #     ("SNAPSHOT",   DataSampler, {}),
        # ])

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

        # # (3) Deduplication
        # blocks.append([
        #     ("none_dedup", Deduplicator, {}),
        #     ("first",      Deduplicator, {}),
        # ])

        # (4) Whitespace cleaning
        blocks.append([
            ("NONE", WhitespaceCleaner, {}),
            ("WC",   WhitespaceCleaner, {}),
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
            ("NONE_CONVERT", UnitConverter, {}),
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

        # # (11) Distribution shape correction
        # blocks.append([
        #     ("NONE",       DistributionShapeCorrector, {}),
        #     ("LOG1P",      DistributionShapeCorrector, {}),
        #     ("SQRT",       DistributionShapeCorrector, {}),
        #     ("YEOJOHNSON", DistributionShapeCorrector, {}),
        # ])

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

        # (13) Consistency checker (optional)
        if self.include_consistency_checker:
            blocks.append([
                ("CC", Consistency_checker, {}),
                ("PC", Consistency_checker, {}),
            ])

        # # (14) VIF multicollinearity cleaning
        # blocks.append([
        #     ("NONE",          VIFMulticollinearityCleaner, {}),
        #     ("DROP_HIGH_VIF", VIFMulticollinearityCleaner, {}),
        # ])

        # # (15) Feature Selection
        # blocks.append([
        #     ("NONE",        FeatureSelector, {}),
        #     ("VARIANCE",    FeatureSelector, {}),
        #     ("MUTUAL_INFO", FeatureSelector, {}),
        # ])

        # # (16) Polynomial + reducer
        # blocks.append([
        #     ("NONE",               PolyPCATransformer, {}),
        #     ("PCA",                PolyPCATransformer, {}),
        #     ("SPARSEPCA",          PolyPCATransformer, {}),
        #     ("MINIBATCHSPARSEPCA", PolyPCATransformer, {}),
        #     ("KERNELPCA",          PolyPCATransformer, {}),
        # ])

        goals = [("MARS", Regressor, {}), ("LR", Classifier, {}), ("NN", Classifier, {})]
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

    def _maybe_shuffle_blocks(self, blocks):
        if not self.randomize_blocks:
            return blocks
        if self._rng is None:
            self._rng = np.random.RandomState()

        fixed_prefix = []
        rest = blocks[:]

        # keep sampler + invalid repair fixed
        if len(rest) >= 1:
            fixed_prefix.append(rest.pop(0))
        if len(rest) >= 1:
            fixed_prefix.append(rest.pop(0))

        # keep imputer fixed if present next
        if len(rest) >= 1:
            has_imputer = any((cls is Imputer) for (_n, cls, _e) in rest[0])
            if has_imputer:
                fixed_prefix.append(rest.pop(0))

        idx = np.arange(len(rest))
        self._rng.shuffle(idx)
        rest = [rest[i] for i in idx]
        return fixed_prefix + rest

    def Initialization_Reward_Matrix(self, dataset):
        dataset = self._ensure_full_label_store(dataset)

        # Check missingness ONLY on features
        check_missing = dataset["train"].copy().isnull().sum().sum() > 0 if dataset.get("train") is not None else False

        blocks, goals = self._build_blocks(check_missing)
        blocks = self._maybe_shuffle_blocks(blocks)

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

    # ---------------------- instantiate & run ---------------------- #
    def _instantiate_and_run(self, a_idx, dataset, target_goal, target_prepare):
        name, cls, extra = self._actions[a_idx]

        # preprocessing
        if self._block_id_of[a_idx] < self._goal_block_idx:

            dataset = self._ensure_full_label_store(dataset)

            # Drop NaN/inf before any module EXCEPT Imputer (FEATURES ONLY)
            if cls is not Imputer:
                dataset = self._drop_missing_features_only_except_imputer(dataset)

            # (A) DataSampler
            if cls is DataSampler:
                out = cls(dataset=dataset, strategy=str(name).upper().strip(),
                          verbose=self.verbose, **extra).transform()
                return self._ensure_full_label_store(out)

            # (B) WhitespaceCleaner (DataFrame-based)
            if cls is WhitespaceCleaner:
                strat = "wc" if str(name).upper().strip() == "WC" else "none"
                df_tr = dataset.get('train', None)
                df_te = dataset.get('test', None)

                out = dataset.copy()
                if df_tr is not None:
                    out['train'] = WhitespaceCleaner(dataset=df_tr, strategy=strat, verbose=self.verbose).transform()
                if df_te is not None:
                    out['test'] = WhitespaceCleaner(dataset=df_te, strategy=strat, verbose=self.verbose).transform()

                return self._ensure_full_label_store(out)

            # (C) FeatureSelector
            if cls is FeatureSelector:
                nm = str(name).upper().strip()
                if nm == "NONE":
                    fs_strategy = "NONE_fs"
                elif nm == "VARIANCE":
                    fs_strategy = "VARIANCE"
                elif nm == "MUTUAL_INFO":
                    fs_strategy = "MUTUAL_INFO"
                else:
                    fs_strategy = "NONE_fs"

                X = dataset.get('train', None)
                if X is None:
                    return dataset

                # IMPORTANT: use FULL target series; selector should only look up by X.index
                yfull = dataset.get("_target_full", dataset.get("target", None))
                y = None if yfull is None else yfull.loc[X.index]

                if y is None and fs_strategy == "MUTUAL_INFO":
                    raise ValueError("FeatureSelector MUTUAL_INFO requires dataset['target'].")

                fs = FeatureSelector(dataset=X, strategy=fs_strategy, verbose=self.verbose, **extra)
                X_new = fs.transform(y_train=y)

                out = dataset.copy()
                out['train'] = X_new

                if 'test' in dataset and dataset['test'] is not None:
                    common_cols = [c for c in X_new.columns if c in dataset['test'].columns]
                    out['test'] = dataset['test'][common_cols].copy()

                return self._ensure_full_label_store(out)

            # (D) Normalizer
            if cls is Normalizer:
                out = cls(dataset=dataset, strategy=name,
                          exclude=target_prepare, verbose=self.verbose, **extra).transform()
                return self._ensure_full_label_store(out)

            # (E) UnitConverter
            if cls is UnitConverter:
                out = cls(dataset=dataset, strategy=name,
                          exclude=target_prepare, verbose=self.verbose, **extra).transform()
                return self._ensure_full_label_store(out)

            # (F) PolyPCATransformer
            if cls is PolyPCATransformer:
                reducer = str(name).lower().strip()
                out = cls(dataset=dataset, reducer=reducer, verbose=self.verbose, **extra).transform()
                return self._ensure_full_label_store(out)

            # (G) DistributionShapeCorrector
            if cls is DistributionShapeCorrector:
                s = str(name).lower().strip()
                out = cls(dataset=dataset, strategy=s,
                          exclude=[target_prepare] if target_prepare else None,
                          verbose=self.verbose, **extra).transform()
                return self._ensure_full_label_store(out)

            # (H) Consistency checker (optional)
            if cls is Consistency_checker:
                fname = self.file_name or ""
                out = cls(dataset=dataset, strategy=str(name).upper().strip(),
                          file_name=fname, verbose=self.verbose, **extra).transform()
                return self._ensure_full_label_store(out)

            # (I) Remaining dict-based modules (including Imputer)
            out = cls(dataset=dataset, strategy=name, verbose=self.verbose, **extra).transform()
            return self._ensure_full_label_store(out)

        # ---------------- goals ---------------- #
        if name != self.goal:
            return dataset

        d_goal = self._prepare_for_goal(dataset)

        print()
        if name == "MARS":
            return Regressor(dataset=d_goal, strategy="MARS",
                             target=target_goal, verbose=self.verbose, **extra).transform()

        if name == "LR":
            # IMPORTANT: Classifier must compute sensitive from dataset['sensitive'] (not train['Sex'])
            return Classifier(dataset=d_goal, strategy="LR",
                              target=target_goal, verbose=self.verbose,
                              dataset_name=self.dataset_name, **extra).transform()
        

        if name == "NN":
            # IMPORTANT: Classifier must compute sensitive from dataset['sensitive'] (not train['Sex'])
            return Classifier(dataset=d_goal, strategy="NN",
                              target=target_goal, verbose=self.verbose,
                              dataset_name=self.dataset_name, **extra).transform()
        

        raise ValueError("Unknown goal: %s" % name)

    # ---------------------- pipeline ---------------------- #
    def pipeline(self, dataset, actions_list, target_goal, target_prepare, check_missing):
        dataset = self._ensure_full_label_store(dataset.copy())

        print("\nStart pipeline")
        print("-------------")
        start_time = time.time()

        res = None
        n = None

        for a_global in actions_list:
            n = self._instantiate_and_run(a_global, dataset, target_goal, target_prepare)

            if isinstance(n, dict) and 'train' in n:
                dataset = self._ensure_full_label_store(n)

            if self._block_id_of[a_global] == self._goal_block_idx:
                res = n

        t = time.time() - start_time
        print("End Pipeline CPU time: %s seconds" % t)
        return n, res, t

    # ---------------------- traverse ---------------------- #
    def show_traverse(self, dataset, q_learn, g, target1, target2, check_missing):
        actions_names = [name for (name, _cls, _extra) in self._actions]
        learn_rows_idx = self._learn_rows_idx

        goal_start = len(self._actions) - len(self._goals)
        desired_goal_global = goal_start + g

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
                    qm = float(qm) if qm is not None else None
                except Exception:
                    qm = None
                metrics['quality_metric'] = qm
                print("Quality metric ", qm)

                if self._goals[g][0] == "LR" and qm is not None and qm >= self.f_goal:
                    print("Achieved in ", i_learnrow + 1)
                    return True, i_learnrow + 1

                if self._goals[g][0] == "MARS" and qm is not None and qm <= self.f_goal:
                    print("Achieved in ", i_learnrow + 1)
                    return True, i_learnrow + 1
                
                if self._goals[g][0] == "NN" and qm is not None and qm >= self.f_goal:
                    print("Achieved in ", i_learnrow + 1)
                    return True, i_learnrow + 1

        return False, i_learnrow + 1 if 'i_learnrow' in locals() else 1

    # ---------------------- learning loop ---------------------- #
    def learn2clean(self, r_state):
        goals_names = ["MARS", "LR", "NN"]
        if self.goal not in goals_names:
            raise ValueError("Goal invalid. Choose 'MARS' or 'LR'.")
        g = goals_names.index(self.goal)

        # IMPORTANT: target name check uses immutable series name
        if self.dataset.get("target", None) is None or self.target_goal != self.dataset['target'].name:
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
