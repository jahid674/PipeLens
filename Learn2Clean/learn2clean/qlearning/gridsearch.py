#!/usr/bin/env python3
# coding: utf-8
# Author: Adapted for Learn2Clean

import time
import itertools
import numpy as np
from copy import deepcopy

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
from ..imputation.imputer import Imputer
from ..regression.regressor import Regressor
from ..classification.classifier import Classifier


class GridSearcher:
    """
    Exhaustive grid search for preprocessing pipelines + model.
    Shuffles the pipeline order before evaluation.
    """

    def __init__(self, dataset, goal, target_goal, target_prepare,
                 verbose=False, random_state=None, f_goal=None):
        self.dataset = dataset
        self.goal = goal
        self.target_goal = target_goal
        self.target_prepare = target_prepare
        self.verbose = verbose
        self.random_state = random_state
        self.f_goal = f_goal

        self._blocks = None
        self._goals = None
        self._actions = None
        self._block_offsets = None
        self._goal_global_idx = None

        self._init_blocks_and_actions()

    # -------- Blocks (same as Qlearner) --------
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

        blocks.append([("none_dedup", Deduplicator, {}),
                       ("first",      Deduplicator, {})])

        blocks.append([("NONE_lowercase", Lowercaser, {}),
                       ("LC",             Lowercaser, {})])

        blocks.append([("NONE_translate", LanguageTranslator, {}),
                       ("GT",             LanguageTranslator, {})])

        blocks.append([("NONE_punct", PunctuationRemover, {}),
                       ("PR",         PunctuationRemover, {})])

        blocks.append([("NONE_stopword", StopwordRemover, {}),
                       ("SW",            StopwordRemover, {})])

        blocks.append([("NONE_spell", SpellChecker, {}),
                       ("SC",         SpellChecker, {})])

        blocks.append([("NONE_tokenize", Tokenizer, {}),
                       ("WS",            Tokenizer, {}),
                       ("NLTK",          Tokenizer, {})])

        blocks.append([("NONE_convert", UnitConverter, {}),
                       ("UC",           UnitConverter, {})])

        blocks.append([("NONE_normalization", Normalizer, {}),
                       ("SS",                 Normalizer, {}),
                       ("RS",                 Normalizer, {}),
                       ("MA",                 Normalizer, {}),
                       ("MM",                 Normalizer, {})])

        blocks.append([("NONE_outlier", Outlier_detector, {}),
                       ("IF",           Outlier_detector, {}),
                       ("LOF_1",        Outlier_detector, {}),
                       ("LOF_5",        Outlier_detector, {}),
                       ("LOF_10",       Outlier_detector, {}),
                       ("LOF_20",       Outlier_detector, {}),
                       ("LOF_30",       Outlier_detector, {})])

        goals = [("MARS", Regressor, {}), ("LR", Classifier, {})]
        return blocks, goals

    def _init_blocks_and_actions(self):
        check_missing = self.dataset['train'].copy().isnull().sum().sum() > 0
        blocks, goals = self._build_blocks(check_missing)
        self._blocks = blocks
        self._goals = goals

        actions = []
        block_offsets = []
        offset = 0
        for b in blocks:
            actions.extend(b)
            block_offsets.append(offset)
            offset += len(b)

        goal_offset = offset
        actions.extend(goals)

        self._actions = actions
        self._block_offsets = block_offsets
        self._goal_global_idx = goal_offset + [g[0] for g in goals].index(self.goal)

    # -------- Helpers --------
    def _instantiate_and_run(self, a_idx, dataset):
        name, cls, extra = self._actions[a_idx]

        if a_idx < self._goal_global_idx:
            if cls is Normalizer:
                return cls(dataset=dataset, strategy=name,
                           exclude=self.target_prepare, verbose=self.verbose, **extra).transform()
            else:
                return cls(dataset=dataset, strategy=name,
                           verbose=self.verbose, **extra).transform()
        else:
            if self.goal == "MARS":
                return Regressor(dataset=dataset, strategy="MARS",
                                 target=self.target_goal, verbose=self.verbose, **extra).transform()
            elif self.goal == "LR":
                return Classifier(dataset=dataset, strategy="LR",
                                  target=self.target_goal, verbose=self.verbose, **extra).transform()

    def _run_pipeline(self, action_indices):
        dset = deepcopy(self.dataset)
        res = None
        for idx in action_indices:
            out = self._instantiate_and_run(idx, dset)
            if isinstance(out, dict) and 'train' in out:
                dset = out
            else:
                res = out
        return res

    def _block_global_indices(self, block_id):
        start = self._block_offsets[block_id]
        size = len(self._blocks[block_id])
        return list(range(start, start + size))

    def build_search_space(self):
        block_choices = [self._block_global_indices(b_id) for b_id in range(len(self._blocks))]
        combos = list(itertools.product(*block_choices))
        pipelines = [list(combo) + [self._goal_global_idx] for combo in combos]
        return pipelines

    # -------- Main search --------
    def search(self):
        if self.target_goal != self.dataset['target'].name:
            raise ValueError("Target variable invalid.")
        if self.goal not in ("MARS", "LR"):
            raise ValueError("Goal invalid. Choose 'MARS' or 'LR'.")

        space = self.build_search_space()
        total = len(space)

        rng = np.random.default_rng(self.random_state)
        rng.shuffle(space)

        best_metric, best_pipeline = None, None
        evaluated = 0
        start = time.time()

        for pipe in space:
            metrics = self._run_pipeline(pipe)
            evaluated += 1

            if metrics is None or 'quality_metric' not in metrics:
                continue

            m = metrics['quality_metric']
            if best_metric is None:
                best_metric, best_pipeline = m, pipe
            else:
                if self.goal == "LR" and m > best_metric:
                    best_metric, best_pipeline = m, pipe
                elif self.goal == "MARS" and m < best_metric:
                    best_metric, best_pipeline = m, pipe

            if self.f_goal is not None:
                if self.goal == "LR" and best_metric >= self.f_goal:
                    break
                if self.goal == "MARS" and best_metric <= self.f_goal:
                    break

        elapsed = time.time() - start
        names = [self._actions[i][0] for i in best_pipeline] if best_pipeline else None

        return {
            'best_metric': best_metric,
            'best_pipeline': best_pipeline,
            'best_names': names,
            'evaluated': evaluated,
            'total': total,
            'elapsed_seconds': elapsed
        }
