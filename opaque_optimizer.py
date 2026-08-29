import logging
import pandas as pd
import numpy as np
from pipeline_execution import PipelineExecutor
import itertools
import operator
import re


class ScoreLookup:
    """Scoring helper kept local to the opaque optimizer."""

    def __init__(self, pipeline_order, metric_type):
        self.pipeline_order = pipeline_order
        self.metric_type = metric_type

    def utility_look_up(self, profiles_df, elem):
        column_names = self.pipeline_order + [f'utility_{self.metric_type}']
        try:
            return round(profiles_df.loc[
                (profiles_df[column_names[0]] == elem[0])
                & (profiles_df[column_names[1]] == elem[1])
                & (profiles_df[column_names[2]] == elem[2])
            ].iloc[0][f'utility_{self.metric_type}'], 5)
        except Exception as e:
            print(e)
            import pdb
            pdb.set_trace()

    def identify_param(self, rank_list, comb_size):
        return list(itertools.combinations(rank_list, comb_size))

    def score_values(self, historical_data, coefs, comb_size, comb):
        coef_lst = [coefs[comb[i]] for i in range(comb_size)]
        score = {}
        for param in historical_data:
            result = [param[i] for i in list(comb)]
            score[tuple(param)] = sum(x * y for x, y in zip(coef_lst, result))
        return sorted(score.items(), key=operator.itemgetter(1))


class OpaqueOptimizer:
    def __init__(self, dataset_name, model_type, metric_type, pipeline_type, pipeline_order, filename_train, filename_test):
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.metric_type = metric_type
        self.pipeline_type = pipeline_type
        self.pipeline_order = pipeline_order
        self.filename_train = filename_train
        self.filename_test = filename_test

        self.fail = 0
        self.pass_ = 0
        self.fail_with_fallback = 0
        self.rank_iter = 0
        self.rank_f = 0
        self.ranges = {}
        self.base_strategies = pipeline_order

        self.historical_data_pd = pd.read_csv(self.filename_test)
        self.historical_data = self.historical_data_pd.values.tolist()

        self.executor_pass = PipelineExecutor(
            pipeline_type=self.pipeline_type,
            dataset_name=self.dataset_name,
            metric_type=self.metric_type,
            pipeline_ord=self.pipeline_order
        )

        self.pasing_hist_data = pd.read_csv(self.filename_train)
        self.coefs, self.coef_rank = self.executor_pass.score_parameter(self.pasing_hist_data)

        # Fix stale/out-of-range ranking indices.
        self.clean_coef_rank()

        self.score_lookup = ScoreLookup(pipeline_order, metric_type)

    def set_ranges(self):
        for strategy in self.pipeline_order:
            self.ranges[strategy] = list(np.unique(self.pasing_hist_data[strategy]))

    def clean_coef_rank(self):
        """
        Keep only ranking indices that correspond to existing base pipeline components.
        """
        clean_rank = []

        for val in self.coef_rank:
            try:
                idx = int(val)
            except Exception:
                continue

            if 0 <= idx < len(self.base_strategies):
                clean_rank.append(idx)
            else:
                print(
                    f"Warning: coef_rank index {idx} is out of range for "
                    f"{len(self.base_strategies)} base strategies. Skipping it."
                )

        self.coef_rank = clean_rank

    def make_safe_params(self, params):
        """
        Fix invalid parameter indices before calling current_par_lookup.

        Example:
        If stopword=3 but valid values are 1..2, this converts 3 -> 2.
        """
        safe_params = []

        for step, val in zip(self.base_strategies, params):
            val = int(val)

            try:
                self.executor_pass._safe_param_index(step, val)
                safe_params.append(val)

            except ValueError as e:
                msg = str(e)

                match = re.search(r"valid are\s+(-?\d+)\.\.(-?\d+)", msg)

                if match:
                    min_valid = int(match.group(1))
                    max_valid = int(match.group(2))

                    fixed_val = min(max(val, min_valid), max_valid)

                    print(
                        f"Warning: Invalid parameter for '{step}'. "
                        f"Got {val}, using {fixed_val} instead."
                    )

                    safe_params.append(fixed_val)
                else:
                    raise e

        return safe_params

    def optimize(self, init_params, f_goal, max_iterations=500):
        self.rank_iter = 0
        self.rank_f = -1

        cur_params = init_params.copy()

        cur_params_opt = {
            strategy: selection
            for strategy, selection in zip(
                self.base_strategies,
                init_params[:len(self.base_strategies)]
            )
        }

        cur_param_check = cur_params[:len(self.base_strategies)]
        cur_param_check = self.make_safe_params(cur_param_check)

        opt_f = self.executor_pass.current_par_lookup(
            self.base_strategies,
            cur_param_check
        )

        if self.pipeline_type == 'ml':
            self.set_ranges()

        seen = set()

        if opt_f < f_goal:
            self.rank_iter = 1
            self.rank_f = opt_f
            return

        seen.add(tuple(cur_params_opt.items()))

        found = False

        for iter_size in range(len(self.coef_rank) + 1):
            if self.rank_iter >= max_iterations:
                print(f"Stopping: reached maximum iterations = {max_iterations}")
                break

            if iter_size == 0:
                found, opt_f, cur_params_opt = self.optimistic_search(
                    seen,
                    cur_params_opt,
                    f_goal,
                    opt_f,
                    max_iterations=max_iterations
                )
            else:
                found, opt_f, cur_params_opt = self.exhaustive_search(
                    iter_size,
                    seen,
                    cur_params_opt,
                    f_goal,
                    opt_f,
                    max_iterations=max_iterations
                )

            if found:
                break

        if not found:
            self.fail += 1

    def optimistic_search(self, seen, cur_params_opt, f_goal, opt_f, max_iterations=1000):
        print('coeef', self.coef_rank)

        for val in self.coef_rank:
            if self.rank_iter >= max_iterations:
                print(f"Stopping optimistic search: reached maximum iterations = {max_iterations}")
                return False, opt_f, cur_params_opt

            if val < 0 or val >= len(self.base_strategies):
                print(
                    f"Warning: Skipping invalid coef_rank index {val}. "
                    f"Valid range is 0 to {len(self.base_strategies) - 1}."
                )
                continue

            cur_strategy = self.base_strategies[val]
            current_paramter_value = self.tau(cur_strategy, val)

            cur_params = cur_params_opt.copy()
            cur_params[cur_strategy] = current_paramter_value

            logging.info(f'Next param {cur_params}')

            if tuple(cur_params.items()) in seen:
                continue

            safe_params = self.make_safe_params(
                [int(v) for v in cur_params.values()]
            )

            cur_f = self.executor_pass.current_par_lookup(
                self.base_strategies,
                safe_params
            )

            seen.add(tuple(cur_params.items()))
            self.rank_iter += 1

            logging.info(f'utiltiy found : {cur_f} ,optimal utility : {opt_f}')

            opt_f, cur_params_opt, found = self.f_lookup(
                cur_f,
                f_goal,
                cur_params_opt,
                cur_params,
                opt_f
            )

            logging.info(f'Optimal paramater {cur_params_opt} ')

            if found:
                return True, opt_f, cur_params_opt

        return False, opt_f, cur_params_opt

    def exhaustive_search(self, comb_size, seen, cur_params_opt, f_goal, opt_f, max_iterations=1000):
        logging.info('Fall back')
        self.fail_with_fallback += 1

        comb_lst = self.score_lookup.identify_param(self.coef_rank, comb_size)

        for comb in comb_lst:
            if self.rank_iter >= max_iterations:
                print(f"Stopping exhaustive search: reached maximum iterations = {max_iterations}")
                return False, opt_f, cur_params_opt

            sorted_params = self.score_lookup.score_values(
                self.historical_data,
                self.coefs,
                comb_size,
                comb
            )

            for elem, score in sorted_params:
                if self.rank_iter >= max_iterations:
                    print(f"Stopping exhaustive search: reached maximum iterations = {max_iterations}")
                    return False, opt_f, cur_params_opt

                cur_params = cur_params_opt.copy()

                for j in range(comb_size):
                    idx = comb[j]

                    if idx < 0 or idx >= len(self.base_strategies):
                        print(
                            f"Warning: Skipping invalid comb index {idx}. "
                            f"Valid range is 0 to {len(self.base_strategies) - 1}."
                        )
                        continue

                    cur_strategy = self.base_strategies[idx]
                    cur_params[cur_strategy] = round(elem[idx], 5)

                if tuple(cur_params.items()) in seen:
                    continue

                seen.add(tuple(cur_params.items()))
                self.rank_iter += 1
                self.rank_f = opt_f

                logging.info(f'Next param {cur_params}')

                safe_params = self.make_safe_params(
                    [int(v) for v in cur_params.values()]
                )

                cur_f = self.executor_pass.current_par_lookup(
                    self.base_strategies,
                    safe_params
                )

                opt_f, cur_params_opt, found = self.f_lookup(
                    cur_f,
                    f_goal,
                    cur_params_opt,
                    cur_params,
                    opt_f
                )

                logging.info(
                    f'Optimal paramater {cur_params_opt}, optimal utility {opt_f} '
                )

                if found:
                    return True, opt_f, cur_params_opt

        return False, opt_f, cur_params_opt

    def f_lookup(self, cur_f, f_goal, cur_params_opt, cur_params, opt_f):
        found = False

        if self.pipeline_type == 'ml':
            if cur_f <= f_goal:
                self.rank_f = cur_f
                self.pass_ += 1
                found = True
            elif cur_f < opt_f:
                opt_f = cur_f
                cur_params_opt = cur_params

        return opt_f, cur_params_opt, found

    def identify_param(self, rank_list, comb_size):
        return list(itertools.combinations(rank_list, comb_size))

    def score_values(self, historical_data, coefs, comb_size, comb):
        i = 0
        coef_lst = []
        score = {}

        while i < comb_size:
            coef_lst.append(coefs[comb[i]])
            i += 1

        for param in historical_data:
            result = [param[i] for i in list(comb)]
            score[tuple(param)] = sum([x * y for x, y in zip(coef_lst, result)])

        sorted_params = sorted(score.items(), key=operator.itemgetter(1))

        return sorted_params

    def tau(self, cur_strategy, val):
        return self.ranges[cur_strategy][-1] if self.coefs[val] < 0 else self.ranges[cur_strategy][0]

    def write_quartiles(self, csv_writer, algorithm, metric, quartiles, f_goal, f_goals):
        if self.model_type != 'reg':
            base = round(1 - f_goal, 2)
        else:
            base = round(1 - (f_goal - min(f_goals)) / min(f_goals), 2)

        for i, q in enumerate(quartiles, 1):
            csv_writer.writerow([base, algorithm, f"{metric} q{i}", round(q, 5)])
