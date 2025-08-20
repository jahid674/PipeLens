import logging
import pandas as pd
import numpy as np
import itertools
import operator

from pipeline_execution import PipelineExecutor
from score_lookup import ScoreLookup

class GlassBoxOptimizer:
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
        #self.coefs_profile, self.profile_ranking, self.param_coeff, self.param_rank = self.executor_pass.rank_profile_parameter(self.filename_train)
        self.profiles = self.executor_pass.get_header(self.filename_train)
        #print(self.profiles)
        self.score_lookup = ScoreLookup(pipeline_order, metric_type)
    
    def set_ranges(self):
        for strategy in self.pipeline_order:
            self.ranges[strategy] = list(np.unique(self.pasing_hist_data[strategy]))

    def optimize(self, init_params, f_goal):
        self.rank_iter = 0
        self.rank_f = 0
        cur_params = init_params.copy()
        cur_params_opt = {strategy: selection for strategy, selection in zip(self.base_strategies, init_params)}
        cur_param_check=cur_params[:len(self.base_strategies)]
        opt_f = self.score_lookup.utility_look_up(self.historical_data_pd, init_params)
        logging.info(f'Evaluating {[int(v) for v in cur_params_opt.values()]}--Initial Utiltiy {opt_f}--Target Utility {f_goal}')
        #opt_f = self.executor_pass.current_par_lookup(cur_param_check)

        if self.pipeline_type == 'ml':
            self.set_ranges()

        seen = set()
        if opt_f < f_goal:
            self.rank_iter = 1
            self.rank_f = opt_f
            return

        #seen.add(tuple(cur_params_opt.items()))
        found = False
        for iter_size in range(3+ 1): #Changed it
            if iter_size == 0:
                logging.info(f'first loop iter_size = {iter_size}')
                found, opt_f, cur_params_opt = self.optimistic_search(seen, cur_params_opt, f_goal, opt_f)
            '''else:
                logging.info(f'second loop iter_size = {iter_size}')
                found, opt_f, cur_params_opt = self.exhaustive_search(iter_size, seen, cur_params_opt, f_goal, opt_f)'''
            if found:
                break
        if not found:
            self.fail += 1

    '''def optimistic_search(self, seen, cur_params_opt, f_goal, opt_f):
        for profile_index in self.profile_ranking:
            logging.info(f'first loop profile = {self.profiles[profile_index]}')
            #need correction here
            #distinguish between profile and p
            profile_name = self.profiles[profile_index]
            coef_rank = self.param_rank[profile_name]
            logging.info(f'Current Profile name {profile_name} : corresponding ranking parameter: {coef_rank}')
            for val in coef_rank:
                cur_strategy = self.base_strategies[val]
                logging.info(f'Strategy selected  : {cur_strategy}')
                current_param_value=self.rank_individual_intervention(cur_strategy, profile_index, profile_name, val)
                logging.info(f'order of parameter  value: {current_param_value}')                
                logging.warn(f'current Iteration before parameter selection {self.rank_iter}')
                cur_params = cur_params_opt.copy()
                cur_params[cur_strategy] = current_param_value
                logging.info(f'next parameter {cur_params}, optimal parameter found {cur_params_opt}')

                if tuple(cur_params.items()) in seen:
                    continue
                seen.add(tuple(cur_params.items()))
                cur_f  = self.score_lookup.utility_look_up(self.historical_data_pd,list(cur_params.values()))
                #cur_f = self.executor_pass.current_par_lookup(cur_params)

                self.rank_iter += 1
                logging.info(f'next parameter {cur_params}, optimal parameter found {cur_params_opt}')
                logging.warn(f'Current iteration after parameter selection {self.rank_iter}')
                logging.info(f'updated utility after parameter selection {cur_f}')

                if cur_f <= f_goal:
                    self.rank_f = cur_f
                    self.pass_ += 1
                    logging.error("Target achieved")
                    return True, cur_f, cur_params
                elif cur_f < opt_f:
                    opt_f = cur_f
                    cur_params_opt = cur_params.copy()
        return False, opt_f, cur_params_opt'''
    
    def optimistic_search(self, seen, cur_params_opt, f_goal, opt_f):
        optimal_position=2
        print("[INFO] Running optimistic search with combined interventions...")
        ranked_interventions = self.executor_pass.evaluate_combined_intervention([int(v) for v in cur_params_opt.values()], self.filename_train, new_components=['outlier', 'whitespace', 'punctuation', 'stopword'])
        original_order = self.pipeline_order.copy()
        print('original order', original_order)
        for component, strategy, similarity, uti in ranked_interventions:
            logging.info(f"[SEARCH] Trying intervention: {component} → {strategy}")

            if component in original_order:
                idx = original_order.index(component)
                cur_params = cur_params_opt.copy()
                cur_params[original_order[idx]] = strategy
                intervened_order = original_order.copy()
                print('intervened order', intervened_order)
                print('current_param',cur_params)

            else:
                # New component → insert at optimal_position
                if optimal_position > len(cur_params_opt):
                    logging.warning(f"Skipping {component}: optimal_position {optimal_position} out of range.")
                    continue
                intervened_order = original_order[:optimal_position] + [component] + original_order[optimal_position:]
                cur_params_items = list(cur_params_opt.items())
                cur_params_items = cur_params_items[:optimal_position] + [(component, strategy)] + cur_params_items[optimal_position:]
                cur_params = dict(cur_params_items)

            if len(cur_params) != len(intervened_order):
                logging.warning(f"[SKIP] Misaligned parameter length for {component}")
                continue

            if tuple(cur_params) in seen:
                continue
            #seen.add(tuple(cur_params))
            prev_order = self.pipeline_order

            
            cur_f = uti
            self.rank_iter += 1
            logging.info(f"[TRY] {component}={strategy},pipeline: {intervened_order}, Utility={cur_f:.4f}, Best={opt_f:.4f}")

            if cur_f <= f_goal:
                logging.info("✅ Target achieved 🎯")
                self.rank_f = cur_f
                self.pass_ += 1
                return True, cur_f, cur_params
            elif cur_f < opt_f:
                opt_f = cur_f
                print('cur_F',cur_f)
                #cur_params_opt = cur_params.copy()
                cur_params_opt = cur_params.copy()
                original_order = intervened_order
                seen.add(tuple(cur_params.items()))


        return False, opt_f, cur_params_opt


    def exhaustive_search(self, comb_size, seen, cur_params_opt, f_goal, opt_f):
        self.fail_with_fallback += 1
        for profile_index in self.profile_ranking:
            profile_name = self.profiles[profile_index]
            coeff=self.coefs_profile[profile_index]<0
            
            logging.info(f'profile = {profile_name}')
            sorted_params, sorted_params_lst = self.rank_intervention_combination(self,profile_name, profile_index)
            if (coeff):
                sorted_params.reverse()
                sorted_params_lst.reverse()
            
            cur_params = cur_params_opt.copy()
            for id,val in enumerate(self.base_strategies):
                cur_params[val] = sorted_params_lst[comb_size-1][1][id]


            if tuple(cur_params.items()) in seen:
                continue
            seen.add(tuple(cur_params.items()))

            cur_f  = self.score_lookup.utility_look_up(self.historical_data_pd,list(cur_params.values()))
            #cur_f = self.executor_pass.current_par_lookup(cur_params)
            self.rank_iter += 1

            if cur_f <= f_goal:
                self.rank_f = cur_f
                self.pass_ += 1
                return True, cur_f, cur_params
            elif cur_f < opt_f:
                opt_f = cur_f
                cur_params_opt = cur_params.copy()
                
        return False, opt_f, cur_params_opt
    
    def rank_individual_intervention(self, cur_strategy, idx, prof_name, val):
        if self.coefs_profile[idx] > 0:
            current_param_value = self.ranges[cur_strategy][0] if self.param_coeff[prof_name][val] > 0 else self.ranges[cur_strategy][-1]
        else:
            current_param_value = self.ranges[cur_strategy][0] if self.param_coeff[prof_name][val] < 0 else self.ranges[cur_strategy][-1]
        
        return current_param_value
    
    def similarity_based_ranking(self, new_comp, cur_par):
        combined_rank=self.executor_pass.evaluate_combined_intervention(new_comp, cur_par)
        return combined_rank


    
    def rank_intervention_combination(self,profile_name):
        score_map = {}
        lst = []
        for row in self.historical_data:
            score_val = sum([row[i] * self.param_coeff[profile_name][i] for i in range(len(self.base_strategies))])
            lst.append(score_val, row)
            score_map[score_val] = row

        sorted_param = sorted(score_map.items(), key=operator.itemgetter(0))
        sorted_params_lst = sorted(lst, key=lambda x: x[0])

        return sorted_param,sorted_params_lst

    def write_quartiles(self, csv_writer, algorithm, metric, quartiles, f_goal, f_goals):
        if self.model_type != 'reg':
            base = round(1 - f_goal, 2)
        else:
            base = round(1 - (f_goal - min(f_goals)) / min(f_goals), 2)
        for i, q in enumerate(quartiles, 1):
            csv_writer.writerow([base, algorithm, f"{metric} q{i}", round(q, 5)])



