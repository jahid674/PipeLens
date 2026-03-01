import logging
import pandas as pd
import numpy as np
from pipeline_execution import PipelineExecutor
from score_lookup import ScoreLookup
import itertools
import operator


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

        self.executor_pass = PipelineExecutor(pipeline_type=self.pipeline_type,
                                         dataset_name=self.dataset_name,
                                         metric_type=self.metric_type,
                                         pipeline_ord=self.pipeline_order)
        
        self.pasing_hist_data = pd.read_csv(self.filename_train)
        self.coefs, self.coef_rank = self.executor_pass.score_parameter(self.pasing_hist_data)

        self.score_lookup = ScoreLookup(pipeline_order, metric_type)

    def set_ranges(self):
        for strategy in self.pipeline_order:
            self.ranges[strategy] = list(np.unique(self.pasing_hist_data[strategy]))

    def optimize(self, init_params, f_goal):
        self.rank_iter = 0
        self.rank_f = -1
        cur_params = init_params.copy()
        cur_params_opt = {strategy: selection for strategy, selection in zip(self.base_strategies, init_params[:len(self.base_strategies)])}
        cur_param_check=cur_params[:len(self.base_strategies)]
        opt_f = self.executor_pass.current_par_lookup(self.base_strategies, cur_param_check)
        #opt_f = self.executor_pass.current_par_lookup(cur_param_check)
        #print(cur_params)
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
            if iter_size == 0:
                found, opt_f, cur_params_opt = self.optimistic_search(seen, cur_params_opt, f_goal, opt_f)
            else:
                found, opt_f, cur_params_opt = self.exhaustive_search(iter_size, seen, cur_params_opt, f_goal, opt_f)
            if found:
                break
        if not found:
            self.fail += 1

    def optimistic_search(self, seen, cur_params_opt, f_goal, opt_f):
        print('coeef',self.coef_rank)
        for val in self.coef_rank:
            cur_strategy = self.base_strategies[val]
            current_paramter_value = self.tau(cur_strategy, val)
            cur_params = cur_params_opt.copy()
            cur_params[cur_strategy] = current_paramter_value
            logging.info(f'Next param {cur_params}')
            if tuple(cur_params.items()) in seen:
                continue
            #cur_f = self.executor_pass.current_par_lookup(cur_params)
            cur_f = self.executor_pass.current_par_lookup(self.base_strategies, [int(v) for v in cur_params.values()])
            seen.add(tuple(cur_params.items()))
            self.rank_iter += 1
            logging.info(f'utiltiy found : {cur_f} ,optimal utility : {opt_f}')
            opt_f, cur_params_opt, found = self.f_lookup(cur_f, f_goal, cur_params_opt, cur_params, opt_f)
            logging.info(f'Optimal paramater {cur_params_opt} ')
            if found:
                return True, opt_f, cur_params_opt
        return False, opt_f, cur_params_opt

    def exhaustive_search(self, comb_size, seen, cur_params_opt, f_goal, opt_f):
        logging.info('Fall back')
        self.fail_with_fallback += 1
        comb_lst = self.score_lookup.identify_param(self.coef_rank, comb_size)
        for comb in comb_lst:
            sorted_params = self.score_lookup.score_values(self.historical_data, self.coefs, comb_size, comb)
            for elem, score in sorted_params:
                cur_params = cur_params_opt.copy()
                for j in range(comb_size):
                    cur_strategy = self.base_strategies[comb[j]]
                    cur_params[cur_strategy] = round(elem[comb[j]], 5)

                if tuple(cur_params.items()) in seen:
                    continue

                seen.add(tuple(cur_params.items()))
                self.rank_iter += 1
                self.rank_f = opt_f
                logging.info(f'Next param {cur_params}')
                #cur_f = self.executor_pass.current_par_lookup(cur_params)
                cur_f = self.executor_pass.current_par_lookup(self.base_strategies, [int(v) for v in cur_params.values()])
                opt_f, cur_params_opt, found = self.f_lookup(cur_f, f_goal, cur_params_opt, cur_params, opt_f)
                logging.info(f'Optimal paramater {cur_params_opt}, optimal utility {opt_f} ')
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
            i=0
            coef_lst=[]
            score = {}
            while i<comb_size:
                    coef_lst.append(coefs[comb[i]])
                    i+=1
                                        
            for param in historical_data:
                    result = [param[i] for i in list(comb)]
                    score[tuple(param)] = sum([x*y for x,y in zip( coef_lst,result)])
            sorted_params = sorted(score.items(), key=operator.itemgetter(1))

            return sorted_params
    
    def tau(self, cur_strategy, val):
        return self.ranges[cur_strategy][-1] if self.coefs[val] < 0 else self.ranges[cur_strategy][0]
    
    #for both algorithm would be same
    def write_quartiles(self, csv_writer, algorithm, metric, quartiles, f_goal, f_goals):
        if self.model_type != 'reg':
            base = round(1 - f_goal, 2)
        else:
            base = round(1 - (f_goal - min(f_goals)) / min(f_goals), 2)
        for i, q in enumerate(quartiles, 1):
            csv_writer.writerow([base, algorithm, f"{metric} q{i}", round(q, 5)])

    
         
