import random
from score_lookup import ScoreLookup
from pipeline_execution import PipelineExecutor

class GridSearch:
    def __init__(self, historical_data, pipeline_order, metric_type):
        self.historical_data_pd = historical_data
        self.pipeline_order = pipeline_order
        self.metric_type = metric_type
        self.historical_data = historical_data.values.tolist()
        self.score_lookup = ScoreLookup(pipeline_order, metric_type)
        self.gs_idistr = []
        self.gs_fdistr = []
        self.executor_pass = PipelineExecutor(pipeline_type=self.pipeline_type,
                                         dataset_name=self.dataset_name,
                                         metric_type=self.metric_type,
                                         pipeline_ord=self.pipeline_order)

    def grid_search(self, f_goal, seen):
        #self.gs_idistr.clear()
        #self.gs_fdistr.clear()
        gs_iter = 0
        gs_f = 0

        cur_order = self.historical_data.copy()
        random.shuffle(cur_order)

        for elem in cur_order:
            cur_params = {strategy: selection for strategy, selection in zip(self.pipeline_order, elem[:len(self.pipeline_order)])}
            param_tuple = tuple(cur_params.items())
            if param_tuple in seen:
                continue

            seen.add(param_tuple)
            cur_f  = self.f_score_look_up2(self.historical_data_pd,list(cur_params.values()))
            #cur_f = self.executor_pass.current_par_lookup(cur_params)
            gs_iter += 1
            
            if cur_f <= f_goal:
                gs_f = cur_f
                self.gs_fdistr.append(gs_f)
                self.gs_idistr.append(gs_iter)
                return gs_iter, gs_f
