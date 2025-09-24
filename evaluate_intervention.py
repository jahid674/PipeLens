'''def evaluate_interventions(self, cur_par, filename_training, new_components):
        filename_training = f'historical_data/historical_data_train_profile_{model_type}_{metric_type}_{dataset_name}.csv'
        original_order = self.pipeline_order
        insertion_positions = list(range(len(original_order)))
        _, self.rank_profile = self.rank_profile_new_comp(filename_training, ['outlier'])

        best_sim = -float('inf')
        best_component = None
        best_insert_pos = None
        best_result = None
        best_utility = None
        global_ranking = []


        def _eval_config(eval_order, eval_params, tag_component, tag_strategy, position=None):
            X_test = self.X_test.copy()
            y_test = self.y_test.copy()

            if self.pipeline_type == 'ml':
                X_test = self.noise_injector.inject_noise(X_test, noise_type='outlier', frac=self.tau)
                X_test, y_test = self.noise_injector.inject_noise(X_test, y_test, noise_type='class_imbalance')
                _, _, sensitive = self.getIdxSensitive(X_test, self.sensitive_var)
                X, y, sens = X_test.copy(), y_test.copy(), sensitive.copy()
                X = self.noise_injector.inject_noise(X, noise_type='missing', frac=self.tau)
            else:
                X, y, sens = X_test.copy(), y_test.copy(), None

            param_record, frac_data = [], []
            self.frac_header = []
            p = Profile()
            numerical_columns = X.select_dtypes(include=['int', 'float']).columns
            utility = None
            fraction_outlier = None
            last_handler = None
            first_component = eval_order[0]
            if first_component.lower() != 'missing':
                X = X.dropna(axis=1)

            for i, step in enumerate(eval_order):
                param_index = self._safe_param_index(step, int(eval_params[i]))
                handler = self._load_handler(step, param_index)
                last_handler = handler
                X, y, sens, util_tmp, fraction_outlier, frac_header, frac_value = self._apply_step(handler, X, y, sens)
                if frac_header is not None:
                    frac_data.append(frac_value)
                    self.frac_header.append(frac_header)
                if util_tmp is not None:
                    utility = util_tmp
                param_record.append(param_index + 1)

            self.headers, sens_data = last_handler.get_profile_metric(y, sens)
            prof_data = frac_data + sens_data
            profile_gen, key_profile = p.populate_profiles(
                pd.concat([X, y], axis=1),
                numerical_columns,
                self.target_variable_name,
                fraction_outlier,
                self.metric_type
            )

            out_cols = eval_order
            row = param_record + prof_data + profile_gen + [utility]
            col_headers = out_cols + self.frac_header + self.headers + key_profile + [f'utility_{self.metric_type}']

            df = pd.DataFrame([row], columns=col_headers)
            test_file = f'historical_data/insertion/test_row_profile.csv'
            df.to_csv(test_file, index=False)

            sim = self.profile_similarity_df(filename_training, test_file, cur_par, self.rank_profile, metric='cosine')
            logging.info(f'Intervention: component={tag_component}, strategy={tag_strategy}, similarity={sim}, utility={utility}')

            if sim is not None:
                global_ranking.append((tag_component, tag_strategy, sim, utility, position))

            return sim, utility

        # (A) try alternate strategies for existing steps
        for i, component in enumerate(original_order):
            num_strategies = self.strategy_counts[component]
            current_strategy = cur_par[i]
            for new_strategy in range(1, num_strategies + 1):
                if new_strategy == current_strategy:
                    continue
                new_cur_par = cur_par[:i] + [new_strategy] + cur_par[i + 1:]
                _eval_config(original_order, new_cur_par, component, new_strategy)

        # (B) try inserting new components at positions
        for comp in new_components:
            comp_ranges = self.strategy_counts[comp]
            for strat_idx in range(comp_ranges):
                for insert_pos in insertion_positions:
                    new_order = original_order[:insert_pos] + [comp] + original_order[insert_pos:]
                    new_cur_par = cur_par[:insert_pos] + [strat_idx + 1] + cur_par[insert_pos:]
                    sim, utility = _eval_config(new_order, new_cur_par, comp, strat_idx + 1, insert_pos)
                    if sim is not None and sim > best_sim:
                        best_component = comp
                        best_result = new_cur_par
                        best_sim = sim
                        best_utility = utility
                        best_insert_pos = insert_pos

        global_ranking.sort(key=lambda x: x[2], reverse=True)
        for idx, (component, strategy, sim, utility, pos) in enumerate(global_ranking):
            if pos is not None:
                print(f"New component: {component} @ {pos}, -- Strategy -> {strategy}, Similarity={sim:.4f}, Utility={utility:.4f}")
            else:
                print(f"Existing component: {component}, -- Strategy -> {strategy}, Similarity={sim:.4f}, Utility={utility:.4f}")
        return global_ranking
    
    def evaluate_interventions1(self, cur_par, filename_training, new_components):
        import time, logging   # <-- add time (and logging if not already imported)
        passing_par = self.get_passing_pipeline(filename_training, f'utility_{self.metric_type}', threshold_val=182)
        print("Passing pipeline parameters:", passing_par)
        t0 = time.perf_counter()   # <-- start timer
        try:
            # ====== your existing body starts here ======
            original_order = self.pipeline_order
            insertion_positions = list(range(len(original_order)))
            #_, self.rank_profile = self.rank_profile_new_comp(filename_training, ['outlier'])
            prof_cols = self.get_header(filename_training)

            best_sim = -float('inf')
            best_component = None
            best_insert_pos = None
            best_result = None
            best_utility = None
            global_ranking = []


            def _eval_config(eval_order, eval_params, tag_component, tag_strategy, position=None):
                X_test = self.X_test.copy()
                y_test = self.y_test.copy()

                if self.pipeline_type == 'ml':
                    X_test = self.noise_injector.inject_noise(X_test, noise_type='outlier', frac=.1)
                    X_test, y_test = self.noise_injector.inject_noise(X_test, y_test, noise_type='class_imbalance')
                    _, _, sensitive = self.getIdxSensitive(X_test, self.sensitive_var)
                    X, y, sens = X_test.copy(), y_test.copy(), sensitive.copy()
                    X = self.noise_injector.inject_noise(X, noise_type='missing', frac=self.tau)
                else:
                    X, y, sens = X_test.copy(), y_test.copy(), None

                param_record, frac_data = [], []
                self.frac_header = []
                p = Profile()
                numerical_columns = X.select_dtypes(include=['int', 'float']).columns
                utility = None
                fraction_outlier = None
                last_handler = None

                for i, step in enumerate(eval_order):
                    param_index = self._safe_param_index(step, int(eval_params[i]))
                    handler = self._load_handler(step, param_index)
                    last_handler = handler
                    X, y, sens, util_tmp, fraction_outlier, frac_header, frac_value = self._apply_step(handler, X, y, sens)
                    if frac_header is not None:
                        frac_data.append(frac_value)
                        self.frac_header.append(frac_header)
                    if util_tmp is not None:
                        utility = util_tmp
                    param_record.append(param_index + 1)

                self.headers, sens_data = last_handler.get_profile_metric(y, sens)
                prof_data = frac_data + sens_data
                profile_gen, key_profile = p.populate_profiles(
                    pd.concat([X, y], axis=1),
                    numerical_columns,
                    self.target_variable_name,
                    fraction_outlier,
                    self.metric_type
                )

                out_cols = eval_order
                row = param_record + prof_data + profile_gen + [utility]
                col_headers = out_cols + self.frac_header + self.headers + key_profile + [f'utility_{self.metric_type}']

                df = pd.DataFrame([row], columns=col_headers)
                test_file = f'historical_data/insertion/test_row_profile.csv'
                df.to_csv(test_file, index=False)

                sim = self.profile_similarity_df(filename_training, test_file, passing_par, prof_cols, metric='cosine')
                #logging.info(f'Intervention: component={tag_component}, strategy={tag_strategy}, position={position}, similarity={sim}, utility={utility}')
                return sim, utility

            # (A) try alternate strategies for existing steps (append directly)
            for i, component in enumerate(original_order):
                num_strategies = self.strategy_counts[component]
                current_strategy = cur_par[i]
                for new_strategy in range(1, num_strategies + 1):
                    if new_strategy == current_strategy:
                        continue
                    new_cur_par = cur_par[:i] + [new_strategy] + cur_par[i + 1:]
                    sim, utility = _eval_config(original_order, new_cur_par, component, new_strategy, position=None)
                    if sim is not None:
                        global_ranking.append((component, new_strategy, sim, utility, None))
                        if sim > best_sim:
                            best_component = component
                            best_result = new_cur_par
                            best_sim = sim
                            best_utility = utility
                            best_insert_pos = None

            # (B) try inserting new components — ONLY keep best insertion position per (component, strategy)
            for comp in new_components:
                comp_ranges = self.strategy_counts[comp]
                for strat_idx in range(comp_ranges):
                    best_for_this_strat = None  # (sim, utility, pos)
                    for insert_pos in insertion_positions:
                        new_order = original_order[:insert_pos] + [comp] + original_order[insert_pos:]
                        new_cur_par = cur_par[:insert_pos] + [strat_idx + 1] + cur_par[insert_pos:]
                        sim, utility = _eval_config(new_order, new_cur_par, comp, strat_idx + 1, position=insert_pos)

                        if sim is None:
                            continue

                        # Track best over positions for this (comp, strategy)
                        if (best_for_this_strat is None) or (sim > best_for_this_strat[0]):
                            best_for_this_strat = (sim, utility, insert_pos)

                        if sim > best_sim:
                            best_component = comp
                            best_result = new_cur_par
                            best_sim = sim
                            best_utility = utility
                            best_insert_pos = insert_pos

                    if best_for_this_strat is not None:
                        sim_star, util_star, pos_star = best_for_this_strat
                        global_ranking.append((comp, strat_idx + 1, sim_star, util_star, pos_star))

            global_ranking.sort(key=lambda x: x[2], reverse=True)

            # Pretty print
            for idx, (component, strategy, sim, utility, pos) in enumerate(global_ranking, start=1):
                if pos is not None:
                    print(f"New component: {component} @ {pos}, Strategy {strategy}, similarity={sim}")
                    #logging.info(f"New component: {component} @ {pos}, Strategy {strategy}, utility={utility:.4f}")
                else:
                    print(f"Existing component: {component}, Strategy {strategy}, similarity={sim}")
                    #logging.info(f"Existing component: {component}, Strategy {strategy}, utility={utility:.4f}")    

            return global_ranking

            S = np.array([t[2] for t in global_ranking], dtype=float) 
            U = np.array([t[3] for t in global_ranking], dtype=float)

            fused = self.fused_geometric_mean(S, U, wS=1.0, wU=1.0)
            global_ranking = [(*t, float(f)) for t, f in zip(global_ranking, fused)]
            global_ranking.sort(key=lambda x: x[5], reverse=True)
            for idx, (component, strategy, sim, utility, pos, fused_score) in enumerate(global_ranking, start=1):
                if pos is not None:
                    print(f"[{idx}] New component: {component} @ {pos} | Strategy {strategy} | "
                        f"Sim={sim:.4f} | Utility={utility:.4f} | Fused={fused_score:.4f}")
                else:
                    print(f"[{idx}] Existing component: {component} | Strategy {strategy} | "
                    f"Sim={sim:.4f} | Utility={utility:.4f} | Fused={fused_score:.4f}")

            return global_ranking
        
        finally:
            elapsed = time.perf_counter() - t0
            print(f"[evaluate_interventions1] runtime: {elapsed:.3f} s")
            try:
                logging.info(f"evaluate_interventions1_runtime_seconds={elapsed:.6f}")
            except Exception:
                pass


    
    def evaluate_interventions_ba(self, cur_par, filename_training, new_components):
        """
        Evaluate two kinds of interventions:
        (A) Parameter changes for existing steps  -> single similarity (full pipeline).
        (B) New component insertions              -> BEFORE and AFTER similarity (around the inserted step only).

        Returns
        -------
        param_change_ranking : list[tuple]
            (component, strategy, similarity, utility) sorted by similarity desc.
        insertion_before_ranking : list[tuple]
            (component, strategy, insert_pos, similarity_before) sorted by similarity desc.
        insertion_after_ranking : list[tuple]
            (component, strategy, insert_pos, similarity_after) sorted by similarity desc.
        """
        import os
        import logging
        import pandas as pd
        import numpy as np

        # Use your standard training file convention (keep behavior)
        filename_training = f'historical_data/historical_data_train_profile_{model_type}_{metric_type}_{dataset_name}.csv'
        original_order = self.pipeline_order
        insertion_positions = list(range(len(original_order)))  # keep your original rule

        # Learn profile ranking used for similarity and capture training header set
        _, self.rank_profile = self.rank_profile_new_comp(filename_training, ['outlier'])
        train_headers = self.get_header(filename_training)  # includes sens/profile cols present in training

        out_dir = 'historical_data/insertion'
        os.makedirs(out_dir, exist_ok=True)

        # -------------------- helpers --------------------

        def _safe_get_sens_metrics(handler, y, sens):
            """Return (sens_headers, sens_values) if the handler provides them; else empty."""
            if handler is not None and hasattr(handler, "get_profile_metric") and callable(handler.get_profile_metric):
                return handler.get_profile_metric(y, sens)
            return [], []

        def _align_to_training_schema(df_row: pd.DataFrame, eval_order_cols, frac_headers, sens_headers, key_profile):
            """
            Ensure the emitted row has the same shape/order as training:
            [pipeline params] + frac_headers + sens_headers + profile + utility.
            Adds any missing training headers with 0.0 and drops extras.
            """
            # Add any training headers missing from this snapshot (fill with 0.0)
            for c in train_headers:
                if c not in df_row.columns and c not in eval_order_cols and c not in frac_headers:
                    df_row[c] = 0.0

            # Build profile block in a stable order: sens + key_profile + (remaining train headers)
            profile_block = []
            for c in list(sens_headers) + list(key_profile):
                if c not in profile_block:
                    profile_block.append(c)
            for c in train_headers:
                if (c not in profile_block) and (c not in eval_order_cols) and (c not in frac_headers):
                    profile_block.append(c)

            final_cols = list(eval_order_cols) + list(frac_headers) + profile_block + [f'utility_{self.metric_type}']

            # Ensure all final columns exist
            for c in final_cols:
                if c not in df_row.columns:
                    df_row[c] = 0.0

            # Keep only final columns, in order; sanitize NaN/inf
            df_row = df_row[final_cols]
            df_row = df_row.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            return df_row

        def _emit_snapshot_row(X, y, sens, fraction_outlier,
                            eval_order_cols, param_record_vals,
                            frac_headers, frac_values,
                            sens_headers, sens_values,
                            key_profile, profile_values,
                            tag, utility=None):
            """
            Emit a single-row CSV snapshot with full schema (pipeline + frac + sens + profile + utility)
            and align it to training schema.
            """
            df = pd.DataFrame(
                [list(param_record_vals) + list(frac_values) + list(sens_values) + list(profile_values) + [utility]],
                columns=list(eval_order_cols) + list(frac_headers) + list(sens_headers) + list(key_profile) + [f'utility_{self.metric_type}']
            )
            df = _align_to_training_schema(df, eval_order_cols, frac_headers, sens_headers, key_profile)
            path = os.path.join(out_dir, f'{tag}.csv')
            df.to_csv(path, index=False)
            return path

        # ---------- A) Parameter changes: full-pipeline single similarity (same behavior as original) ----------
        def _eval_config_full(eval_order, eval_params, tag_component, tag_strategy):
            X_test = self.X_test.copy()
            y_test = self.y_test.copy()

            if self.pipeline_type == 'ml':
                X_test = self.noise_injector.inject_noise(X_test, noise_type='outlier', frac=self.tau)
                X_test, y_test = self.noise_injector.inject_noise(X_test, y_test, noise_type='class_imbalance')
                _, _, sensitive = self.getIdxSensitive(X_test, self.sensitive_var)
                X, y, sens = X_test.copy(), y_test.copy(), sensitive.copy()
                X = self.noise_injector.inject_noise(X, noise_type='missing', frac=self.tau)
            else:
                X, y, sens = X_test.copy(), y_test.copy(), None

            param_record, frac_data = [], []
            self.frac_header = []
            p = Profile()
            numerical_columns = X.select_dtypes(include=['int', 'float']).columns
            utility = None
            fraction_outlier = None
            last_handler = None

            for i, step in enumerate(eval_order):
                param_index = self._safe_param_index(step, int(eval_params[i]))
                handler = self._load_handler(step, param_index)
                last_handler = handler
                X, y, sens, util_tmp, fraction_outlier, bef_hdr, bef_val = self._apply_step(handler, X, y, sens)
                if bef_hdr is not None:
                    self.frac_header.append(bef_hdr)
                    frac_data.append(bef_val)
                if util_tmp is not None:
                    utility = util_tmp
                param_record.append(param_index + 1)

            # sens metrics from last handler (safely)
            sens_headers, sens_values = _safe_get_sens_metrics(last_handler, y, sens)

            # profile block
            profile_values, key_profile = p.populate_profiles(
                pd.concat([X, y], axis=1),
                numerical_columns,
                self.target_variable_name,
                fraction_outlier,
                self.metric_type
            )

            # Emit a full-pipeline snapshot aligned to training
            test_file = _emit_snapshot_row(
                X, y, sens, fraction_outlier,
                eval_order_cols=eval_order,
                param_record_vals=param_record,
                frac_headers=self.frac_header,
                frac_values=frac_data,
                sens_headers=sens_headers,
                sens_values=sens_values,
                key_profile=key_profile,
                profile_values=profile_values,
                tag=f'full_{tag_component}_{tag_strategy}',
                utility=utility
            )

            sim = self.profile_similarity_df(filename_training, test_file, cur_par, self.rank_profile, metric='cosine')
            logging.info(f'[PARAM-CHANGE] component={tag_component}, strategy={tag_strategy}, similarity={sim}, utility={utility}')
            return sim, utility

        # ---------- B) Insertions: BEFORE/AFTER around inserted step only ----------
        def _profile_before_after(eval_order, eval_params, insert_pos, label):
            """
            Run steps up to the inserted component, snapshot BEFORE with sens/profile,
            apply inserted step once, snapshot AFTER with sens/profile. Only up to that point.
            """
            X_test = self.X_test.copy()
            y_test = self.y_test.copy()

            if self.pipeline_type == 'ml':
                X_test = self.noise_injector.inject_noise(X_test, noise_type='outlier', frac=self.tau)
                X_test, y_test = self.noise_injector.inject_noise(X_test, y_test, noise_type='class_imbalance')
                _, _, sensitive = self.getIdxSensitive(X_test, self.sensitive_var)
                X, y, sens = X_test.copy(), y_test.copy(), sensitive.copy()
                X = self.noise_injector.inject_noise(X, noise_type='missing', frac=self.tau)
            else:
                X, y, sens = X_test.copy(), y_test.copy(), None

            param_record, frac_data, frac_headers = [], [], []
            fraction_outlier = None
            last_handler = None
            p = Profile()
            numerical_columns = X.select_dtypes(include=['int', 'float']).columns

            # Run up to (not including) inserted component
            for j, step in enumerate(eval_order[:insert_pos]):
                param_index = self._safe_param_index(step, int(eval_params[j]))
                handler = self._load_handler(step, param_index)
                last_handler = handler
                X, y, sens, util_tmp, fraction_outlier, bef_hdr, bef_val = self._apply_step(handler, X, y, sens)
                if bef_hdr is not None:
                    frac_headers.append(bef_hdr)
                    frac_data.append(bef_val)
                param_record.append(param_index + 1)
            fraction_outlier = 0.13

            # BEFORE snapshot sens/profile (safe)
            sens_headers_bef, sens_values_bef = _safe_get_sens_metrics(last_handler, y, sens)
            profile_values_bef, key_profile = p.populate_profiles(
                pd.concat([X, y], axis=1),
                numerical_columns,
                self.target_variable_name,
                fraction_outlier,
                self.metric_type
            )

            pre_params_full = param_record + [int(v) for v in eval_params[len(param_record):]]
            before_file = _emit_snapshot_row(
                X, y, sens, fraction_outlier,
                eval_order_cols=eval_order,
                param_record_vals=pre_params_full,
                frac_headers=frac_headers,
                frac_values=frac_data,
                sens_headers=sens_headers_bef,
                sens_values=sens_values_bef,
                key_profile=key_profile,
                profile_values=profile_values_bef,
                tag=f'before_{label}',
                utility=None
            )

            # Apply inserted component once
            step = eval_order[insert_pos]
            param_index = self._safe_param_index(step, int(eval_params[insert_pos]))
            handler = self._load_handler(step, param_index)
            X, y, sens, util_tmp, fraction_outlier, bef_hdr, bef_val = self._apply_step(handler, X, y, sens)
            if bef_hdr is not None:
                frac_headers.append(bef_hdr)
                frac_data.append(bef_val)
            param_record.append(param_index + 1)

            # AFTER snapshot sens/profile (from inserted handler)
            fraction_outlier = 0.13
            sens_headers_aft, sens_values_aft = _safe_get_sens_metrics(handler, y, sens)
            profile_values_aft, _ = p.populate_profiles(
                pd.concat([X, y], axis=1),
                numerical_columns,
                self.target_variable_name,
                fraction_outlier,
                self.metric_type
            )

            post_params_full = param_record + [int(v) for v in eval_params[len(param_record):]]
            after_file = _emit_snapshot_row(
                X, y, sens, fraction_outlier,
                eval_order_cols=eval_order,
                param_record_vals=post_params_full,
                frac_headers=frac_headers,
                frac_values=frac_data,
                sens_headers=sens_headers_aft,
                sens_values=sens_values_aft,
                key_profile=key_profile,
                profile_values=profile_values_aft,
                tag=f'after_{label}',
                utility=util_tmp if isinstance(util_tmp, (float, int)) else None
            )

            return before_file, after_file

        # -------------------- collect rankings --------------------
        param_change_ranking = []      # (component, strategy, sim, utility)
        insertion_before_ranking = []  # (component, strategy, insert_pos, sim_before)
        insertion_after_ranking  = []  # (component, strategy, insert_pos, sim_after)

        # (A) PARAMETER CHANGES — full pipeline single similarity
        for i, component in enumerate(original_order):
            num_strategies = self.strategy_counts[component]
            current_strategy = cur_par[i]
            for new_strategy in range(1, num_strategies + 1):
                if new_strategy == current_strategy:
                    continue
                new_params = cur_par[:i] + [new_strategy] + cur_par[i + 1:]
                sim, utility = _eval_config_full(original_order, new_params, component, new_strategy)
                if sim is not None:
                    param_change_ranking.append((component, new_strategy, float(sim), utility))

        # (B) NEW COMPONENT INSERTIONS — BEFORE/AFTER around the inserted step
        for comp in new_components:
            comp_ranges = self.strategy_counts[comp]
            for strat_idx in range(comp_ranges):
                for insert_pos in insertion_positions:
                    new_order = original_order[:insert_pos] + [comp] + original_order[insert_pos:]
                    new_params = cur_par[:insert_pos] + [strat_idx + 1] + cur_par[insert_pos:]

                    label = f'{comp}_ins{insert_pos}_{strat_idx + 1}'
                    before_file, after_file = _profile_before_after(
                        eval_order=new_order,
                        eval_params=new_params,
                        insert_pos=insert_pos,
                        label=label
                    )
                    sim_before = self.profile_similarity_df(
                        filename_training, before_file, cur_par, self.rank_profile, metric='cosine'
                    )
                    sim_after = self.profile_similarity_df(
                        filename_training, after_file, cur_par, self.rank_profile, metric='cosine'
                    )

                    if sim_before is not None:
                        insertion_before_ranking.append((comp, strat_idx + 1, insert_pos, float(sim_before)))
                    if sim_after is not None:
                        insertion_after_ranking.append((comp, strat_idx + 1, insert_pos, float(sim_after)))

                    logging.info(f'[INSERT] comp={comp}@{insert_pos}, strat={strat_idx + 1}, '
                                f'sim_before={sim_before}, sim_after={sim_after}')

        # -------------------- sort & print --------------------
        param_change_ranking.sort(key=lambda x: x[2], reverse=True)
        insertion_before_ranking.sort(key=lambda x: x[3], reverse=True)
        insertion_after_ranking.sort(key=lambda x: x[3], reverse=True)

        print("\n=== Parameter-change ranking (by similarity, desc) ===")
        for comp, strat, sim, util in param_change_ranking:
            util_str = f"{util:.4f}" if isinstance(util, (float, int)) else "NA"
            print(f"{comp} -> {strat}, Similarity={sim:.4f}, Utility={util_str}")

        print("\n=== Insertion ranking: BEFORE (by similarity, desc) ===")
        for comp, strat, pos, sim in insertion_before_ranking:
            print(f"{comp}@{pos} -> {strat}, SimilarityBefore={sim:.4f}")

        print("\n=== Insertion ranking: AFTER (by similarity, desc) ===")
        for comp, strat, pos, sim in insertion_after_ranking:
            print(f"{comp}@{pos} -> {strat}, SimilarityAfter={sim:.4f}")

        return param_change_ranking, insertion_before_ranking, insertion_after_ranking




    
    def evaluate_interventions_predicted_utility(self, cur_par, filename_training, new_components):

        # ---- 1) Load & split ----
        df_train = pd.read_csv(filename_training)
        utility_col = f'utility_{self.metric_type}'
        if utility_col not in df_train.columns:
            raise ValueError(f"Training file must contain column '{utility_col}'")

        feature_cols = [c for c in df_train.columns if c != utility_col]
        X_all = df_train[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
        y_all = pd.to_numeric(df_train[utility_col], errors='coerce').fillna(0.0).values

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_all, y_all, test_size=0.20, random_state=42
        )

        # ---- 2) Scale + polynomial expansion ----
        base_scaler = StandardScaler(with_mean=True, with_std=True)
        X_tr_scaled = base_scaler.fit_transform(X_tr)

        poly_degree = 1
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False, interaction_only=False)
        X_tr_poly = poly.fit_transform(X_tr_scaled)

        # ---- 3) Regularized regression (ElasticNet with CV) ----
        # Handles correlated polynomial features better, reduces overfitting.
        enet = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],                 # try different L1/L2 mixes
            alphas=np.logspace(-4, 2, 30),            # alpha grid 1e-4 ... 1e2
            cv=5,                                     # 5-fold CV on the training fold
            max_iter=10000,
            n_jobs=None,                               # set to -1 if you want parallelism
            random_state=42
        )
        enet.fit(X_tr_poly, y_tr)

        # ---- 4) Hold-out evaluation ----
        X_val_scaled = base_scaler.transform(X_val)
        X_val_poly = poly.transform(X_val_scaled)
        y_val_pred = np.ravel(enet.predict(X_val_poly))

        r2_val = r2_score(y_val, y_val_pred)
        rmse_val = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))

        print(f"[Polynomial utility model | ElasticNetCV] Hold-out (20%)  R^2 = {r2_val:.4f} | RMSE = {rmse_val:.4f}")
        print(f"Chosen alpha = {enet.alpha_:.6g} | l1_ratio = {enet.l1_ratio_}")
        logging.info(
            "Hold-out (20%%): R2=%.6f RMSE=%.6f | alpha=%.6g l1_ratio=%s",
            r2_val, rmse_val, enet.alpha_, str(enet.l1_ratio_)
        )

        # ---- helpers (unchanged) ----
        def _build_feature_row(eval_order, eval_params):
            X_test = self.X_test.copy()
            y_test = self.y_test.copy()
            n_half = len(self.X_test) // 2
            X_test = self.X_test.iloc[:n_half].copy()
            y_test = self.y_test.iloc[:n_half].copy()




            if self.pipeline_type == 'ml':
                #X_test = self.noise_injector.inject_noise(X_test, noise_type='outlier', frac=self.tau)
                #X_test, y_test = self.noise_injector.inject_noise(X_test, y_test, noise_type='class_imbalance')
                _, _, sensitive = self.getIdxSensitive(X_test, self.sensitive_var)
                X, y, sens = X_test.copy(), y_test.copy(), sensitive.copy()
                X = self.noise_injector.inject_noise(X, noise_type='missing', frac=self.tau)
            else:
                X, y, sens = X_test.copy(), y_test.copy(), None

            p = Profile()
            numerical_columns = X.select_dtypes(include=['int', 'float']).columns

            param_record, frac_data = [], []
            self.frac_header = []
            utility = None
            fraction_outlier = None
            last_handler = None

            for i, step in enumerate(eval_order):
                param_index = self._safe_param_index(step, int(eval_params[i]))
                handler = self._load_handler(step, param_index)
                last_handler = handler
                X, y, sens, util_tmp, fraction_outlier, frac_header, frac_value = self._apply_step(handler, X, y, sens)
                if frac_header is not None:
                    frac_data.append(frac_value)
                    self.frac_header.append(frac_header)
                if util_tmp is not None:
                    utility = util_tmp
                param_record.append(param_index + 1)

            self.headers, sens_data = last_handler.get_profile_metric(y, sens)
            prof_data = frac_data + sens_data
            profile_gen, key_profile = p.populate_profiles(
                pd.concat([X, y], axis=1),
                numerical_columns,
                self.target_variable_name,
                fraction_outlier,
                self.metric_type
            )

            out_cols = eval_order
            row = param_record + prof_data + profile_gen + [utility]
            col_headers = out_cols + self.frac_header + self.headers + key_profile + [f'utility_{self.metric_type}']
            row_df = pd.DataFrame([row], columns=col_headers)
            return row_df, out_cols

        def _align_features(test_row_df):
            missing = [c for c in feature_cols if c not in test_row_df.columns]
            for c in missing:
                test_row_df[c] = 0.0
            extras = [c for c in test_row_df.columns if c not in feature_cols and c != utility_col]
            if extras:
                test_row_df = test_row_df.drop(columns=extras)
            test_row_df = test_row_df.reindex(columns=feature_cols)
            test_row_df = test_row_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
            return test_row_df

        def _to_poly_space(X_df):
            X_scaled_row = base_scaler.transform(X_df.values)
            X_poly_row = poly.transform(X_scaled_row)
            return X_poly_row

        # ---- 5) Score all interventions with the trained, regularized model ----
        original_order = self.pipeline_order
        insertion_positions = list(range(2, len(original_order)))   # keep your existing rule
        results = []  # (component, strategy, insert_pos or None, y_pred)

        # (A) Alternate strategies for existing components
        for i, component in enumerate(original_order):
            num_strategies = self.strategy_counts[component]
            current_strategy = cur_par[i]
            for new_strategy in range(1, num_strategies + 1):
                if new_strategy == current_strategy:
                    continue
                new_cur_par = cur_par[:i] + [new_strategy] + cur_par[i + 1:]
                test_row_df, _ = _build_feature_row(original_order, new_cur_par)
                X_test = _align_features(test_row_df.drop(columns=[utility_col], errors='ignore'))
                X_poly_row = _to_poly_space(X_test)
                y_pred = float(np.ravel(enet.predict(X_poly_row))[0])
                y_truth =test_row_df[utility_col].values[0] if utility_col in test_row_df.columns else None
                logging.info(f'[ALT] {component} -> {new_strategy}, predicted_utility={y_pred:.6f}')
                results.append((component, new_strategy, None, y_pred, y_truth))

        # (B) Insert new components
        for comp in new_components:
            comp_ranges = self.strategy_counts[comp]
            for strat_idx in range(comp_ranges):
                for insert_pos in insertion_positions:
                    new_order = original_order[:insert_pos] + [comp] + original_order[insert_pos:]
                    new_cur_par = cur_par[:insert_pos] + [strat_idx + 1] + cur_par[insert_pos:]
                    test_row_df, _ = _build_feature_row(new_order, new_cur_par)
                    X_test = _align_features(test_row_df.drop(columns=[utility_col], errors='ignore'))
                    X_poly_row = _to_poly_space(X_test)
                    y_pred = float(np.ravel(enet.predict(X_poly_row))[0])
                    y_truth =test_row_df[utility_col].values[0] if utility_col in test_row_df.columns else None
                    logging.info(f'[INS] {comp}@{insert_pos} -> {strat_idx + 1}, predicted_utility={y_pred:.6f}')
                    results.append((comp, strat_idx + 1, insert_pos, y_pred , y_truth))

        # ---- 6) Rank & print (lower predicted utility is better) ----
        # use y_pred (not abs) since utility is on an absolute scale where lower is better.
        results.sort(key=lambda x: x[3], reverse=False)
        for component, strategy, pos, y_pred, truth in results:
            if pos is None:
                print(f"{component} -> {strategy}, PredictedUtility={y_pred:.4f}, Truth={truth}")
            else:
                print(f"{component}@{pos} -> {strategy}, PredictedUtility={y_pred:.4f}, Truth={truth}")

        return results'''