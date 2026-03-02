PipeLens is a framework for debugging and optimizing data science pipelines through causal interventions. Modern data science systems often fail not because of code bugs, but due to:

Mismatch between data properties and module assumptions
Improper pipeline configuration
Poor module orderin
Suboptimal parameter choices
Data drift or distribution shifts

PipeLens addresses these issues by:
Modeling pipelines as Directed Acyclic Graphs (DAGs)
Leveraging data profiles
Using historical executions
Performing causally guided structural and parameter interventions

Unlike traditional hyperparameter tuning or AutoML, PipeLens intervenes on an existing failing pipeline to identify minimal fixes that restore utility.

🚀 Key Features
✅ Works for ML, Entity Matching, and Regression pipelines
✅ Supports structural interventions (insert/delete/swap modules)
✅ Supports parameter interventions
✅ Two modes:

Glass-box (access to intermediate outputs)
Opaque-box (only final utility visible)
✅ Uses historical runs to learn a proxy utility model
✅ Prioritizes minimal, causally effective interventions



Code Overview:
For easy implementation: Use the config.json to declare your dataset, utility goal, and other selections. Additionally provide the historical data and metric path. Then go to result.py and choose the basleine methods and run it. You will see the performance at metric path, you can also log the internal details from result.py

Key file:
i) pipeline_execution.py(Includes Executioon of historical dataset, perturbation, ranking strategy)
ii) glassbox_optimizer.py(provides the glass box algorithm--ALGORITHM 2)
iii)opaquebox_optimizer.py(provides opaquebox algorithm--ALGORITHM 1)
iv)modules(for global module catalog)
