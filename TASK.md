Overview



Goal: turn a large real-world codebase into an unsaturated evaluation benchmark for coding agents and compare models.



You may use any tools or models to design the environment, tasks, and evaluators.



Task requirements

Repository
Choose a public GitHub repository with ≥ 1,000 merged PRs.
Evaluation environment
Convert the repository into one or more evaluation environments (e.g., Docker-based).
The environment must support automatic evaluation of model-generated code changes.
Tasks
Design 100 evaluation questions/tasks.
Tasks may be based on PRs, issues, tests, code structure, docs, configs, or anything else.
Each task defines a starting state and a prompt.
Scoring
Implement an automatic scoring procedure that maps a solution to a score in [0, 1].
Benchmark
Use mini-swe-agent to evaluate ≥ 3 models/configurations on the tasks.
Report
Briefly describe the environment, task design, scoring method, results, shortcomings, and how you would improve or scale the benchmark.

Submit only:

The evaluation environment(s)
Scripts/procedures for task generation, evaluation, scoring, and benchmarking
A short written report
Multiple scripts are allowed.


Upload everything below and if its over 10MB, please share a google drive link. 
