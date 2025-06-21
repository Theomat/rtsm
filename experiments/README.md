
Solving
==

First, you need to generate all sub instances by running:

```bash
# requires python and rtsm installed
./experiments/generate_all_sub_instances.sh
```

Now the list of commands to be run to run all the experiments can be obtained by using:

```bash
# requires nothing but the generated commands require python and rtsm installed
./experiments/generate_all_solve_commands.sh
```

Executing these commands will produce all the necessary data, by default each task has a timeout of 3000s but afterwards it needs, let us say 1min or 2min to check the solution and capture the data which is then saved.

Now, what we actually want is exploitable data, in order to exploit the data, we need to run:

```bash
# requires nothing
python experiments/data_aggregator.py results csv
python experiments/analysis/stats_tests.py csv
```

Then you can use all scripts under the folder analysis, they do not require arguments.
