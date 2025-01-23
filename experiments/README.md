

Benchmarks
==

By default, all relevant benchmarks are in the benchmarks folder, incase you want to re-generate them, we describe the process.

To generate benchmarks from the existing data, assuming you are in the rtsm folder and that it is installed, you can run the following two scripts:

```bash
./experiments/gen_aslib_benchmarks.sh
./experiments/gen_resist_benchmarks.sh
```


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

Executing these commands will produce all the necessary data, by default each task has a timeout of 3600s but afterwards it needs, let us say 1min or 2min to check the solution and capture the data which is then saved.

Now, what we actually want is exploitable data, in order to merge all these data together we can run:

```bash
# requires nothing
./experiments/aggregate_data_into_csv.sh
```
