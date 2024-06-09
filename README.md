# Ranked Test Suite Minimisation (RTSM)

![Overview of RTSM](./images/overview.png)

The goal is to minimise a test set while keeping the discriminating power of the test set.
Here a test is an instance on which performances can be measured for a variant.
A variant is an instance of a program, it can be different algorithms or different versiosn of the smae algorithms for example with different parameters.
Our tool takes as input a performance matrix of these variants on the set of tests.
Then we choose the prediction model to be used (can be none), using a prediction model reduces a bit Generalizability but enables a better minimisation.
In practice, using a linear model enables enormous gain at little cost.
Then our tool produces the subset of tests you need to keep the discriminative power of your tests.

In other words, it can be used to minimise benchmarks, to study variability of software, etc.
A lot of options are configurable.
Of course, this is not magic, this assumes that the next variants that are going to be tested are somehow in the same distribution as the variants used in order to minimise.

See the associated paper for the technical details.

<!-- toc -->

- [Installation](#installation)
- [Usage](#usage)
  - [Check your solution](#check-your-solution)
  - [Multiprocessing](#multiprocessing)
  - [Start from an existing solution](#start-from-an-existing-solution)
  - [Multi performance metrics](#multi-performance-metrics)
  - [Genetic Algorithm](#genetic-algorithm)
- [Input Format](#input-format)
  - [CSV](#csv)
- [Citing](#citing)

<!-- tocstop -->

## Installation

Install ``rtsm`` with your favorite tool by cloning this repository.

## Usage

The best way is to use the help flags but this section describes typical use cases.
A few examples are provided in the ``examples`` folder which we will use to describe the main usages of ``rtsm``.

Here is how you can start by simply running the following command:

```bash
python -m rtsm.crunch ./examples/humaneval_pass200.csv
```

Note that you can also use the same command without the ``crunch`` but the ``crunch`` tries to optimise the solution and find best parameters to get the best solution as fast as possible, we recommend using the ``crunch`` version unless you know what you are doing.

### Check your solution

This example being trivial it should finish instantly.
So we produced a file ``rtsm_solutions.json`` which contains one or more solution to our problem.
Now we would like to check our solution, we can use:

```bash
python -m rtsm.check_solution ./examples/humaneval_pass200.csv rtsm_solutions.json
```

You sould get a ranking error of 0 when using predictions.
Now this instance is quite easy and has few tests that can actually be removed.

### Multiprocessing

The ``SAT20-MAIN.csv`` contains 400 tests with none that can be at first glance deemed necessary making it a much harder problem so we will use multiple CPUs:

```bash
python -m rtsm.crunch ./examples/SAT20-MAIN.csv  -p 8
```

The first progress can come after a few minutes, it took 12min on my MacBook Pro (M1, 2020).
So let's say that after a while, we want to stop but we'd like not to lose our progress.
In fact, we can just kill the process and while exiting the best solutions found so far will be saved automatically.
Great!

### Start from an existing solution

Now, I would like to start from a solution that I found to see if I can find a better one:

```bash
python -m rtsm.crunch ./examples/SAT20-MAIN.csv  -p 8 --start my_solution.json
```

### Export your prediction model

Let us say now that we have a solution that we got with the ``linear+`` predictor model and we would like to export the prediction model to use it elsewhere then we can do:

```bash
python -m rtsm.export ./examples/x264_etime.csv my_solution.json --predictor "linear+" -o dst.json
```

### Multi performance metrics

What if you wanted to minimise the test set for two performance metrics? Well if both performance metrics are in your input file, that will be done automatically, you have nothing to do!

For multiple performance metrics, the accuracy parameter is such that the worse accuracy among all performance metrics is greater than this accuracy threshold.

### Genetic Algorithm

There is a GA solver, which is available only if [PyGAD](https://pygad.readthedocs.io/en/latest/index.html) is installed. However, it is not recommended as it is **dramatically slower** and offers **dramatically worse** performances than other methods. In other words, we did not manage to make it work despite our attemps. If you find a set of parameters that make geentic algorithm work, please reach out or contribute.

## Input Format

### CSV

There are two required columns in your CSV, one entitled ``variant`` and one entitled ``test``, every other column is considered as a different performance metric.
Both columns ``variant`` and ``test`` are considered as ``strings`` wherease all other columns are considered as ``float``.

See ``examples/x264_kbs_etime.csv`` for an example file that contains two different performance metrics.


## Citing

If you use this software, we encourage you to cite us:

TODO

```bibtex
@article{matricon24rtsm,
  author  = {Matricon, Th{\'e}o
               and Acher, Mathieu},
  year    = {2024},
  title   = {Minimising benchmarks and keep ranking variants
accurately}

}
```
