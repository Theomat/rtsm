# rtsm

Ranked Test Suite Minimisation (RTSM) a.k.a. Read The Super Manual

<!-- toc -->

- [Installation](#installation)
- [Usage](#usage)

<!-- tocstop -->

## Installation

Install ``rtsm`` with your favorite tool.

## Usage

The best way is to use the help flags but this section describes typical use cases.
A few examples are provided in the ``examples`` folder which we will use to describe the main usages of ``rtsm``.

Here is how you can start by simply running the following command:

```bash
python -m rtsm.crunch ./examples/humaneval_pass200.csv
```

This example being trivial it should finish instantly.
So we produced a file ``rtsm_solutions.json`` which contains one or more solution to our problem.
Now we would like to check our solution, we can use:

```bash
python -m rtsm.check_solution ./examples/humaneval_pass200.csv rtsm_solutions.json
```

You sould get a ranking error of 0 when using predictions.
Now this instance is quite easy and has few tests that can actually be removed.
The ```SAT20-MAIN.csv`` contains 400 tests with none that can be at first glance deemed necessary making it a much harder problem so we will use multiple CPUs:

```bash
python -m rtsm.crunch ./examples/SAT20-MAIN.csv  -p 8
```

The first progress can come after a few minutes, it took 12min on my MacBook Pro (M1, 2020).
So let's say that after a while, we want to stop but we'd like not to lose our progress.
In fact, we can just kill the process and while exiting the best solutions found so far will be saved automatically.
Great! Now, I would like to start from a solution that I found to see if I can find a better one:

```bash
python -m rtsm.crunch ./examples/SAT20-MAIN.csv  -p 8 --start my_solution.json
```

The ``x264_etime.csv`` contains 1397 tests, that's quite a lot, it already took minutes for one sample on 400 tests so it's likely to be very slow.
Well, we can actually use divide and conquer, it's easy:

```bash
python -m rtsm.crunch ./examples/x264_etime.csv  -p 8 --splits 14
```

This will split the 1397 test into 14 packets of approximately the same size and then do merging in order to find a solution.
This is much faster than other approaches, however we are making more greedy decisions.
