# Examples

<!-- toc -->

- [HumanEval (Plus)](#human-eval-plus)
- [x264](#x264)
- [ASLib](#as-lib)
  - [SAT20-MAIN](sat20-main)

<!-- tocstop -->

## Human Eval (Plus)

**Variants**: 120
**Tests**: 164

These examples are taken from [evalplus](https://github.com/evalplus/evalplus).
The idea is compress the HumanEval dataset based on the leaderboard of a few LLMs.
In other words, we look to only have the few HumanEval tasks that bring about information in differencing the LLMs.

We provide two variants for two parameters, pass 1 and pass 200 and wether HumanEval was used or HumanEval Plus.

## x264

**Variants**: 201
**Tests**: 1397

These files contain performance of different x264 configurations for different video inputs.
The idea is to keep only files that are relevant in order to diferentiate between two configurations.

## AS Lib

See: [GitHub repository](https://github.com/coseal/aslib_data)

This repository contains a number of scenarios fro competitions between solvers, these problems can be rephrased as testing variants.

### SAT20-MAIN

**Variants**: 67
**Tests**: 400

Describes the time used by the SAT solvers of the SAT20 main track competiton.
