# DarkThoughtBench

Many organizations nowadays propose monitoring Chains of Thought to check for alignment. But how reliable is this? We use the recent DarkBench to analyze this question quantitatively.

We conduct multiple experiments to t:

1. How well can a Judge-LLM predict a DarkBench score from only the Chain of Thought?
2. If a chain of thought tries to explicitly avoid a dark pattern, how likely is the final response still to contain it?

## Results

## Reproduce our project

First, open the terminal and run

```
export OPENAI_API_KEY={your api key}
export CLAUDE_API_KEY={your api key}
export GEMINI_API_KEY={your api key}
```

### From scratch

### Using our results

## Project Organization

```
├── README.md          <- The top-level README for developers using this project.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
├── src   <- Source code for use in this project.
│   ├── __init__.py
├── logs
│   ├── 2025-04-05T03-14-06+02-00_darkbench_QLsQ4nWaNH4wCo2YgXdVVH.eval    <- results file from the run, but with broken brand-bias eval
│   ├── 2025-04-05T15-34-26+02-00_darkbench_QwJrkQ5jgRMtCe5esTDCw5.eval    <- results file with fixed brand-bias eval
│

```

---
