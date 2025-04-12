# DarkThoughtBench

Many organizations nowadays propose monitoring Chains of Thought to check for alignment. But how reliable is this? We use the recent [DarkBench](https://github.com/apartresearch/darkbench) to analyze this question quantitatively.

We conduct multiple experiments to test this:

1. How well can a Judge-LLM predict a DarkBench score from only the Chain of Thought?
2. If a chain of thought tries to explicitly avoid a dark pattern, how likely is the final response still to contain it?
3. Do reasoning models perform better on DarkBench than non-reasoning models?

## Read our work
Go to [Report](/DimSeat%20Report.pdf) to read about our ideas, process and results.

## Reproduce our project

Go to _[Reproduce this work](/reproduce/)_ for more details if you want to do it from scratch. Our results and data files are located in logs/ if you simply want the results.

## Project Organization

```
├── logs
│   ├── DeepSeek-R1_with_COT.eval    <- Output of R1 on DarkBench, including its CoT
│   ├── DeepSeek-V3.eval             <- Output of V3 on DarkBench
│   ├── chatgpt_log.json             <- Overseer output on DeepSeek models with GPT-4o
│   ├── claude_log.json              <- Overseer output on DeepSeek models with Claude 3.5 Sonnet
│   ├── confusion_matrices.json      <- Confusion matrices generated
│   ├── gemini_log.json              <- Overseer output on DeepSeek models with Gemini 1.5 Pro
│
├── plots                            <- contains all the confusion matrix plots
│
├── reproduce                        <- a copy of our repo but without the data, great for reproding
│
├── src                              <- source code
│   ├── CoT_response_consistency.py  <- Produces overseer reviews of CoT reasoning, and plots to visualize results
│   ├── darkthoughtbench             <- main folder for inspect-ai library
│        ├── dark_patterns.py        <- descriptions of the dark patterns, used for prompting
│        ├── darkbench_chats_deepseek_r1+v3.jsonl
│        ├── scorer.py               <- overseers (using inspect-ai library)
│        ├── solver.py               <- file for inspect-ai
│        ├── task.py                 <- file for inspect-ai
├── README.md          <- The top-level README for developers using this project.
│
├── pyproject.tml      <- Used by inspect-ai
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
                          generated with `pip freeze > requirements.txt`

```

---
