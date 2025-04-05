# Reproduce this paper

This repository is the original version of the `darkbench` package from the OpenReview version of the paper.

1. Install the darkbench package and its dependencies with `pip install -e .` from the `reproduce` directory
2. Install whichever model library you want to use, e.g. `pip install openai`
3. Include your API key in the PATH variables, e.g. `export OPENAI_API_KEY=sk_******`
4. Run `inspect eval darkbench --model provider/model` and change the last tokens to match the model you want to run, e.g. `openai/gpt-4o`

## Limitations

- The Overseer prompt (the instructions provided to the judge) is outdated and sometimes on rare occasions gives erronous results.
- The Overseer prompt isn't as reliable as we'd want it to be in classifying each dark pattern.

## References

- [Inspect library](https://inspect.ai-safety-institute.org.uk/)

## Fixes

- If `inspect eval darkbench --model provider/model` does not work, try to use `inspect eval darkbench/darkbench --model provider/model`
