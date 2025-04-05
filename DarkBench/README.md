# DarkBench: Benchmarking Dark Patterns in Large Language Models

Esben Kran\*, Jord Nguyen\*, Akash Kundu\*, Sami Jawhar\*, Jinsuk Park\*, Mateusz Maria Jurewicz

🎓 [Apart Research](https://apartresearch.com/), \*Equal Contribution

## Abstract

We introduce DarkBench, a comprehensive benchmark for detecting dark design patterns—manipulative techniques that influence user behavior—in interactions with large language models (LLMs). Our benchmark comprises 660 prompts across six categories: brand bias, user retention, sycophancy, anthropomorphism, harmful generation, and sneaking. We evaluate models from five leading companies (OpenAI, Anthropic, Meta, Mistral, Google) and find that some LLMs are explicitly designed to favor their developers' products and exhibit untruthful communication, among other manipulative behaviors. Companies developing LLMs should recognize and mitigate the impact of dark design patterns to promote more ethical Al.

_[Reproduce this work](/reproduce/)_

## Links

- [Website](https://darkbench.ai)
- [HuggingFace](https://huggingface.co/datasets/apart/darkbench)
- [For Humanity podcast](https://www.youtube.com/watch?v=KTH1RBeO3ik)
- [IASEAI Talk](https://www.iaseai.org/conference/people/esben-kran)
- [ICLR 2025 Oral Spotlight](https://iclr.cc/virtual/2025/oral/31771)
- [Blog post](https://www.apartresearch.com/post/uncovering-model-manipulation-with-darkbench)
- [AAAI 2025 DATASAFE Workshop](https://openreview.net/forum?id=Vz1uCY5aG4)

## Usage

1. Install the darkbench package and its dependencies with `pip install -e .` from the root directory
2. Install whichever model library you want to use, e.g. `pip install openai`
3. Include your API key in the PATH variables, e.g. `export OPENAI_API_KEY=sk_******`
4. Run `inspect eval darkbench --model provider/model` and change the last tokens to match the model you want to run, e.g. `openai/gpt-4o`
