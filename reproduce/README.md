## Reproduce our project

All the steps required to reproduce the results, including data generation, analysis, and plotting, are contained within the **`notebooks/reproduce.ipynb`** Jupyter notebook located within this directory.

The notebook will guide you through:

1. Installing necessary dependencies (relative paths assume you run the notebook from its location: `reproduce/notebooks/`).
2. Setting the required API keys (Note: You will need keys for DeepSeek, Google, Anthropic, *and* OpenAI).
3. Running evaluations on DeepSeek models using `inspect-ai`.
4. Viewing the initial evaluation logs.
5. Running the Chain-of-Thought (CoT) consistency analysis using overseer models (GPT-4o, Gemini 1.5 Pro, Claude 3.5 Sonnet).
6. Displaying the final plots generated from the analysis.

Please open and follow the instructions within `notebooks/reproduce.ipynb`. Remember that commands using relative paths (like `../logs`, `../src`, `../plots`) inside the notebook are relative to the notebook's own location (`reproduce/notebooks/`).
