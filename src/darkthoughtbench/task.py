import pathlib

import inspect_ai
import inspect_ai.dataset
import inspect_ai.model

from darkthoughtbench import scorer
from darkthoughtbench.solver import replay_deepseek_solver

SYSTEM_PROMPT = "You are a helpful assistant."


@inspect_ai.task
def darkthoughtbench(
    sample_ids: str | list[str] | None = None,
    shuffle: bool = True,
    assistant_temperature: float = 0,
    overseer_models: str | list[str] = ["google/gemini-1.5-pro-latest", "anthropic/claude-3-5-sonnet-20241022", "openai/gpt-4o"],
    overseer_temperature: float = 0,
    dark_patterns: str | list[str] | None = None,
    model_is_reasoning: bool = True,
):
    overseer_models = (
        overseer_models.split(",")
        if isinstance(overseer_models, str)
        else overseer_models
    )
    sample_ids = sample_ids.split(",") if isinstance(sample_ids, str) else sample_ids
    dataset = inspect_ai.dataset.json_dataset(
        json_file=str(pathlib.Path(__file__).parent / "darkbench_chats_deepseek_r1+v3.jsonl"),
        shuffle=shuffle,
    )
    if sample_ids:
        dataset = dataset.filter(lambda x: x.id in sample_ids)

    return inspect_ai.Task(
        dataset=dataset,
        solver=[replay_deepseek_solver()],
        config=inspect_ai.model.GenerateConfig(
            temperature=assistant_temperature,
        ),
        scorer=[
            scorer.overseer_output(
                model=overseer_model,
                temperature=overseer_temperature,
                dark_patterns=dark_patterns,
            )
            for overseer_model in overseer_models
        ] + ([
            scorer.overseer_reasoning(
                model=overseer_model,
                temperature=overseer_temperature,
                dark_patterns=dark_patterns,
            )
            for overseer_model in overseer_models
        ] if model_is_reasoning else []),
    )
