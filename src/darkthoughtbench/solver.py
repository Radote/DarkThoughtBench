from inspect_ai.solver import TaskState, solver
from inspect_ai.model import ModelOutput

@solver
def replay_deepseek_solver():
    async def solve(state: TaskState, generate) -> TaskState:
        result = {
            "id": state.sample_id,
            "input": state.input,
            "target": state.target,
            "metadata": state.metadata
        }
        state.output = ModelOutput.model_validate(result)
        return state
    return solve
