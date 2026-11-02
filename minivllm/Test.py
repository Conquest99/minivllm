from pathlib import Path

from minivllm import EngineConfig, LLMEngine, SamplingParams

MODEL = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B-Instruct"
llm = LLMEngine(EngineConfig(model=str(MODEL)))

prompts = [
    "Use Chinese answer to me,The futrue of Ai is ",
    "Explain the theory of relativity in simple terms.",
    "What are the benefits of meditation for mental health?",
]

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated Text: {output.text}")
