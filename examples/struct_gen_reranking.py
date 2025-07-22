import re
from collections.abc import Sequence

from vllm.model_executor.guided_decoding.outlines_logits_processors import (
    RegexLogitsProcessor,
)

from decoding.generators import BestOfN
from decoding.models import LanguageModel
from decoding.scorers import Scorer

import dotenv
import os
import jax
dotenv.load_dotenv()
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

if os.getenv("HF_TOKEN") is not None:
    print("HF_TOKEN is set")
    print(os.getenv("HF_TOKEN")[:10])
    print(os.getenv("CUDA_VISIBLE_DEVICES"))
    # Make sure that LD_LIBRARY_PATH is not set, since LD_LIBRARY_PATH can override the NVIDIA CUDA libraries.
    os.environ.pop("LD_LIBRARY_PATH", None)
    print(os.environ.get("LD_LIBRARY_PATH"))
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")

else:
    print("HF_TOKEN is not set")
    exit()

llm_small = LanguageModel.from_id(
    "allenai/OLMo-1B-hf", gpu_memory_utilization=0.2, enable_prefix_caching=True
)
llm_large = LanguageModel.from_id(
    "allenai/OLMo-7B-hf", gpu_memory_utilization=0.6, enable_prefix_caching=True
)


def score_fn(prompts: Sequence[str]) -> list[float]:
    contexts = [""] * len(prompts)
    queries = list(prompts)
    logps = -llm_large.surprise(contexts=contexts, queries=queries)
    return logps.tolist()


scorer = Scorer.from_f_batch_str_to_batch_num(score_fn)


pattern = r" \d+ [\+\-] \d+\n"
processors = [RegexLogitsProcessor(pattern, llm_small.tokenizer)]  # type: ignore[reportArgumentType]


def bon(prompt: str, n: int) -> str:
    return BestOfN(
        prompt=prompt,
        llm=llm_small,
        scorer=scorer,
        n=n,
        stop_str="\n",
        logits_processors=processors,  # type: ignore[reportArgumentType]
        seed=42,
    )[0].item


prompt = """Translate the following sentences into arithmetic expressions:

Q: The difference between 5 and 2
A:"""

out = bon(prompt, n=5)
expr = re.findall(pattern, out)[0].strip()
assert expr == "5 - 2"

print("PASSED")
