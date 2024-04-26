from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM

from llama_index.llms.together import TogetherLLM
from dotenv import load_dotenv
import os

load_dotenv()


class TogetherModel(DeepEvalBaseLLM):

    def __init__(self):
        self.model = TogetherLLM(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            api_key = os.getenv("TogetherAI_API_KEY")
        )
    
    def load_model(self):
        return self.model

    def generate(self, prompt):
        answer = self.model.complete(prompt)
        return answer
    
    # Todo verify this method
    async def a_generate(self, prompt):
        answer = await self.model.complete(prompt)
        return answer
    
    def get_model_name(self):
        return "MistraLAI Mixtral 8x7B Instruct v0.1"
    
    def generate_raw_response(self, prompt, logprobs=0, top_logprobs=0, raw=True):
        return (self.model.complete(prompt),0)

model = TogetherModel()
print(model.generate("Teste"))

coherence_metric = GEval(
    name="Coherence",
    criteria="Coherence - determine if the actual output is coherent with the input.",
    # NOTE: you can only provide either criteria or evaluation_steps, and not both
    evaluation_steps=["Check whether the sentences in 'actual output' aligns with that in 'input'"],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=model,
    async_mode=False
)

test_case = LLMTestCase(
    input="The sun is shining bright today",
    actual_output="The weather's getting really hot."
)

coherence_metric.measure(test_case)
print(coherence_metric.score)
print(coherence_metric.reason)
