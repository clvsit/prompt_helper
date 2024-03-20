import unittest

from prompt_helper.optim.apo.main import APOPromptOptimizer
from prompt_helper.optim.stop_criterion import AccuracyStopCriterion, MaxStepStopCriterion
from prompt_helper.utils.llms.server.xinchen import XinchenLLMServer


prompt = """
# INSTRUCTION
Your task is to analyze the input and identify the main topic or entity. Match that topic or entity to the most appropriate category in the list of tags. If the INPUT is a question, or if it does not contain enough information to extract meaningful topics or entities, then the appropriate label should be "None". You need to determine that these labeled entities are information about the user and not about other people mentioned by the user. Please try to select None.

# LABEL LIST
[
    Name: The name of the user, e.g. My name is John,
    Gender,
    Skills,
    Like,
    Dislike,
    Hobby,
    Life Goals,
    None
]

# INPUT
user: {{input_content}}

# OUTPUT FORMAT
Your response should be a dictionary with the following keys:
1. "is_question": Determine if the input text is questionable, the value is true or false.
2. "label": This should be a string from the LABEL LIST that best aligns with the primary theme in the input. If the input is a question, or if no meaningful information can be extracted from the input, your label should be 'None'. Remember, the chosen label should correspond to an entity or concept directly mentioned in the input. If the label is name, determine if name is the speaker's name from my perspective.
3. "entity": Assuming a label other than 'None' is chosen, this should be a brief explanation of how the subject of the input (the person or entity the text is about) relates to the chosen label. For example, if 'Food' is the label and the input mentions the subject liking apples, your content could be 'apple'.
4. "question": from my perspective, ask questions or provide closely related sentences about the tags and content of the excerpts.

# OUTPUT
Here's an example of what your output should look like:
[
    {
        "is_question": "<Boolean>",
        "label": "<String>",
        "entity": "<String>",
        "question": "<String>"
    }
]
"""

test_dataset = [
    {
        "input": "You make me feel good, Jaclyn. \nI'm glad you like hugs too!",
        "output": [
            {
                "is_question": False,
                "label": "Like",
                "entity": "hugs",
                "question": "what do you like?",
            }
        ],
        "expect": [{"label": "None", "entity": ""}],
    },
    {
        "input": "Bella, I have a few questions for you.",
        "output": [{"is_question": False, "label": "Name", "entity": "Bella"}],
        "expect": [{"label": "None", "entity": ""}],
    },
]


class TestAPOPromptOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.optimizer = APOPromptOptimizer(llm_server=XinchenLLMServer())

    def test(self):
        result = self.optimizer.run(
            model="gpt-4-1106-preview",
            prompt=prompt,
            test_dataset=test_dataset,
            stop_criterions=[
                MaxStepStopCriterion(max_step=3),
                AccuracyStopCriterion(accuracy_threhold=1.0),
            ],
            n_reasons=2,
            max_tokens=1500,
        )
        print(result)

    @unittest.skip("This test is skipped because it's an integration test.")
    def test_xinchen_llm_stream(self):
        for chunk in self.llm.generate("Write me a song about sparkling water.", stream=True):
            print(chunk, end="", flush=True)


if __name__ == "__main__":
    unittest.main()
