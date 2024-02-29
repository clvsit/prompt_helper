import unittest

from prompt_helper.utils.llms.xinchen_llm import EnglishChatLLM


class TestXinchenLlm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm = EnglishChatLLM()

    @unittest.skip("This test is skipped because it's an integration test.")
    def test_xinchen_llm(self):
        resp = self.llm("How many people live in canada as of 2023?")
        print(resp)

    def test_xinchen_llm_stream(self):
        for chunk in self.llm.stream("Write me a song about sparkling water."):
            print(chunk, end="", flush=True)


if __name__ == "__main__":
    unittest.main()
