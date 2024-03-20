import unittest

from prompt_helper.utils.llms.server.xinchen import XinchenLLMServer


class TestXinchenLLMServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm = XinchenLLMServer()

    def test_xinchen_server(self):
        resp = self.llm.generate("How many people live in canada as of 2023?")
        print(resp)

    @unittest.skip("This test is skipped because it's an integration test.")
    def test_xinchen_llm_stream(self):
        for chunk in self.llm.generate("Write me a song about sparkling water.", stream=True):
            print(chunk, end="", flush=True)


if __name__ == "__main__":
    unittest.main()
