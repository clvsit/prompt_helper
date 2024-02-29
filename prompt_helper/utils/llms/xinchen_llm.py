import json
from typing import Any, List, Mapping, Optional, Iterator

import requests
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk


class EnglishChatLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        resp = requests.post(
            "http://model-hub-test.heyfriday.cn/v1/chat/completions",
            json={
                "task_id": "c56c27d5102b42c1a8c41ddc51d904dc",
                "model": "english_chat",
                "dialogue_id": "671184",
                "character_name": "Stepmom's family",
                "messages": [{"role": "system", "content": prompt, "name": ""}],
                "temperature": 0.7,
                "top_p": 1.0,
                "n": 1,
                "stream": False,
                "stop": None,
                "max_tokens": 500,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "search_web": False,
                "ability_type": "",
                "regenerate_history": None,
                "param": None,
            },
        )
        return resp.text

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        resp = requests.post(
            "http://model-hub-test.heyfriday.cn/v1/chat/completions",
            json={
                "task_id": "c56c27d5102b42c1a8c41ddc51d904dc",
                "model": "english_chat",
                "dialogue_id": "671184",
                "character_name": "Stepmom's family",
                "messages": [{"role": "system", "content": prompt, "name": ""}],
                "temperature": 0.7,
                "top_p": 1.0,
                "n": 1,
                "stream": True,
                "stop": None,
                "max_tokens": 500,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "search_web": False,
                "ability_type": "",
                "regenerate_history": None,
                "param": None,
            },
        )

        for chunk in resp.iter_lines():
            if not chunk:
                continue

            chunk_data: str = chunk.decode().replace("data:", "")
            yield GenerationChunk(text=chunk_data)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"n": 10}
