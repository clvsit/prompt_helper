import uuid
import json
from typing import Any, List, Mapping, Optional, Iterator

import requests
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk

from . import ABSLLMServer


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
        model: str = kwargs.get("model", "gpt-3.5-turbo")
        temperature: float = kwargs.get("temperature", 0.7)
        top_p: float = kwargs.get("top_p", 1.0)
        max_tokens: int = kwargs.get("max_tokens", 500)
        presence_penalty: float = kwargs.get("presence_penalty", 0.0)
        frequency_penalty: float = kwargs.get("frequency_penalty", 0.0)
        timeout: int = kwargs.get("timeout", 120)

        try:
            response = requests.post(
                "http://bridge.xinchenai.com/v1/chat/completions",
                json={
                    "request_id": str(uuid.uuid4()),
                    "messages": [{"role": "system", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "model": model,
                    "source": "joyland",
                },
                verify=False,
                timeout=timeout,
            )
            result = json.loads(response.text)

            if result["code"] == 0:
                return json.dumps(result)
            else:
                print(result)
                return ""
        except Exception as e:
            print(f"chatgpt Error: {e}")
            return None

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


class XinchenLLMServer(ABSLLMServer):
    def __init__(self):
        super().__init__()
        self.__llm = EnglishChatLLM()

    def generate(self, prompt: str, stream: bool = False, **kwargs) -> dict:
        if stream:
            resp = self.__llm.stream(prompt, kwargs)
            return resp

        try:
            resp: str = self.__llm(prompt, **kwargs)
            return {"code": 1, "msg": "Success", "data": resp}
        except Exception as error:
            return {"code": 0, "msg": f"Failed to generate response. reason: {error}", "data": ""}
