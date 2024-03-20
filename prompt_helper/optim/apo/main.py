import json
from typing import List

import tqdm
from loguru import logger

from .meta_prompt import apo_meta_prompt, apo_refine_meta_prompt
from ..abs_optimizer import ABSPromptOptimizer
from ..stop_criterion.abs_stop_criterion import ABSStopCriterion
from prompt_helper.utils.common import Consoler
from prompt_helper.utils.llms.server import ABSLLMServer


class APOPromptOptimizer(ABSPromptOptimizer):
    def __init__(self, llm_server: ABSLLMServer = None):
        self.__llm_server = llm_server

    def __generate_gradient(
        self, prompt: str, failure_case: str, n_reasons: int = 2, model: str = ""
    ) -> str:
        """生成梯度

        Args:
            prompt (str): prompt 文本
            failure_case (str): 错误示例
            n_reasons (int, optional): 总结的错误原因数量. Defaults to 2.
            model (str, optional): 调用的模型. Defaults to "".

        Raises:
            Exception: _description_

        Returns:
            str: _description_
        """

        apo_prompt_replaced = (
            apo_meta_prompt.replace("{{prompt}}", prompt)
            .replace("{{failure_string}}", failure_case)
            .replace("{{n_reasons}}", str(n_reasons))
        )
        # logger.info(apo_prompt_replaced)
        resp = self.__llm_server.generate(
            prompt=apo_prompt_replaced,
            stream=False,
            temperature=0.0,
            max_new_tokens=500,
            model=model,
        )

        if resp["code"] == 0:
            raise Exception(resp)

        resp_dict: dict = json.loads(resp["data"])
        gradient: str = resp_dict["data"]["choices"][0]["message"]["content"]

        return gradient

    def __generate_new_prompt(
        self,
        prompt: str,
        failure_case: str,
        gradient: str,
        max_tokens: int = 1500,
        model: str = "",
    ) -> str:
        """
        生成新 prompt
        :param prompt:       str
        :param failure_case: str
        :param gradient:     str
        :param max_tokens:   int
        :param model:        str
        """
        apo_refine_prompt_replaced = (
            apo_refine_meta_prompt.replace("{{prompt}}", prompt)
            .replace("{{failure_string}}", failure_case)
            .replace("{{max_tokens}}", str(max_tokens))
            .replace("{{gradient}}", gradient)
        )
        # logger.info(apo_refine_prompt_replaced)
        resp = self.__llm_server.generate(
            prompt=apo_refine_prompt_replaced,
            stream=False,
            temperature=0.0,
            max_new_tokens=max_tokens + 10,
            model=model,
        )

        resp_dict: dict = json.loads(resp["data"])
        new_prompt: str = resp_dict["data"]["choices"][0]["message"]["content"]

        return new_prompt

    def __make_failure_case_str(self, test_dataset: List[dict], failure_cases: List[dict]) -> str:
        """
        构造 failure case 字符串格式
        :param test_dataset:  List[dict] 测试数据集
        :param failure_cases: List[dict] failure case 列表
        :return str 返回 failure case 字符串格式
        """
        failure_case_str = ""

        for idx, case in enumerate(failure_cases):
            failure_case_str += (
                f"case {idx + 1}: user: {test_dataset[case['idx']]}\n{case['result']}\n"
            )

        return failure_case_str

    def run(
        self,
        model: str,
        prompt: str,
        test_dataset: List[dict],
        stop_criterions: List[ABSStopCriterion],
        n_reasons: int = 5,
        max_tokens: int = 1500,
        is_debug: bool = False,
    ):
        """
        :param is_debug:     bool       是否打开 debug，默认为 False
        """

        is_stop_flag: bool = False
        stop_reason: str = ""
        step: int = 0
        failure_cases = [
            {"idx": idx, "result": data["output"], "reason": ""}
            for idx, data in enumerate(test_dataset)
        ]

        while True:
            step += 1

            Consoler.print_in_panel(f"epoch-{step}: 生成梯度", title="APO 自动 prompt")
            failure_case_str: str = self.__make_failure_case_str(test_dataset, failure_cases)
            gradient: str = self.__generate_gradient(
                prompt=apo_meta_prompt,
                failure_case=failure_case_str,
                n_reasons=n_reasons,
                model=model,
            )
            logger.info(f"epoch-{step} | gradient:\n{gradient}")
            # pdb.set_trace()

            Consoler.print_in_panel(f"epoch-{step}: 生成新 prompt", title="APO 自动 prompt")
            new_prompt = self.__generate_new_prompt(
                prompt=prompt,
                failure_case=failure_case_str,
                gradient=gradient,
                max_tokens=max_tokens,
                model=model,
            )
            logger.info(f"epoch-{step} | new prompt:\n{new_prompt}")
            # pdb.set_trace()

            Consoler.print_in_panel(f"epoch-{step}: 开始执行评测流程", title="APO 自动 prompt")
            eval_result: dict = self.eval(prompt=new_prompt, test_dataset=test_dataset)
            accuracy: float = eval_result["accuracy"]
            failure_cases: List[dict] = eval_result["failure_cases"]
            logger.info(f"epoch-{step} | accuracy: {accuracy}")

            for stop_criterion in stop_criterions:
                if stop_criterion.is_stop(accuracy):
                    is_stop_flag = True
                    stop_reason = stop_criterion.stop_reason
                    break

            prompt = new_prompt

            if is_stop_flag:
                return {"prompt": prompt, "step": step, "stop_reason": stop_reason}

    def eval(self, prompt: str, test_dataset: List[dict]) -> dict:
        """
        执行评测流程
        :param prompt:       str        prompt 文本
        :param test_dataset: List[dict] 测试数据集
        :return dict
        """
        data_idx, hit_count = 0, 0
        error_info_list = []

        for data in tqdm.tqdm(test_dataset):
            resp = self.__llm_server.generate(
                prompt=prompt.replace("{{input_content}}", data["input"]),
                stream=False,
                temperature=0.0,
                max_new_tokens=500,
            )

            resp_dict: dict = json.loads(resp["data"])
            result_str: str = resp_dict["data"]["choices"][0]["message"]["content"]

            try:
                result_json = json.loads(result_str)

                if (
                    result_json[0]["label"] == data["expect"][0]["label"]
                    and result_json[0]["entity"] == data["expect"][0]["entity"]
                ):
                    hit_count += 1
                else:
                    error_info_list.append(
                        {"idx": data_idx, "reason": "no_hit", "result": result_json[0]}
                    )
            except Exception as error:
                error_info_list.append(
                    {"idx": data_idx, "reason": f"{error}", "result": result_str}
                )
            finally:
                data_idx += 1

        return {"accuracy": hit_count / len(test_dataset), "failure_cases": error_info_list}
