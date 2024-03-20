import re
from typing import List

import spacy
import torch
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM

from .types import LexicalUnits


class SelectiveContext:
    def __init__(self, model_name_or_path: str, device: str = "cuda", lang: str = "en"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
        self.device = device

        self.sent_tokenize_pattern: str = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
        self.phrase_mask_token: str = ""
        self.sent_mask_token: str = "<...some content omiited.>"
        self.keep_leading_word = False
        self.mask_token = ""

        if lang == "en":
            self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
            self.nlp.add_pipe("merge_noun_chunks")
        elif lang == "zh":
            self.nlp = spacy.load("zh_core_web_sm", disable=["ner"])

    def __call__(
        self, context: str, reduce_ratio: float = 0.35, reduce_level: str = "phrase"
    ) -> List[str]:
        if reduce_level not in ["phrase", "sentence", "token"]:
            raise ValueError(
                f"reduce_level should be one of ['sentence', 'phrase', 'token'], got {reduce_level}"
            )

        context: str = re.sub(r"\s+", " ", context)
        sentences: List[str] = [
            sent.strip() for sent in re.split(self.sent_tokenize_pattern, context) if sent.strip()
        ]

        lexical_unit = self.__split_lexical_unit(sentences=sentences, unit_type=reduce_level)

        sentences_after_mask = []
        masked_sentences = []
        ppl_threshold = np.nanpercentile(lexical_unit.self_infos, reduce_ratio * 100)

        for sentence, info in zip(lexical_unit.texts, lexical_unit.self_infos):
            if info < ppl_threshold:
                masked_sentences.append(sentence)
                sentences_after_mask.append(self.__mask_sentence(sentence, reduce_level))
            else:
                sentences_after_mask.append(sentence)

        masked_context = (
            " ".join(sentences_after_mask)
            if reduce_level == "sentence"
            else "".join(sentences_after_mask)
        )

        return context, masked_context

    """
    工具函数区域
    """

    def __mask_sentence(self, sentence: str, reduce_level: str) -> str:
        """掩码句子

        Args:
            sentence (str): 句子文本
            reduce_level (str): 掩码级别

        Returns:
            str: 返回掩码后的句子文本
        """
        if reduce_level == "phrase":
            return self.phrase_mask_token
        elif reduce_level == "sentence":
            if self.keep_leading_word:
                leading_few_words = " ".join(word_tokenize(sentence)[: self.num_lead_words]) + " "
            else:
                leading_few_words = ""
            return leading_few_words + self.mask_token
        elif reduce_level == "token":
            return ""

    def __get_self_information(self, sentence: str) -> dict:
        """获取句子内各 token 的自信息

        Args:
            sentence (str): _description_

        Returns:
            dict: 字典，包含 tokens 和 self_infos 两个键
        """
        inputs = self.tokenizer(sentence, add_special_tokens=False, return_tensors="pt").to(
            self.device
        )
        input_ids = inputs["input_ids"]
        input_ids_expaned = input_ids[:, 1:].unsqueeze(-1)

        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        self_info = -torch.log(probs)

        tokens: List[str] = [
            self.tokenizer.decode(token_) for token_ in input_ids.squeeze().tolist()[1:]
        ]
        self_infos: List[float] = (
            self_info[:, :-1].gather(-1, input_ids_expaned).squeeze(-1).squeeze(0).tolist()
        )

        return {"tokens": tokens, "self_infos": self_infos}

    def _calculate_lexical_unit(self, tokens: List[str], self_info: List[float]):
        def __noun_phrases(sent):
            noun_phrases = []
            doc = self.nlp(sent)

            for index, chunk in enumerate(doc):
                if index == 0:
                    noun_phrases.append(chunk.text)
                else:
                    noun_phrases.append(doc[index - 1].whitespace_ + chunk.text)
            return noun_phrases

        def __unit_info(tokens, self_info, units):
            current_unit_idx = 0
            current_position = 0
            unit_self_info = [[] for _ in range(len(units))]

            for idx, (token, info) in enumerate(zip(tokens, self_info)):
                current_position += len(token)
                if current_position == len(units[current_unit_idx]):
                    unit_self_info[current_unit_idx].append(info)
                    current_position = current_position - len(units[current_unit_idx])
                    current_unit_idx += 1
                elif current_position > len(units[current_unit_idx]):
                    counter_ = 1
                    current_position = current_position - len(units[current_unit_idx])
                    current_unit_idx += 1
                    while current_position >= len(units[current_unit_idx]):
                        counter_ += 1
                        current_position = current_position - len(units[current_unit_idx])
                        current_unit_idx += 1
                        if current_unit_idx >= len(units):
                            break
                    partial_info = info / counter_
                    for _ in range(counter_):
                        unit_self_info[(current_unit_idx - 1) - _].append(partial_info)
                else:
                    if token == " ":
                        continue
                    unit_self_info[current_unit_idx].append(info)

            unit_self_info_ = [np.mean(info) for info in unit_self_info]
            return unit_self_info_

        sent = "".join(tokens)
        noun_phrases = __noun_phrases(sent)
        noun_phrases_info = __unit_info(tokens, self_info, noun_phrases)

        return noun_phrases, noun_phrases_info

    def __split_lexical_unit(self, sentences: List[str], unit_type: str = "phrase"):
        texts: List[str] = []
        self_infos: List[float] = []

        for sentence in sentences:
            infos: dict = self.__get_self_information(sentence)
            tokens, self_info = infos["tokens"], infos["self_infos"]

            if unit_type == "phrase":
                noun_phrases, noun_phrases_info = self._calculate_lexical_unit(tokens, self_info)

                # We need to add a space before the first noun phrase for every sentence except the first one
                if texts:
                    texts[0] = f" {texts[0]}"

                texts.extend(noun_phrases)
                self_infos.extend(noun_phrases_info)
            elif unit_type == "token":
                texts.extend(tokens)
                self_infos.extend(self_info)
            else:
                texts.append(sentence)
                self_infos.append(np.mean(self_info))

        return LexicalUnits(unit_type=unit_type, texts=texts, self_infos=self_infos)
