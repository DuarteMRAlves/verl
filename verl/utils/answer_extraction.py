import dataclasses
import re

from typing import Callable


@dataclasses.dataclass(frozen=True)
class AnswerExtractionResult:
    success: bool
    thinking: str
    answer: str

    @classmethod
    def successful(cls, thinking: str, answer: str) -> "AnswerExtractionResult":
        return cls(success=True, thinking=thinking, answer=answer)

    @classmethod
    def failed(cls) -> "AnswerExtractionResult":
        return cls(success=False, thinking="", answer="")


AnswerExtractor = Callable[[str], AnswerExtractionResult]


def force_think_extractor(output: str) -> AnswerExtractionResult:
    if output.count("<think>") != 0 or output.count("</think>") != 1:
        return AnswerExtractionResult.failed()
    
    match = re.match(r"^(.*)<\/think>(.*?)$", output, re.DOTALL)
    if not match:
        return AnswerExtractionResult.failed()
    
    thinking = match.group(1).strip()
    answer = match.group(2).strip()
    return AnswerExtractionResult.successful(thinking=thinking, answer=answer)


_answer_extractors = {
    "full": lambda x: AnswerExtractionResult.successful(thinking="", answer=x),
    "force-think": force_think_extractor,
}

def get_answer_extractor(name: str) -> AnswerExtractor:
    global _answer_extractors
    if name not in _answer_extractors:
        raise ValueError(f"Unknown answer extractor: {name}")
    return _answer_extractors[name]
