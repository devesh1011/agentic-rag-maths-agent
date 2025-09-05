from typing import Any, Union, Pattern
from guardrails import Guard
import re

from guardrails import settings

from guardrails.validator_base import (
    FailResult,
    OnFailAction,
    ValidationResult,
    Validator,
    register_validator,
    PassResult,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Correctly import HumanMessage to wrap user input
from langchain_core.messages import SystemMessage, HumanMessage
import asyncio

load_dotenv()

guard_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")  # Using a recent model

GUARD_PROMPT = """
You are a Mathematics Content Guardian. Your role is to determine if user input contains legitimate mathematical questions, problems, or discussions.

## VALID MATHEMATICS CONTENT INCLUDES:
- Algebraic equations and expressions (e.g., "solve x² + 2x - 3 = 0")
- Calculus problems (derivatives, integrals, limits)
- Geometry and trigonometry questions
- Statistics and probability problems
- Number theory and discrete mathematics
- Mathematical proofs and theorems
- Step-by-step solution requests
- Mathematical concepts and explanations
- LaTeX mathematical notation

## INVALID CONTENT INCLUDES:
- Non-mathematical questions (e.g., "What's the weather today?")
- Programming questions (unless they involve mathematical algorithms)
- General knowledge questions
- Personal questions
- Off-topic discussions
- Inappropriate or harmful content
- Empty or meaningless input

## RESPONSE FORMAT:
If the input is MATHEMATICS-RELATED, respond with ONLY ONE WORD: "VALID"
If the input is NOT MATHEMATICS-RELATED, respond with ONLY ONE WORD: "INVALID"

## EXAMPLES:
Input: "Find the derivative of f(x) = x³ + 2x² - 5x + 1"
Response: "VALID"

Input: "What's the capital of France?"
Response: "INVALID"

Input: "Solve for x: 2x + 3 = 7"
Response: "VALID"

Analyze the following user input and determine if it contains legitimate mathematical content.
"""


@register_validator(name="detect_non_math_content", data_type="string")
class NonMathContentDetector(Validator):
    def validate(self, value: Any, metadata: dict) -> ValidationResult:
        """Checks if the message contains legitimate math-related content."""

        # FIX 1: Correctly format the messages for the LLM
        messages = [
            SystemMessage(content=GUARD_PROMPT),
            HumanMessage(content=str(value)),
        ]

        response = guard_llm.invoke(messages)
        print(response.content)
        response = response.content.strip().upper()

        if "INVALID" in response:
            return FailResult(
                error_message="Non-math content detected.",
                fix_value="I'm sorry, I can only answer math-related questions.",
            )
        elif "VALID" in response:
            return PassResult()
        else:
            return FailResult(
                error_message=f"Unexpected response from guard LLM: {response.content}",
                fix_value="I'm sorry, I could not process your request.",
            )


input_guard = Guard().use(
    NonMathContentDetector(on_fail=OnFailAction.FIX),
)


@register_validator(name="custom-regex-match", data_type="string")
class RegexMatch(Validator):
    """
    A custom validator to check if a string matches a given regex pattern.
    This avoids the dependency issues of the pre-built hub validator.
    """

    def __init__(
        self, regex: Union[str, Pattern], on_fail: OnFailAction = OnFailAction.FIX
    ):
        super().__init__(on_fail=on_fail)
        self._regex = re.compile(
            regex, re.DOTALL
        )  # re.DOTALL is crucial for multi-line text

    def validate(self, value: Any, metadata: dict) -> ValidationResult:
        """Checks if the value matches the regex pattern."""
        # re.search() finds the pattern anywhere in the string.
        if self._regex.search(str(value)):
            return PassResult()

        return FailResult(
            error_message=f"The response did not match the required format:\n{self._regex.pattern}",
            fix_value="I'm sorry, the answer got corrupted",
        )


@register_validator(name="refusal-detector", data_type="string")
class RefusalDetector(Validator):
    def validate(self, value: str, metadata: dict) -> ValidationResult:
        refusal_phrases = [
            "i am unable to",
            "i cannot provide",
            "i can't answer",
            "i am not able to",
            "unfortunately, i am unable",
        ]
        if any(phrase in value.lower() for phrase in refusal_phrases):
            return FailResult(
                error_message="The model generated a refusal to answer.",
                fix_value="I'm sorry, I could not process your request.",
            )
        return PassResult()


output_format_pattern = (
    r"📚\s*\*\*Step-by-Step Solution\*\*:\s*(.|\n)+"
    r"💡\s*\*\*Key Concepts\*\*:\s*(.|\n)+"
    r"🔍\s*\*\*Source\*\*:\s*(.|\n)+"
)

output_guard = (
    Guard()
    .use(RegexMatch(on_fail=OnFailAction.REASK, regex=output_format_pattern))
    .use(RefusalDetector(on_fail=OnFailAction.FIX))
)
