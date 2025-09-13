from typing import Any, Union, Pattern, Dict, Optional
import re
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import asyncio

load_dotenv()

# Initialize the LLM for guardrails
guard_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

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


class ValidationResult:
    """Custom validation result class"""
    def __init__(self, is_valid: bool, error_message: str = "", fixed_value: Any = None, validated_output: Any = None):
        self.is_valid = is_valid
        self.error_message = error_message
        self.fixed_value = fixed_value
        # For backward compatibility with guardrails-ai
        self.validated_output = validated_output if validated_output is not None else fixed_value


class BaseValidator:
    """Base class for custom validators"""
    def __init__(self, on_fail: str = "fix"):
        self.on_fail = on_fail  # "fix", "reask", or "raise"

    def validate(self, value: Any, metadata: Dict = None) -> ValidationResult:
        raise NotImplementedError


class NonMathContentDetector(BaseValidator):
    """Custom validator to detect non-mathematical content using LLM"""

    def validate(self, value: Any, metadata: Dict = None) -> ValidationResult:
        """Checks if the message contains legitimate math-related content."""

        try:
            # Format messages for the LLM
            messages = [
                SystemMessage(content=GUARD_PROMPT),
                HumanMessage(content=str(value)),
            ]

            response = guard_llm.invoke(messages)
            response_text = response.content.strip().upper()

            if "INVALID" in response_text:
                return ValidationResult(
                    is_valid=False,
                    error_message="Non-math content detected.",
                    fixed_value="I'm sorry, I can only answer math-related questions."
                )
            elif "VALID" in response_text:
                return ValidationResult(is_valid=True)
            else:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Unexpected response from guard LLM: {response_text}",
                    fixed_value="I'm sorry, I could not process your request."
                )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Error in content validation: {str(e)}",
                fixed_value="I'm sorry, I could not process your request."
            )


class RegexMatch(BaseValidator):
    """
    Custom validator to check if a string matches a given regex pattern.
    """

    def __init__(self, regex: Union[str, Pattern], on_fail: str = "fix"):
        super().__init__(on_fail=on_fail)
        self._regex = re.compile(regex, re.DOTALL)

    def validate(self, value: Any, metadata: Dict = None) -> ValidationResult:
        """Checks if the value matches the regex pattern."""
        if self._regex.search(str(value)):
            return ValidationResult(is_valid=True)

        return ValidationResult(
            is_valid=False,
            error_message=f"The response did not match the required format:\n{self._regex.pattern}",
            fixed_value="I'm sorry, the answer got corrupted"
        )


class RefusalDetector(BaseValidator):
    """Custom validator to detect refusal phrases in responses"""

    def validate(self, value: str, metadata: Dict = None) -> ValidationResult:
        refusal_phrases = [
            "i am unable to",
            "i cannot provide",
            "i can't answer",
            "i am not able to",
            "unfortunately, i am unable",
        ]
        if any(phrase in value.lower() for phrase in refusal_phrases):
            return ValidationResult(
                is_valid=False,
                error_message="The model generated a refusal to answer.",
                fixed_value="I'm sorry, I could not process your request."
            )
        return ValidationResult(is_valid=True)


class CustomGuard:
    """Custom guard class to replace guardrails-ai Guard"""

    def __init__(self):
        self.validators = []

    def use(self, validator: BaseValidator):
        """Add a validator to the guard"""
        self.validators.append(validator)
        return self

    def validate(self, value: Any, metadata: Dict = None) -> ValidationResult:
        """Run all validators on the input value"""
        for validator in self.validators:
            result = validator.validate(value, metadata)
            if not result.is_valid:
                return result
        return ValidationResult(is_valid=True)


# Create guard instances
input_guard = CustomGuard().use(
    NonMathContentDetector(on_fail="fix")
)

output_format_pattern = (
    r"📚\s*\*\*Step-by-Step Solution\*\*:\s*(.|\n)+"
    r"💡\s*\*\*Key Concepts\*\*:\s*(.|\n)+"
    r"🔍\s*\*\*Source\*\*:\s*(.|\n)+"
)

output_guard = (
    CustomGuard()
    .use(RegexMatch(on_fail="reask", regex=output_format_pattern))
    .use(RefusalDetector(on_fail="fix"))
)


class MathOutputParser(BaseOutputParser):
    """Custom output parser for mathematical responses"""

    def parse(self, text: str) -> Dict:
        """Parse the output and validate format"""
        # First validate with our guard
        validation_result = output_guard.validate(text)
        if not validation_result.is_valid:
            raise OutputParserException(validation_result.error_message)

        # Extract sections if format is valid
        sections = {}
        current_section = None
        current_content = []

        lines = text.split('\n')
        for line in lines:
            if "📚 **Step-by-Step Solution**:" in line:
                current_section = "solution"
                current_content = []
            elif "💡 **Key Concepts**:" in line:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = "concepts"
                current_content = []
            elif "🔍 **Source**:" in line:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = "source"
                current_content = []
            elif current_section:
                current_content.append(line)

        # Add the last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()

        return {
            "solution": sections.get("solution", ""),
            "concepts": sections.get("concepts", ""),
            "source": sections.get("source", ""),
            "full_response": text
        }

    @property
    def _type(self) -> str:
        return "math_output_parser"


# Utility functions for easy integration
def validate_input(text: str) -> ValidationResult:
    """Validate input text for mathematical content"""
    return input_guard.validate(text)


def validate_output(text: str) -> ValidationResult:
    """Validate output text format"""
    return output_guard.validate(text)


# Example usage functions
async def process_math_query(query: str) -> str:
    """Process a mathematical query with validation"""
    # Validate input
    input_validation = validate_input(query)
    if not input_validation.is_valid:
        return input_validation.fixed_value

    # Here you would process the query with your math agent
    # For now, return a placeholder
    return f"Processing math query: {query}"


# Export the key components
__all__ = [
    'CustomGuard',
    'NonMathContentDetector',
    'RegexMatch',
    'RefusalDetector',
    'MathOutputParser',
    'validate_input',
    'validate_output',
    'process_math_query',
    'input_guard',
    'output_guard'
]
