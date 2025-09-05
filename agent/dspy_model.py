import dspy


class GenerateMathSolution(dspy.Signature):
    """Generate a detailed, step-by-step solution for a given mathematical question, using the provided context. Follow the standard response format."""

    context = dspy.InputField(
        desc="Relevant information from a knowledge base or web search. This is considered authoritative."
    )
    question = dspy.InputField(desc="The user's mathematical question.")
    solution = dspy.OutputField(desc="The final, formatted, step-by-step solution.")


class MathRAG(dspy.Module):
    """The DSPy module for generating math solutions."""

    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(GenerateMathSolution)

    def forward(self, question, context):
        """The execution logic of the module."""
        result = self.generate_answer(question=question, context=context)
        return result
