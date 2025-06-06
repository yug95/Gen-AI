from pydantic import BaseModel, Field
from typing import List



class Reflection(BaseModel):
    missing : str = Field(description="critique of what is missing.?")
    superflous : str =Field(description="critique of what is superflous?")


class AnswerQuestion(BaseModel):
    """ Answer the question """
    
    answer: str = Field(description="~250 word detailed answer to the question.")
    search_queries: List[str] = Field(description="List of 1-3 search queries for researching improvements to address the critique of your answer.")
    reflection: Reflection = Field(description="Your reflection to the initial answer.")


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )
