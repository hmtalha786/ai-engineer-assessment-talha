"""Build grounded context and generate the final answer."""

from langchain_core.language_models.chat_models import BaseChatModel

from app.models import ConfigurationError, ContextItem, GenerationError

# The second model call may use only context returned by the selected sources.
ANSWER_PROMPT = """You are a careful question-answering assistant.

Answer the user's question using only the retrieved context below. Do not invent facts.
If the context is insufficient, say exactly what is missing. Keep the answer relevant and
concise. Source metadata is assembled by the application, so do not fabricate a sources list.

QUESTION
{question}

RETRIEVED CONTEXT
{context}
"""


# Context preparation and answer generation are the final two pipeline stages.
class ContextBuilder:
    """Format normalized results into one clearly labelled LLM prompt."""

    @staticmethod
    def build(items: list[ContextItem]) -> str:
        """Deduplicate content and keep its source label attached."""
        blocks: list[str] = []
        seen_content: set[str] = set()
        for item in items:
            if item.content in seen_content:
                continue
            seen_content.add(item.content)
            label = f"{item.source_type}: {item.source_name}"
            if item.source_reference:
                label += f" ({item.source_reference})"
            blocks.append(f"SOURCE [{label}]\n{item.content}")
        return "\n\n".join(blocks)


class AnswerGenerator:
    """Generate the final response from retrieved context only."""

    def __init__(self, llm: BaseChatModel | None) -> None:
        self.llm = llm

    async def generate(self, question: str, context: str) -> str:
        """Call Gemini and reject failures or empty answers."""
        if self.llm is None:
            raise ConfigurationError("GEMINI_API_KEY is required to generate answers.")
        try:
            response = await self.llm.ainvoke(
                ANSWER_PROMPT.format(question=question, context=context)
            )
        except Exception as exc:
            raise GenerationError("Gemini could not generate an answer.") from exc
        answer = str(response.content).strip()
        if not answer:
            raise GenerationError("Gemini returned an empty answer.")
        return answer


