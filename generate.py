"""Final pipeline steps: build grounded context and generate the answer."""


from models import (
    ContextItem,
    configuration_error,
    generation_error,
    get_chat_model,
)

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


def build_context(items: list[ContextItem]) -> str:
    """Format results into one labelled prompt, dropping duplicate content."""
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


async def generate_answer(question: str, context: str) -> str:
    """Call Gemini with the grounded context and reject empty or failed answers."""
    llm = get_chat_model()
    if llm is None:
        raise configuration_error("GEMINI_API_KEY is required to generate answers.")

    try:
        response = await llm.ainvoke(
            ANSWER_PROMPT.format(question=question, context=context)
        )
    except Exception as exc:
        raise generation_error("Gemini could not generate an answer.") from exc

    answer = str(response.content).strip()
    if not answer:
        raise generation_error("Gemini returned an empty answer.")
    return answer
