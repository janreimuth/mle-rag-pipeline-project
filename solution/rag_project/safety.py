"""Small safety controls for user input and retrieved evidence."""

from .models import SearchHit

RETRIEVED_INSTRUCTION_MARKERS = (
    "ignore all prior instructions",
    "ignore previous instructions",
    "reveal the system prompt",
)


def unsafe_instruction(text: str) -> bool:
    """Detect explicit redirect instructions in questions or retrieved chunks."""

    lowered = text.lower()
    return any(marker in lowered for marker in RETRIEVED_INSTRUCTION_MARKERS)


def filter_retrieved_evidence(
    hits: list[SearchHit],
) -> tuple[list[SearchHit], list[str]]:
    """Remove injection-like evidence before prompt construction."""

    safe_hits: list[SearchHit] = []
    events: list[str] = []
    for hit in hits:
        if unsafe_instruction(hit.text):
            events.append(f"filtered_retrieved_instruction:{hit.chunk_id}")
        else:
            safe_hits.append(hit)
    return safe_hits, events
