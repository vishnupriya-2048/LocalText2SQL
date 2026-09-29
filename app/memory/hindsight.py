ge"""Long-term memory via Hindsight, scoped per connected database.

Each database gets its own Hindsight bank (derived from a non-secret database
identifier), so schema-specific knowledge from one database is never recalled
for another. Memory is supporting context only: failures degrade to "no
memories" instead of breaking the question.
"""

import logging
from typing import Protocol

from hindsight_client import Hindsight

from app.config import Settings

log = logging.getLogger(__name__)

_BANK_MISSION = (
    "You remember knowledge about one relational database that helps translate "
    "questions into SQL: relationships between tables, business terminology and "
    "what it maps to in the schema, interpretations of ambiguous questions, "
    "corrections learned from failed queries, and query approaches that worked."
)
_MAX_MEMORIES = 8


class MemoryUnavailable(Exception):
    pass


class Memory(Protocol):
    enabled: bool

    def recall(self, database_id: str, query: str) -> list[str]: ...

    def retain(self, database_id: str, content: str, context: str) -> None: ...


class NullMemory:
    """Used when Hindsight is not configured."""

    enabled = False

    def recall(self, database_id: str, query: str) -> list[str]:
        return []

    def retain(self, database_id: str, content: str, context: str) -> None:
        return None


def bank_id_for(database_id: str) -> str:
    return f"text2sql-{database_id}"


class HindsightMemory:
    enabled = True

    def __init__(self, settings: Settings, client: Hindsight | None = None):
        if client is None:
            if not settings.hindsight_api_url:
                raise MemoryUnavailable("HINDSIGHT_API_URL is not set.")
            client = Hindsight(
                base_url=settings.hindsight_api_url,
                api_key=settings.hindsight_api_key,
                timeout=120.0,
            )
        self._client = client
        self._retain_async = settings.hindsight_retain_async
        self._missions_set: set[str] = set()

    def check(self) -> None:
        try:
            self._client.get_version()
        except Exception as exc:  # noqa: BLE001 - any client/transport failure means "unavailable"
            raise MemoryUnavailable(f"Hindsight is not reachable: {exc}") from None

    def recall(self, database_id: str, query: str) -> list[str]:
        response = self._client.recall(
            bank_id=bank_id_for(database_id), query=query, max_tokens=1024, budget="low"
        )
        return [r.text for r in response.results[:_MAX_MEMORIES]]

    def retain(self, database_id: str, content: str, context: str) -> None:
        bank = bank_id_for(database_id)
        if bank not in self._missions_set:
            try:
                self._client.set_mission(bank, _BANK_MISSION)
            except Exception as exc:  # noqa: BLE001 - mission is an optimisation only
                log.warning("Could not set Hindsight bank mission: %s", exc)
            self._missions_set.add(bank)
        self._client.retain(
            bank_id=bank,
            content=content,
            context=context,
            tags=[database_id],
            retain_async=self._retain_async,
        )


def create_memory(settings: Settings) -> tuple[Memory, str | None]:
    """Return (memory, warning). Falls back to NullMemory if Hindsight is unavailable."""
    if not settings.hindsight_api_url:
        return NullMemory(), "Hindsight is not configured (HINDSIGHT_API_URL); memory is off."
    try:
        memory = HindsightMemory(settings)
        memory.check()
    except MemoryUnavailable as exc:
        return NullMemory(), f"{exc} Memory is off for this session."
    return memory, None
