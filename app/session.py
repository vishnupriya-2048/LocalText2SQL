"""Open a chat session for a database: connect, read schema, check the model, set up memory."""

from dataclasses import dataclass, field

from app.agent.graph import TextToSQLAgent
from app.agent.nodes import AgentDeps, AgentOptions
from app.config import Settings
from app.database import ConnectionConfig, Database, connect
from app.llm.ollama import OllamaLLM
from app.memory.hindsight import create_memory
from app.schema.extractor import DatabaseSchema


@dataclass
class Session:
    db: Database
    schema: DatabaseSchema
    agent: TextToSQLAgent
    warnings: list[str] = field(default_factory=list)

    def close(self) -> None:
        self.db.close()


def open_session(config: ConnectionConfig, settings: Settings, llm: OllamaLLM) -> Session:
    """Raises DatabaseError or LLMError with a user-facing message on failure."""
    llm.check()
    db = connect(config)
    try:
        schema = db.get_schema()
    except Exception:
        db.close()
        raise
    memory, memory_warning = create_memory(settings)
    options = AgentOptions(max_repair_attempts=settings.max_repair_attempts)
    agent = TextToSQLAgent(AgentDeps(db, schema, llm, memory, settings, options))
    warnings = [memory_warning] if memory_warning else []
    if not schema.tables:
        warnings.append("No tables are visible to this user; questions cannot be answered.")
    return Session(db=db, schema=schema, agent=agent, warnings=warnings)
