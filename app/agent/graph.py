"""The Text-to-SQL LangGraph StateGraph.

START → recall_memory → identify_relevant_schema → generate_sql → validate_sql
validate_sql  → ok: execute_sql | repairable & budget left: repair_sql | else: evaluate_result
execute_sql   → success: evaluate_result | error & budget left: repair_sql | else: evaluate_result
repair_sql    → validate_sql
evaluate_result → retain_memory → END
"""

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import AgentDeps, AgentNodes
from app.agent.state import AgentState
from app.sql.generator import Turn


def build_graph(deps: AgentDeps):
    nodes = AgentNodes(deps)
    g = StateGraph(AgentState)
    g.add_node("recall_memory", nodes.recall_memory)
    g.add_node("identify_relevant_schema", nodes.identify_relevant_schema)
    g.add_node("generate_sql", nodes.generate_sql)
    g.add_node("validate_sql", nodes.validate_sql)
    g.add_node("execute_sql", nodes.execute_sql)
    g.add_node("repair_sql", nodes.repair_sql)
    g.add_node("evaluate_result", nodes.evaluate_result)
    g.add_node("retain_memory", nodes.retain_memory)

    g.add_edge(START, "recall_memory")
    g.add_edge("recall_memory", "identify_relevant_schema")
    g.add_edge("identify_relevant_schema", "generate_sql")
    g.add_edge("generate_sql", "validate_sql")
    g.add_conditional_edges(
        "validate_sql",
        nodes.route_after_validation,
        ["execute_sql", "repair_sql", "evaluate_result"],
    )
    g.add_conditional_edges(
        "execute_sql", nodes.route_after_execution, ["evaluate_result", "repair_sql"]
    )
    g.add_edge("repair_sql", "validate_sql")
    g.add_edge("evaluate_result", "retain_memory")
    g.add_edge("retain_memory", END)
    return g.compile()


class TextToSQLAgent:
    """Runs one question through the graph for a connected database."""

    def __init__(self, deps: AgentDeps):
        self.deps = deps
        self.graph = build_graph(deps)

    def ask(self, question: str, history: list[Turn] | None = None, hint: str | None = None) -> AgentState:
        initial: AgentState = {
            "user_question": question,
            "database_type": self.deps.db.display_name,
            "database_id": self.deps.db.identifier,
            "chat_history": history or [],
            "hint": hint,
            "repair_attempts": 0,
            "failed_attempts": [],
        }
        return self.graph.invoke(initial)
