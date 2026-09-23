from langgraph.graph import (
    StateGraph,
    START,
    END
)

from src.agents.state import (
    AutoDSAgentState
)

from src.agents.nodes import (
    problem_agent_node,
    dataset_agent_node,
    quality_agent_node,
    eda_agent_node,
    ml_planning_agent_node,
    synthesis_agent_node
)


# ============================================================
# BUILD AUTODS AGENT GRAPH
# ============================================================

def build_autods_graph():

    graph = StateGraph(
        AutoDSAgentState
    )


    # --------------------------------------------------------
    # ADD AGENTS
    # --------------------------------------------------------

    graph.add_node(
        "problem_agent",
        problem_agent_node
    )

    graph.add_node(
        "dataset_agent",
        dataset_agent_node
    )

    graph.add_node(
        "quality_agent",
        quality_agent_node
    )

    graph.add_node(
        "eda_agent",
        eda_agent_node
    )

    graph.add_node(
        "ml_planning_agent",
        ml_planning_agent_node
    )

    graph.add_node(
        "synthesis_agent",
        synthesis_agent_node
    )


    # --------------------------------------------------------
    # CONNECT WORKFLOW
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "problem_agent"
    )

    graph.add_edge(
        "problem_agent",
        "dataset_agent"
    )

    graph.add_edge(
        "dataset_agent",
        "quality_agent"
    )

    graph.add_edge(
        "quality_agent",
        "eda_agent"
    )

    graph.add_edge(
        "eda_agent",
        "ml_planning_agent"
    )

    graph.add_edge(
        "ml_planning_agent",
        "synthesis_agent"
    )

    graph.add_edge(
        "synthesis_agent",
        END
    )


    # --------------------------------------------------------
    # COMPILE
    # --------------------------------------------------------

    return graph.compile()