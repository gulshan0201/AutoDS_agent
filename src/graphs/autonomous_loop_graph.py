# ============================================================
# AutoDS Phase 4
# Autonomous Feedback Loop Graph
# ============================================================

from langgraph.graph import (
    StateGraph,
    START,
    END
)


from src.agents.state import (
    AutoDSAgentState
)


from src.agents.phase4_nodes import (
    initialize_autonomy_node,
    critic_agent_node,
    experiment_node,
    judge_experiment_node,
    finalizer_node,
    route_after_critic,
    route_after_judge
)


# ============================================================
# BUILD AUTONOMOUS LOOP
# ============================================================

def build_autonomous_loop_graph():

    graph = StateGraph(
        AutoDSAgentState
    )


    # --------------------------------------------------------
    # NODES
    # --------------------------------------------------------

    graph.add_node(
        "initialize",
        initialize_autonomy_node
    )


    graph.add_node(
        "critic",
        critic_agent_node
    )


    graph.add_node(
        "experiment",
        experiment_node
    )


    graph.add_node(
        "judge",
        judge_experiment_node
    )


    graph.add_node(
        "finalizer",
        finalizer_node
    )


    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "initialize"
    )


    graph.add_edge(
        "initialize",
        "critic"
    )


    # --------------------------------------------------------
    # CRITIC ROUTER
    # --------------------------------------------------------

    graph.add_conditional_edges(

        "critic",

        route_after_critic,

        {

            "experiment":
                "experiment",

            "finish":
                "finalizer"
        }
    )


    # --------------------------------------------------------
    # EXPERIMENT → JUDGE
    # --------------------------------------------------------

    graph.add_edge(
        "experiment",
        "judge"
    )


    # --------------------------------------------------------
    # JUDGE ROUTER
    # --------------------------------------------------------

    graph.add_conditional_edges(

        "judge",

        route_after_judge,

        {

            "retry":
                "critic",

            "finish":
                "finalizer"
        }
    )


    # --------------------------------------------------------
    # FINISH
    # --------------------------------------------------------

    graph.add_edge(
        "finalizer",
        END
    )


    return graph.compile()