"""Unit tests for graph_memory_service.py (Ocean v3.1 Knowledge Graph & Memory Governance)."""

import os
import tempfile
import pytest

from app.graph_memory_service import GraphMemoryService


@pytest.fixture
def temp_graph_service():
    """Create a temporary SQLite database for testing graph memory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_ocean_graph.db")
        service = GraphMemoryService(db_path=db_path)
        yield service


def test_add_node_and_get_node(temp_graph_service):
    node = temp_graph_service.add_node(
        name="UMass MSCS",
        entity_type="PROGRAM",
        summary="Master of Science in Computer Science at UMass Amherst",
        status="ACTIVE",
    )
    assert node["name"] == "UMass MSCS"
    assert node["entity_type"] == "PROGRAM"
    assert node["status"] == "ACTIVE"

    retrieved = temp_graph_service.get_node("UMass MSCS")
    assert retrieved is not None
    assert retrieved["summary"] == "Master of Science in Computer Science at UMass Amherst"


def test_add_edge(temp_graph_service):
    edge = temp_graph_service.add_edge(
        source_name_or_id="UMass MSCS",
        target_name_or_id="Graduate School Applications",
        relation_type="SUPERSEDES",
    )
    assert edge["relation_type"] == "SUPERSEDES"
    assert edge["source_id"] == "umass_mscs"
    assert edge["target_id"] == "graduate_school_applications"


def test_supersede_nodes(temp_graph_service):
    # Add old application nodes
    temp_graph_service.add_node("Grad School App Prep", entity_type="TASK", summary="Review application essays", status="ACTIVE")
    temp_graph_service.add_node("Grad School Applications", entity_type="PROJECT", summary="Applying to colleges", status="ACTIVE")

    # Add new active state and supersede old ones
    count = temp_graph_service.supersede_nodes(pattern="Grad School", new_entity_name="UMass MSCS Enrolled")
    assert count == 2

    old_node = temp_graph_service.get_node("Grad School App Prep")
    assert old_node["status"] == "SUPERSEDED"

    new_node = temp_graph_service.get_node("UMass MSCS Enrolled")
    assert new_node["status"] == "ACTIVE"

    # Verify active query excludes superseded nodes
    active = temp_graph_service.query_active_knowledge()
    active_names = [n["name"] for n in active]
    assert "UMass MSCS Enrolled" in active_names
    assert "Grad School App Prep" not in active_names


def test_forget_entity(temp_graph_service):
    temp_graph_service.add_node("Old Financial Advice 2024", entity_type="PREFERENCE", summary="Old advice", status="ACTIVE")

    count, names = temp_graph_service.forget_entity("Old Financial Advice")
    assert count == 1
    assert "Old Financial Advice 2024" in names

    forgotten_node = temp_graph_service.get_node("Old Financial Advice 2024")
    assert forgotten_node["status"] == "DEPRECATED"

    active = temp_graph_service.query_active_knowledge()
    assert len(active) == 0


def test_format_graph_context_for_prompt(temp_graph_service):
    temp_graph_service.add_node("Distributed Systems", entity_type="TOPIC", summary="Raft & Paxos consensus")
    ctx = temp_graph_service.format_graph_context_for_prompt()

    assert "Knowledge Graph Active Context" in ctx
    assert "Distributed Systems" in ctx
    assert "Raft & Paxos consensus" in ctx
