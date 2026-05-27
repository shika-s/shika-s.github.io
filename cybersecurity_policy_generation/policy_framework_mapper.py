from neo4j import GraphDatabase
from collections import defaultdict
import pandas as pd
import os

try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm

    
NEO4J_URI       = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER      = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS      = os.getenv("NEO4J_PASS")
if NEO4J_PASS is None:
    raise ValueError("NEO4J_PASS environment variable must be set.")
NEO4J_DATABASE  = os.getenv("NEO4J_DATABASE", "POLICY-DB")

# Neo4j Framework Context Extraction
# ============================================================

CYPHER_POLICY_FRAMEWORK_MAP = """
MATCH (f:Framework)
WHERE f.name = $framework_name
MATCH (f)-[:HAS_CONTROL_AREA]->(ca:ControlArea)
MATCH (ca)-[:HAS_SUBCONTROL]->(sc:SubControl)
MATCH (sc)-[:REQUIRES_EVIDENCE]->(er:EvidenceRequirement)
MATCH (p:Policy)-[:SATISFIES_REQUIREMENT]->(er)
WHERE p.title = $policy_name
RETURN DISTINCT
  f.name   AS framework_name,
  p.title  AS policy_title,
  ca.name  AS control_name,
  sc.title AS subcontrol_name,
  sc.code    AS subcontrol_id
ORDER BY control_name, subcontrol_name
"""

def get_framework_context_for_policy(policy_name: str,
                                     framework_name: str = "NIST CSF") -> dict:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            rows = session.run(
                CYPHER_POLICY_FRAMEWORK_MAP,
                framework_name=framework_name,
                policy_name=policy_name,
            ).data()
    finally:
        driver.close()

    # Group by control area
    by_control = defaultdict(list)
    for r in rows:
        by_control[r["control_name"]].append({
            "id": r.get("subcontrol_id"),
            "name": r["subcontrol_name"],
        })

    return {
        "framework": framework_name,
        "policy": policy_name,
        "controls": [
            {
                "control_area": control,
                "subcontrols": subs,
            }
            for control, subs in by_control.items()
        ]
    }

def render_framework_context(framework_data: dict) -> str:
    """
    Turn your framework-control mapping into a readable context block
    for the evaluation prompt.
    """
    lines = []
    print(framework_data)
    lines.append(f"Framework: {framework_data['framework']}")
    lines.append(f"Policy: {framework_data['policy']}")
    lines.append("")

    for ctrl in framework_data.get("controls", []):
        lines.append(f"Control Area: {ctrl['control_area']}")
        for sc in ctrl.get('subcontrols', []):
            lines.append(f"  - {sc['id']}: {sc['name']}")
        lines.append("")  # blank line between control areas

    return "\n".join(lines).strip()