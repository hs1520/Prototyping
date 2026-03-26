"""
Retrieval Augmented Generation (RAG) module for MBSE knowledge.

Provides a knowledge base of SysML v2 patterns and best practices,
with retrieval mechanisms to augment LLM prompts with relevant context.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class KnowledgeEntry:
    """A single entry in the MBSE knowledge base."""
    id: str
    title: str
    content: str
    category: str  # e.g., "sysml_pattern", "design_principle", "example"
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# -------------------------------------------------------------------------
# Built-in SysML v2 knowledge base
# -------------------------------------------------------------------------

SYSML_V2_KNOWLEDGE: List[Dict[str, Any]] = [
    {
        "id": "sysml_pkg_001",
        "title": "SysML v2 Package Structure",
        "category": "sysml_pattern",
        "tags": ["package", "namespace", "structure"],
        "content": (
            "A SysML v2 package is the top-level container for model elements. "
            "Use packages to organize related elements:\n\n"
            "```sysml\n"
            "package MySystem {\n"
            "    // Requirements, parts, and connections go here\n"
            "}\n"
            "```\n"
            "Packages can be nested and elements can be imported from other packages."
        ),
    },
    {
        "id": "sysml_part_001",
        "title": "SysML v2 Part Definition",
        "category": "sysml_pattern",
        "tags": ["part def", "block", "component", "structure"],
        "content": (
            "In SysML v2, blocks are defined using 'part def' (part definition):\n\n"
            "```sysml\n"
            "part def Sensor {\n"
            "    attribute samplingRate : Real = 100.0 [Hz];\n"
            "    attribute accuracy : Real = 0.01;\n"
            "    port dataOut : ~SensorPort;\n"
            "}\n"
            "```\n"
            "Parts can have attributes (value properties), ports, actions, and sub-parts."
        ),
    },
    {
        "id": "sysml_port_001",
        "title": "SysML v2 Port Definitions",
        "category": "sysml_pattern",
        "tags": ["port", "interface", "connection"],
        "content": (
            "Ports in SysML v2 define connection points between parts:\n\n"
            "```sysml\n"
            "port def DataPort {\n"
            "    in attribute data : Real;\n"
            "    out attribute timestamp : Real;\n"
            "}\n"
            "part def Controller {\n"
            "    port sensorIn : DataPort;       // conjugated port\n"
            "    port commandOut : ~CommandPort;  // non-conjugated\n"
            "}\n"
            "```\n"
            "The ~ symbol indicates a conjugated port (directions are flipped)."
        ),
    },
    {
        "id": "sysml_req_001",
        "title": "SysML v2 Requirements",
        "category": "sysml_pattern",
        "tags": ["requirement", "constraint", "verification"],
        "content": (
            "Requirements in SysML v2 capture stakeholder needs:\n\n"
            "```sysml\n"
            "requirement SystemPerformance {\n"
            "    doc /* The system shall respond within 100 ms */\n"
            "    attribute responseTime : Real;\n"
            "    require constraint { responseTime <= 100.0 }\n"
            "}\n"
            "```\n"
            "Use 'satisfy' to link design elements to requirements:\n\n"
            "```sysml\n"
            "part def FastController {\n"
            "    satisfy SystemPerformance;\n"
            "}\n"
            "```"
        ),
    },
    {
        "id": "sysml_conn_001",
        "title": "SysML v2 Connections",
        "category": "sysml_pattern",
        "tags": ["connect", "interface", "data flow"],
        "content": (
            "Connections link ports between parts in SysML v2:\n\n"
            "```sysml\n"
            "part system : MySystem {\n"
            "    part ctrl : Controller;\n"
            "    part sensor1 : Sensor;\n"
            "    connect sensor1.dataOut to ctrl.sensorIn;\n"
            "}\n"
            "```\n"
            "Connections ensure data flows are explicitly modeled."
        ),
    },
    {
        "id": "design_001",
        "title": "Sensor-Controller-Actuator Pattern",
        "category": "design_principle",
        "tags": ["cyber-physical", "control", "architecture"],
        "content": (
            "The Sensor-Controller-Actuator (SCA) pattern is fundamental to cyber-physical systems:\n\n"
            "1. Sensors: Measure the physical environment\n"
            "2. Controller: Processes sensor data and computes control actions\n"
            "3. Actuators: Execute control commands in the physical world\n\n"
            "This separates concerns and enables independent testing of each component. "
            "The controller implements the control law that maps sensor readings to actuator commands."
        ),
    },
    {
        "id": "design_002",
        "title": "Layered Architecture for CPS",
        "category": "design_principle",
        "tags": ["architecture", "layers", "separation of concerns"],
        "content": (
            "Cyber-physical systems benefit from a layered architecture:\n\n"
            "1. Physical Layer: Sensors, actuators, physical plant\n"
            "2. Control Layer: Real-time control loops, safety monitors\n"
            "3. Coordination Layer: Mission planning, task allocation\n"
            "4. Application Layer: User interfaces, mission objectives\n\n"
            "Each layer has defined interfaces and communication protocols."
        ),
    },
    {
        "id": "design_003",
        "title": "Redundancy and Fault Tolerance",
        "category": "design_principle",
        "tags": ["safety", "reliability", "redundancy", "fault-tolerance"],
        "content": (
            "For safety-critical cyber-physical systems, redundancy is essential:\n\n"
            "- Triple Modular Redundancy (TMR): Three parallel components, majority voting\n"
            "- Dual Modular Redundancy (DMR): Two components with comparison\n"
            "- Hot Standby: Active backup ready to take over instantly\n"
            "- Cold Standby: Backup that needs time to initialize\n\n"
            "Design consideration: MTBF = MTTF / (1 + MTTF/MTTR)"
        ),
    },
    {
        "id": "dse_001",
        "title": "Design Space Exploration Strategies",
        "category": "design_principle",
        "tags": ["design space", "exploration", "optimization"],
        "content": (
            "Design Space Exploration (DSE) in MBSE involves:\n\n"
            "1. Parameter Space: Discrete or continuous design parameters\n"
            "2. Configuration Space: Different architectural topologies\n"
            "3. Technology Space: Alternative component choices\n\n"
            "Exploration strategies:\n"
            "- Forward exploration: Generate designs then evaluate\n"
            "- Backward inference: Start from requirements, derive designs\n"
            "- Monte Carlo Tree Search: Balance exploration vs exploitation\n"
            "- Pareto-optimal front: Multi-objective optimization"
        ),
    },
    {
        "id": "dse_002",
        "title": "Multi-Objective Design Optimization",
        "category": "design_principle",
        "tags": ["optimization", "Pareto", "trade-off", "quality attributes"],
        "content": (
            "MBSE designs often involve competing objectives:\n\n"
            "Common quality attributes:\n"
            "- Performance: Speed, throughput, latency\n"
            "- Reliability: MTBF, availability, fault tolerance\n"
            "- Cost: Development, production, maintenance\n"
            "- Weight/Size: Physical constraints\n"
            "- Energy: Power consumption, thermal\n\n"
            "Use weighted scoring or Pareto front analysis to navigate trade-offs."
        ),
    },
    {
        "id": "sysml_action_001",
        "title": "SysML v2 Actions and Behaviors",
        "category": "sysml_pattern",
        "tags": ["action", "behavior", "activity", "flow"],
        "content": (
            "Actions in SysML v2 model system behavior:\n\n"
            "```sysml\n"
            "action def SenseAndControl {\n"
            "    in sensorData : Real;\n"
            "    out controlCommand : Real;\n"
            "    \n"
            "    action readSensor { in data : Real; out filtered : Real; }\n"
            "    action computeControl { in filtered : Real; out command : Real; }\n"
            "    \n"
            "    flow from readSensor.filtered to computeControl.filtered;\n"
            "}\n"
            "```\n"
            "Actions can be composed and connected through flows."
        ),
    },
    {
        "id": "cps_example_001",
        "title": "Drone System Architecture",
        "category": "example",
        "tags": ["drone", "UAV", "autonomous", "example"],
        "content": (
            "Example SysML v2 model for an autonomous drone system:\n\n"
            "Key components:\n"
            "- FlightController: Main controller managing flight dynamics\n"
            "- IMU: Inertial Measurement Unit (accelerometer + gyroscope)\n"
            "- GPS: Position and velocity measurement\n"
            "- MotorController: Controls motor speed\n"
            "- Motor (×4): Provides thrust\n"
            "- BatteryManager: Monitors and manages power\n"
            "- MissionPlanner: High-level path and mission planning\n\n"
            "Key requirements: stability (< 0.1° oscillation), precision (< 1m position error), "
            "endurance (> 20 min), safety (auto-land on low battery)."
        ),
    },
    {
        "id": "cps_example_002",
        "title": "Smart Building Management System",
        "category": "example",
        "tags": ["smart building", "HVAC", "IoT", "energy", "example"],
        "content": (
            "Example architecture for a smart building management system:\n\n"
            "Key components:\n"
            "- BuildingController: Central BMS controller\n"
            "- HVAC: Heating, ventilation and air conditioning\n"
            "- OccupancySensor: Detects room occupancy\n"
            "- TemperatureSensor: Monitors ambient temperature\n"
            "- LightingController: Manages lighting zones\n"
            "- EnergyMeter: Tracks energy consumption\n"
            "- SecuritySystem: Access control and monitoring\n\n"
            "Key requirements: comfort (20-22°C), energy efficiency (< 50 kWh/m²/year), "
            "security (access control), occupancy-based control."
        ),
    },
]


class KnowledgeBase:
    """
    MBSE knowledge base for Retrieval Augmented Generation.

    Stores SysML v2 patterns, design principles, and examples.
    Uses TF-IDF based retrieval when vector embeddings are unavailable.
    """

    def __init__(self):
        self.entries: List[KnowledgeEntry] = []
        self._load_builtin_knowledge()

    def _load_builtin_knowledge(self) -> None:
        """Load the built-in SysML v2 knowledge base."""
        for entry_data in SYSML_V2_KNOWLEDGE:
            entry = KnowledgeEntry(
                id=entry_data["id"],
                title=entry_data["title"],
                content=entry_data["content"],
                category=entry_data["category"],
                tags=entry_data["tags"],
            )
            self.entries.append(entry)

    def add_entry(self, entry: KnowledgeEntry) -> None:
        """Add a custom knowledge entry."""
        self.entries.append(entry)

    def search(
        self,
        query: str,
        top_k: int = 3,
        category_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None,
    ) -> List[Tuple[KnowledgeEntry, float]]:
        """
        Search the knowledge base using TF-IDF similarity.

        Returns a list of (entry, score) tuples sorted by relevance.
        """
        candidates = self.entries
        if category_filter:
            candidates = [e for e in candidates if e.category == category_filter]
        if tag_filter:
            candidates = [
                e for e in candidates
                if any(t in e.tags for t in tag_filter)
            ]

        query_terms = self._tokenize(query)
        scored: List[Tuple[KnowledgeEntry, float]] = []
        for entry in candidates:
            score = self._tfidf_score(query_terms, entry)
            scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_by_category(self, category: str) -> List[KnowledgeEntry]:
        """Get all entries of a given category."""
        return [e for e in self.entries if e.category == category]

    def get_by_tags(self, tags: List[str]) -> List[KnowledgeEntry]:
        """Get entries that have any of the specified tags."""
        return [
            e for e in self.entries
            if any(t in e.tags for t in tags)
        ]

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for TF-IDF."""
        import re
        text = text.lower()
        tokens = re.findall(r"\b[a-z][a-z0-9_-]*\b", text)
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "is", "in", "of", "to", "for", "and", "or",
            "with", "that", "this", "it", "be", "are", "was", "have", "has",
        }
        return [t for t in tokens if t not in stop_words]

    def _tfidf_score(self, query_terms: List[str], entry: KnowledgeEntry) -> float:
        """Compute TF-IDF similarity between query and an entry."""
        doc_terms = self._tokenize(
            entry.title + " " + entry.content + " " + " ".join(entry.tags)
        )
        if not doc_terms or not query_terms:
            return 0.0

        doc_freq: Dict[str, int] = {}
        for term in doc_terms:
            doc_freq[term] = doc_freq.get(term, 0) + 1

        score = 0.0
        for term in set(query_terms):
            tf = doc_freq.get(term, 0) / len(doc_terms)
            # Simple IDF approximation
            entries_with_term = sum(
                1 for e in self.entries
                if term in self._tokenize(e.title + " " + e.content)
            )
            if entries_with_term > 0:
                idf = math.log(len(self.entries) / entries_with_term + 1)
            else:
                idf = 0.0
            score += tf * idf

        return score
