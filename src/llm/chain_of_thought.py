"""
Chain of Thought (CoT) prompting module.

Implements structured Chain of Thought prompting techniques for guiding
LLMs through complex MBSE design reasoning tasks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .interface import LLMInterface, Message


SYSML_EXPERT_SYSTEM_PROMPT = """You are an expert in Model Based Systems Engineering (MBSE) 
and SysML v2. Your role is to help design cyber-physical systems using SysML v2 notation.

When designing systems:
1. Always think step-by-step, showing your reasoning
2. Follow SysML v2 syntax precisely
3. Consider both functional and non-functional requirements
4. Ensure traceability between requirements and design elements
5. Design for reliability, safety, and maintainability
6. When refining a model, preserve existing valid structure and only change what is needed
   to satisfy missing requirements or remove inconsistencies

SysML v2 key constructs:
- `package`: top-level namespace
- `part def`: defines a block/component type
- `port`: connection point with optional direction (in/out/inout)
- `attribute`: value property with type and unit
- `requirement`: captures stakeholder needs
- `action`: behavior specification
- `connect`: links ports between parts
- `satisfy`: links design elements to requirements
- `refine`: indicates a more concrete model element elaborates an abstract one
- `enum def`: named enumeration type; values accessed as `EnumName::Value` (ALWAYS `::`, never `.`); always declared at package scope; use when a component has distinct named operational modes (e.g. IDLE, ARMED, HOVER)
"""

REQUIREMENTS_COT_TEMPLATE = """Analyze the following system description and extract a complete set of structured requirements.

System Description:
{description}
{fixed_block}

Reason through each category in order. For each, ask: "What is missing that would cause the design to fail?"

1. FUNCTIONAL (FUNC): Core capabilities the system must provide.
   Ask: What inputs must the system accept? What outputs or actions must it produce?

2. PERFORMANCE (PERF): Quantitative bounds on how well it performs.
   Ask: What are the speed, precision, throughput, capacity, and endurance targets?

3. SAFETY (SAFE): Behaviors that prevent harm or handle faults.
   Ask: What must the system do when sensors fail, power is lost, or communication drops?

4. INTERFACE (INTF): External connections the system depends on.
   Ask: Which external systems, sensors, actuators, networks, or data formats must it interoperate with?

5. CONSTRAINTS (CONS): Non-negotiable limits imposed from outside the system.
   Ask: What physical envelope, regulations, cost caps, or standards must be respected?

6. OPERATIONAL MODES (OPER): Named sequential phases that define the system's lifecycle.
   Ask: Does the system have 3 or more clearly distinct operational phases, each with a
   specific name and unambiguous entry condition (e.g. power-on → self-test → armed →
   executing → returning → shutdown)?
   Rules (STRICT):
   - Generate EXACTLY ONE REQ-OPER-001 if yes, ZERO if no meaningful phase structure exists.
     A thermostat cycling on/off is NOT a mode machine. A multi-phase mission vehicle is.
   - Phase names MUST be short ALL-CAPS identifiers separated by →
     (e.g. IDLE → ARMED → TAKEOFF → CRUISE → HOVER → RETURN → LAND).
   - List NOMINAL phases only in the → sequence, from initial power-on to final shutdown/idle.
     The → chain represents the happy-path lifecycle; do NOT append fault states to its end.
   - If SAFE requirements mandate a system-wide fail-safe mode (EMERGENCY / FAULT / FAILSAFE),
     list it OUTSIDE the → chain as a parenthetical annotation, explicitly stating it is
     reachable from any active phase:
       "(EMERGENCY: reachable from any active phase upon detection of a system-wide fault)"
     Rationale: EMERGENCY is not the step after LAND — it is an interrupt that can fire at
     any point in the nominal sequence. Placing it at the end of → implies a false ordering.

Output rules (STRICT — do not deviate):
- One requirement per line, one capability per requirement (atomic — no "and").
- Requirements state WHAT the system must do, never HOW it does it (no implementation details).
- Format exactly: REQ-<CATEGORY>-<NNN>: The {system_name} shall <action> <object> [<condition>]
  CATEGORY ∈ {{FUNC, PERF, SAFE, INTF, CONS, OPER}}.
  NNN assignment: if FIXED REQUIREMENTS exist above, continue numbering from the next available
  number per category (do NOT reuse any ID already present in the fixed list).
  If no fixed requirements: NNN resets to 001 within each category.
- Each requirement must contain "shall" and at least one verifiable criterion
  (numeric value with unit, explicit threshold, or clear boolean trigger condition).
  Exception: REQ-OPER-001 is verified by the completeness of its phase list, not a numeric threshold.
- Aim for completeness: typically 3–6 requirements per category, adjusted to the system's complexity.
  OPER is always 0 or 1 requirement — never more.

Format illustration (replace content with the actual system's domain):
  REQ-FUNC-001: The {system_name} shall <perform primary function A> within <tolerance X>.
  REQ-PERF-001: The {system_name} shall <metric> at a rate of <value> <unit> under <operating condition>.
  REQ-SAFE-001: The {system_name} shall <fail-safe action> when <fault condition> is detected.
  REQ-INTF-001: The {system_name} shall <exchange data/signal> with <external entity> via <protocol/standard>.
  REQ-CONS-001: The {system_name} shall <operate within / comply with> <limit or regulation>.
  REQ-OPER-001: The {system_name} shall operate in sequential phases: IDLE → ARMED → CRUISE → RETURN → LAND → SHUTDOWN (EMERGENCY: reachable from any active phase upon detection of a system-wide fault condition).

After the requirement list, add a short "Dependencies:" section listing any REQ-X depends on REQ-Y pairs.
"""

DESIGN_COT_TEMPLATE = """Design a SysML v2 prototype model for the system below.

System: {system_name}

Requirements (every REQ ID must appear in a satisfy statement in the model):
{requirements}
{context_block}
Work through these steps before writing the model:

1. DECOMPOSITION
   Name each subsystem, its primary responsibility, and which requirements it addresses.
   Each subsystem becomes a part def.

2. INTERFACES
   For each component, list its ports: name, direction (in/out/inout), and carried data type.
   Every INTF requirement must map to at least one named port def and one connect statement.

3. ATTRIBUTES & CONSTRAINTS
   For each component, list measurable properties with types, values, and SI units.
   Every PERF requirement must map to a numeric attribute with an explicit bound and unit.
   Every CONS requirement must appear as a doc annotation or attribute constraint.

4. BEHAVIOR
   For every FUNC requirement: define an action def in the responsible component.
   For every SAFE requirement: define a state def with explicit fault-entry transitions
     and an emergency action def (e.g., emergencyLand, shutdownSafely).

5. TRACEABILITY
   List every REQ-<CATEGORY>-<NNN> and the single component that primarily satisfies it.
   Format (use underscores, not hyphens): REQ_<CATEGORY>_<NNN> → <ComponentName>

Pre-write verification checklist:
  □ Every part def has ≥ 1 port with direction
  □ Every part def has ≥ 1 attribute with numeric value and unit
  □ Every SAFE requirement has a corresponding state def with fault transitions
  □ Every INTF requirement has a named port def and at least one connect usage
  □ Every REQ ID (underscored form) appears in exactly one satisfy statement

Provide the complete SysML v2 model in a single ```sysml code block. No prose after the block.
"""

EVALUATION_COT_TEMPLATE = """Evaluate how well this SysML v2 model satisfies the requirements.

Model:
{model}

Requirements:
{requirements}

Evaluate step by step:
1. COMPLETENESS: Does the design address all requirements?
2. CONSISTENCY: Are there contradictions or gaps?
3. PERFORMANCE: Will the design meet performance requirements?
4. SAFETY: Are safety requirements satisfied?
5. OVERALL SCORE: Provide a score from 0.0 to 1.0

Provide scores in JSON format at the end:
```json
{{"completeness": 0.0, "consistency": 0.0, "performance": 0.0, "safety": 0.0, "overall": 0.0}}
```
"""

REFINEMENT_COT_TEMPLATE = """Refine the SysML v2 model below to fix the reported issues.

PRESERVE all valid structure — only change what is broken or missing.
Do NOT restructure parts that already satisfy their requirements correctly.

Current Model:
{model}

Evaluation Issues:
{feedback}

Specific items to fix:
{issues}

Work through these steps before writing the model:

1. TRIAGE
   For each issue, identify: (a) which part def is responsible, (b) what is missing or wrong,
   (c) the minimal change needed. Do not change unrelated parts.

2. CATEGORY-SPECIFIC FIXES
   - Untraced requirement (REQ_X_NNN has no satisfy): add `satisfy requirement REQ_X_NNN;`
     inside the responsible part def body. Use underscore form, not hyphens.
     NEVER use `satisfy REQ_X by PartName;` — that form breaks the parser.
   - Missing port: add `<direction> port <name> : <Type>;` with in/out/inout.
   - Missing numeric attribute: add `attribute <name> : Real = <value> [<unit>];`
   - SAFE req without fault behavior: add inside the responsible part def:
       action def emergencyStop {{ }}    // top-level action def in the part body
       state def <Name>Monitor {{
           state nominal;               // nominal state has NO entry action
           state fault {{ entry action stop : emergencyStop; }}
           transition initial then nominal;   // ALWAYS point to nominal, never fault
           transition <name>Fault
               first nominal
               if <faultCondition>     // <faultCondition> MUST be declared as attribute in this part
               then fault;
       }}
     SAFETY MONITOR RULES (violations cause dead state machines — check before writing):
       ✓ transition initial then nominal;    // initial → nominal (no entry action)
       ✗ transition initial then fault;      // WRONG — machine stuck in fault at start, transition can never fire
       ✓ if batteryCharge < 15.0            // OK — 'batteryCharge' declared as attribute in owner part
       ✗ if distanceToWaypoint > 1.0        // WRONG — if 'distanceToWaypoint' not declared as attribute,
                                            // syside rejects it and the simulator cannot drive it
     SYNTAX RULES (SysML v2 canonical — verified against the official examples corpus):
       ✓ transition <name> first <source> [accept <event>] [if <guard>] [do <effect>] then <target>;
       ✗ transition <name> from <source> to <target> when <guard>;       (NOT canonical — use first/if/then)
       ✗ transition <name> -> <target>;                                  (NOT canonical — `->` is for succession)
       ✓ entry action <localName> : <ExistingActionDef>;                 (entry references an action def)
       ✗ entry action def emergencyStop {{ }}                              (NOT canonical — define action def
                                                                          at part-def top level, then reference)
       ✓ transition initial then <state>;                                (canonical initial pseudo-transition)
       ✗ transition <name>Init from entry to <state>;                    (use `transition initial then ...` instead)
   - INTF req without connect: add `connect <partA>.<portA> to <partB>.<portB>;`
     at package level (outside part defs).  SysML v2 uses dot notation for
     connect endpoints, NOT `::` (which is the namespace-qualified-name operator).

3. CONSISTENCY CHECK
   After applying fixes, verify:
   □ Every part def still has ≥ 1 port with direction
   □ Every part def still has ≥ 1 attribute with numeric value and unit
   □ Every SAFE requirement maps to a state def with fault transition
   □ Every INTF requirement maps to a port def and a connect usage
   □ Every REQ ID (underscore form) appears in exactly one satisfy statement
   □ No duplicate element names introduced

Provide the complete refined model in a single ```sysml code block. No prose after the block.
"""

# ---------------------------------------------------------------------------
# Multi-step design generation templates (Phase 2-a)
# ---------------------------------------------------------------------------

ARCHITECTURE_DECOMPOSITION_TEMPLATE = """You are decomposing a system into its top-level architectural components.

System: {system_name}

Requirements:
{requirements}
{context_block}
For each requirement category, identify which subsystems are responsible:
  FUNC  → which component performs the function?
  PERF  → which component owns the measurable bound?
  SAFE  → which component enforces the fail-safe behavior?
  INTF  → which component owns each external connection?
  CONS  → which component is most constrained?
  OPER  → which component owns the operational mode state machine?
          (If REQ-OPER-001 is present, name the owning component and list its mode attribute
          e.g. "FlightController owns flightMode : FlightMode")

Output a numbered component list. For each component write exactly:
  <N>. <ComponentName> — <one-sentence primary responsibility>
     Addresses: <comma-separated REQ IDs>
     Ports needed: <comma-separated port names with direction (in/out/inout)>
     Key attributes: <comma-separated attribute names with SI units>

Rules:
- Component names MUST be PascalCase (no spaces, no hyphens).
- Every REQ ID must appear in at least one "Addresses:" line.
- Aim for 3–7 top-level components; avoid micro-splitting single responsibilities.
- Do not write any SysML syntax yet — plain structured text only.
- Safety interconnect ports (MANDATORY when these component types appear):
    • If a SafetyMonitor (or similar safety-enforcement component) is listed:
        – The main controller/autopilot component MUST include `in overrideCmd` in its port list.
        – SafetyMonitor MUST include `out overrideCmd` in its port list.
    • If a CommunicationSystem (or comms/link component) is listed:
        – It MUST include `out commStatus` in its port list.
        – SafetyMonitor MUST include `in commStatus` in its port list.
    • If a PerceptionSystem (or sensor/IMU/camera component) is listed:
        – It MUST include `out sensorStatus` in its port list.
        – SafetyMonitor MUST include `in sensorStatus` in its port list.
"""

PART_DEFINITIONS_TEMPLATE = """Generate the SysML v2 structural fragment for the system below.
Write ONLY part definitions, port definitions, and attributes — no action def, no state def, no connect, no satisfy yet.

System: {system_name}

Architecture plan:
{architecture}

Requirements (structural focus — PERF and INTF):
{requirements}
{context_block}
Rules:
- One part def per component listed in the architecture plan.
- Every part def MUST have:
    • ≥ 1 port with explicit direction (in / out / inout)
    • ≥ 1 attribute with numeric default value and SI unit
- Port names must match those listed in the architecture plan.
- PERF requirements must appear as attributes with numeric bounds and units.
- INTF requirements must appear as port definitions with matching data types.
  For INTF requirements that name a specific external protocol, use a protocol-derived port
  type name (NOT DataPort) so Step 3 can define the matching typed port def:
    • "MAVLink"                    → MAVLinkPort
    • "ADS-B" / "ADS-B Out"       → ADSBOutPort
    • "AES-256" / "encrypted"      → AES256Port
    • "CAN" / "CAN bus"            → CANPort
    • "I2C"                        → I2CPort
    • "USB"                        → USBPort
    Pattern: <Protocol>Port in PascalCase. Internal inter-component ports keep DataPort.
- If the architecture plan mentions a component with distinct operational modes (REQ-OPER):
  do NOT declare the mode attribute and do NOT generate any `enum def` here.
  The behavioral step (Step 4) is solely responsible for defining the enum and
  injecting the mode attribute via `// ATTR OWNER:`. Leave no placeholder — just omit it.
- Structural/passive parts (Airframe, Chassis, Frame, Fuselage, Housing, etc.) MUST use
  only `DataPort` for their ports — never protocol-derived types (MAVLinkPort, etc.).
  This ensures the connectivity fixer can always wire them to a power or environmental source.
- Use valid SysML v2 syntax throughout.
- Safety interconnect ports (MANDATORY — add these whenever the component type is present):
    • If a SafetyMonitor part def is defined:
        – The main controller/autopilot part def MUST declare `in port overrideCmd : DataPort;`
        – SafetyMonitor MUST declare `out port overrideCmd : DataPort;`
    • If a CommunicationSystem part def is defined:
        – CommunicationSystem MUST declare `out port commStatus : DataPort;`
        – SafetyMonitor MUST declare `in port commStatus : DataPort;`
    • If a PerceptionSystem (or sensor/IMU) part def is defined:
        – PerceptionSystem MUST declare `out port sensorStatus : DataPort;`
        – SafetyMonitor MUST declare `in port sensorStatus : DataPort;`

RUNTIME STATE VARIABLE RULE (MANDATORY):
  Attributes fall into two distinct categories — keep them clearly separated:

  (a) Design parameters (readonly): fixed limits set at design time, never change at runtime.
      Use the `readonly` keyword.  Names typically contain max/min/limit/threshold.
        readonly attribute maxAltitude_m     : Real = 120.0 [m];
        readonly attribute releaseTime_s     : Real = 2.0   [s];
        readonly attribute controlFreq_Hz    : Real = 100.0 [Hz];

  (b) Runtime state variables: values that change during system operation.
      No `readonly` keyword.  Initial value is the safe starting point.
        attribute currentAltitude_m  : Real = 0.0   [m];
        attribute currentAirspeed    : Real = 0.0   [m_s];

  For every performance/limit requirement, declare BOTH the design parameter AND
  its runtime counterpart so that assert constraints can link them:

      readonly attribute maxAltitude_m    : Real = 120.0 [m];   // limit (constant)
      attribute currentAltitude_m : Real = 0.0   [m];   // runtime state (varies)

  Exception — safety fault variables (batteryCharge_pct, commLossTime_s, etc.) are
  ALREADY runtime state variables; do NOT add a separate readonly limit for them —
  the state machine guard threshold IS the limit.

Output a single ```sysml code block containing ONLY the structural fragment (no package wrapper yet).
No prose after the block.
"""

INTERFACE_FLOW_TEMPLATE = """Generate the SysML v2 interface and flow fragment for the system below.
Write ONLY item definitions and typed port definitions — no part def body, no attribute, no action def, no state def, no connect, no satisfy yet.

System: {system_name}

Architecture plan:
{architecture}

Structural fragment (for reference — ports already exist, do NOT repeat them):
{parts_fragment}

Interface requirements (INTF):
{intf_requirements}

Rules:
- For every unique type of data/signal exchanged between components, define one `item def`.
  Name item defs as noun phrases in PascalCase (e.g. `GNSSPositionData`, `BatteryStatus`).
- For every port type used across part defs (DataPort, RfPort, etc.), define a typed
  `port def` that references the correct `item def`.
  Example:
      item def GNSSPositionData {{ attribute lat : Real; attribute lon : Real; }}
      port def GNSSPort {{ in item signal : GNSSPositionData; }}
- If an INTF requirement names a specific protocol (MAVLink, ADS-B, I2C, etc.),
  define an item def capturing that protocol's data payload.
- Do NOT define more than one item def per logical data type — reuse where possible.
- Item def names must NOT clash with part def names or requirement IDs.
- The structural fragment may already reference protocol-derived port type names
  (e.g. `MAVLinkPort`, `ADSBOutPort`, `AES256Port`). Define a `port def` for EACH
  such name exactly as it appears in the structural fragment — do NOT rename them.
  Only introduce new port type names for types not yet referenced in the structural fragment.
- Inside a `port def` body, feature names MUST NOT be SysML v2 reserved keywords.
  Forbidden names: `message`, `frame`, `flow`, `connect`, `item`, `port`, `part`,
  `action`, `state`, `binding`, `succession`, `interface`, `allocation`.
  Use `data`, `signal`, `payload`, `telemetry`, `packet`, `value` instead.
  Example of the WRONG name: `in item message : MAVLinkMsg;`  (message is reserved)
  Example of the CORRECT name: `in item data : MAVLinkMsg;`
- Use valid SysML v2 syntax.

Output a single ```sysml code block containing ONLY the item defs and typed port defs.
No prose after the block.
"""

def _build_profile_block(profile: Optional[Dict[str, Any]]) -> str:
    """
    Build the platform profile injection block for BEHAVIOR_TEMPLATE.

    When profile is None → returns empty string (platform-agnostic mode).
    When profile is provided → returns a constraint block that forces LLM
    to use MAVLink-compatible command names in mode machine transitions.

    Profile keys:
      platform      : str                 e.g. "ArduPilot Copter"
      mode_vocabulary: List[str]          e.g. ["CMD_RTL", "CMD_LAND", ...]
      emergency_mode : str                e.g. "CMD_LAND"
    """
    if not profile:
        return ""

    platform     = profile.get("platform", "unknown platform")
    vocabulary   = profile.get("mode_vocabulary", [])
    emergency    = profile.get("emergency_mode", "")

    if not vocabulary:
        return ""

    vocab_str = ", ".join(vocabulary)
    emrg_line = (
        f"\n  Emergency / override command (MANDATORY): {emergency}"
        if emergency else ""
    )

    return f"""
PLATFORM PROFILE — {platform} (MANDATORY — overrides default naming):
  When generating mode machine accept transitions, command names MUST be
  chosen exclusively from the vocabulary below. Do NOT invent new names.

  Allowed accept commands: {vocab_str}{emrg_line}

  Rule: `accept CMD_X` in SysML maps directly to ArduPilot SET_MODE X
  (remove the CMD_ prefix).  This enables automated SITL verification.

  Correct example:
      transition toRTL
          first PhaseReturnState
          accept CMD_RTL
          then PhaseLandingState;

  WRONG (free naming, breaks SITL):
      transition toRTL
          first PhaseReturnState
          accept ReturnToBaseCmd      // ← not in vocabulary, skip SITL test
          then PhaseLandingState;
"""


BEHAVIOR_TEMPLATE = """Generate the SysML v2 behavioral fragment for the system below.
Write ONLY action definitions and state definitions — no part def, no port, no attribute, no connect, no satisfy yet.

System: {system_name}

Architecture plan:
{architecture}

Structural fragment (for reference — do not repeat):
{parts_fragment}

Behavioral requirements (FUNC and SAFE):
{behavioral_requirements}
{platform_profile_block}
Rules:
- For every FUNC requirement: define an action def that belongs to the responsible component.
  The action def name must be a verb phrase in camelCase (e.g., navigateToWaypoint).
- For every SAFE requirement: see SAFETY ARCHITECTURE RULE below.
  When SAFE requirements define a priority ordering (keywords: "superseding",
  "taking precedence", "unless a higher-priority response is already in progress"),
  use the TWO-LAYER pattern described in SAFETY ARCHITECTURE RULE.
  Otherwise (no priority language) use a single independent fault monitor.
- For every OPER requirement (REQ-OPER-NNN): read the phase sequence from the requirement
  text and apply the MODE MACHINE RULES below to generate an enum def + mode machine state def.
  The enum def goes at package scope (// OWNER: package); the state def goes in the owning
  component identified by the architecture plan (// OWNER: <ComponentName>).
- Action def, state def, and enum def names must be unique across the fragment.
- Use valid SysML v2 syntax.

MODE MACHINE RULES — Generate a mode machine when ANY of these triggers is present:
  (a) PRIMARY: A `REQ-OPER-NNN` requirement exists in the behavioral requirements list.
      Read the phase sequence directly from its text (e.g. "IDLE → ARMED → CRUISE → LAND").
      This is the authoritative trigger — Phase 1 has already determined the phase structure.
  (b) FALLBACK: A FUNC requirement explicitly describes phased or staged operation
      (use only when no REQ-OPER is present and the phased intent is unambiguous).
Do NOT generate a mode machine when neither trigger is present.
Safety fault monitors are the default for SAFE requirements — do NOT replace them with mode machines.

When a mode machine is needed:
1. Declare an `enum def` at PACKAGE scope.  Use the sentinel `// OWNER: package` so the
   assembler places it alongside item defs and requirement defs, NOT inside any part def.
   Values use plain identifiers (no `::` inside the enum def body):

     // OWNER: package
     enum def <Name>Mode {{
         enum <MODE_A>;
         enum <MODE_B>;
         enum <MODE_C>;
     }}

2. Declare the corresponding mode attribute on the owning part.  Annotate it so the
   assembler adds it to the structural fragment if absent:

     // ATTR OWNER: <PartName>
     // attribute <modeName> : <Name>Mode = <Name>Mode::<MODE_A>;

   (This comment line is a hint — the assembler will inject the attribute into the part def.)

3. Define the mode-machine state def.  Transitions are driven by EXTERNAL COMMANDS
   (accept actions), NOT by guard conditions on the owner's own mode attribute.

   MODE MACHINE GUARD RULE (STRICT):
   - NEVER write `if <modeName> == <Name>Mode::<X>` in a mode machine transition.
     `<modeName>` is an attribute that represents the state machine's CURRENT state.
     Using it as a guard is a circular self-reference: the machine cannot move to
     state X by checking whether it is already in state X.
   - Use `accept <CommandDef>` to model transitions commanded from outside
     (e.g. an operator arm command, a GCS takeoff command).
   - For fault/contingency exits (EMERGENCY), use a guard on an EXTERNAL sensor
     value owned by a DIFFERENT part (e.g. `batteryCharge`, `commLossTime`),
     or delegate entirely to a dedicated SafetyMonitor state machine.
   - SEND RULE (MANDATORY when SafetyMonitor is present): Every SafetyMonitor fault
     entry action that is intended to override the mode machine MUST close the causal
     loop by sending the matching accept command through the override port.
     The command name MUST match the accept trigger on the mode machine's emergency
     transition exactly (e.g. if the transition uses `accept CmdToEmergency`, the
     action body must send `CmdToEmergency()`).

     Required pattern:
         // OWNER: SafetyMonitor
         action def initiateBatteryRtb {{
             send CmdToEmergency() to overrideCmd;
         }}
         state def BatteryRtbSafetyBehavior {{
             ...
             state BatteryRtbFault {{
                 entry action onFault : initiateBatteryRtb;
             }}
         }}

     This makes the cross-component causal chain explicit and machine-verifiable.
     Without `send`, the link between SafetyMonitor fault and mode machine emergency
     exists only as a conceptual convention — not in the model.

   ENTRY ACTION RULE (MANDATORY):
   - Nominal phase states MUST NOT have entry actions.
     Entry actions belong ONLY on emergency/fault target states (states entered when a fault fires).
     Nominal phases are operational markers — they do not execute actions on entry.
   - Correct: emergency target state has entry action; all nominal phases do not.

   Template:

     // OWNER: package
     action def <Name>To<MODE_B>Cmd {{}}   // command type — declared once at package scope

     // OWNER: <PartName>
     state def <Name>ModeMachine {{
         state <Name><MODE_A>State;          // nominal — NO entry action
         state <Name><MODE_B>State;          // nominal — NO entry action
         state <Name>EmergencyState {{
             entry action respond : <emergencyActionDef>;   // ONLY emergency states have entry actions
         }}
         transition initial then <Name><MODE_A>State;
         transition <name>ToB
             first <Name><MODE_A>State
             accept <Name>To<MODE_B>Cmd
             then <Name><MODE_B>State;
         transition <name>ToEmergency
             first <Name><MODE_A>State
             accept <EmergencyCmdDef>
             then <Name>EmergencyState;
     }}

SAFETY ARCHITECTURE RULE — Two-layer pattern for priority-ordered SAFE requirements:

  Trigger: ANY of the SAFE requirements uses priority language:
    "superseding", "taking precedence over all", "unless a higher-priority response
    is already in progress", or any explicit ordering between safety responses.

  When triggered, generate exactly TWO structural elements inside the SafetyMonitor:

  ── LAYER 1: Fault Detection Monitors ──────────────────────────────────────────
  One state def per SAFE requirement. Each monitor:
    • Detects its fault condition via a guard transition.
    • Has a fault state with an entry action for REQUIREMENT-SPECIFIC side effects
      only (e.g., transmitting an alert, locking a payload). See examples below.
    • MUST NOT send override commands to `overrideCmd` directly.
      Command dispatch is the SOLE responsibility of the SafetyArbiter (Layer 2).

  Examples of allowed monitor actions:
    – transmit alert to GCS (`send alertCmd() to somePort;`)
    – lock payload (`send lockCmd() to payloadPort;`)
    – log fault

  Examples of FORBIDDEN monitor actions:
    – `send CMD_LAND() to overrideCmd;`   ← FORBIDDEN: arbiter's job
    – `send CMD_RTL() to overrideCmd;`    ← FORBIDDEN

  ── LAYER 2: Safety Arbiter ────────────────────────────────────────────────────
  Exactly ONE `SafetyArbiter` state def per SafetyMonitor.
    • States are ordered lowest → highest priority.
    • Each fault state has the entry action that sends the appropriate override command.
    • Priority is enforced via guard exclusions: a lower-priority state's guard
      MUST include `and not <higher-priority-condition>` for EVERY higher-priority
      condition above it.
    • The highest-priority state needs NO exclusions in its guard.

  Canonical template (adapt priority count and conditions to requirements):

    // OWNER: SafetyMonitor
    action def initiateBatteryRtb    {{ send CMD_RTL()  to overrideCmd; }}
    action def initiateEmergencyLand {{ send CMD_LAND() to overrideCmd; }}
    action def deployParachute       {{ send CMD_LAND() to parachutePort; }}

    // Layer 1 monitors — fault detection only, NO override commands
    // OWNER: SafetyMonitor
    state def BatteryRtbMonitor {{
        state RtbNominal;
        state RtbDetected;        // no entry action needed here — arbiter handles response
        transition initial then RtbNominal;
        transition RtbNominal → RtbDetected if batterySoc <= 25.0;
    }}

    // OWNER: SafetyMonitor
    state def BatteryLandMonitor {{
        state LandNominal;
        state LandDetected;
        transition initial then LandNominal;
        transition LandNominal → LandDetected if batterySoc < 15.0;
    }}

    // OWNER: SafetyMonitor
    state def PropulsionFailureMonitor {{
        state PropNominal;
        state PropDetected;
        transition initial then PropNominal;
        transition PropNominal → PropDetected if propulsionCriticalFailure;
    }}

    // Layer 2 arbiter — SOLE sender of override commands, priority ordered
    // OWNER: SafetyMonitor
    state def SafetyArbiter {{
        state ArbNominal;

        // Priority 1 (lowest): Battery RTB
        // Guard excludes all higher-priority conditions
        state ArbRtbMode {{
            entry action onRtb : initiateBatteryRtb;
        }}

        // Priority 2: Emergency land (battery critical OR comm loss)
        // Guard excludes parachute condition (priority 3)
        state ArbLandMode {{
            entry action onLand : initiateEmergencyLand;
        }}

        // Priority 3 (highest): Parachute deploy
        // No exclusions needed — overrides everything
        state ArbParachuteMode {{
            entry action onParachute : deployParachute;
        }}

        transition initial then ArbNominal;

        // Highest priority first — guard has no exclusions
        transition ArbNominal → ArbParachuteMode
            if propulsionCriticalFailure;

        // Second priority — exclude parachute condition
        transition ArbNominal → ArbLandMode
            if batterySoc < 15.0
            and not propulsionCriticalFailure;

        transition ArbNominal → ArbLandMode
            if commLossTime > 10.0
            and not propulsionCriticalFailure;

        // Lowest priority — exclude all higher-priority conditions
        transition ArbNominal → ArbRtbMode
            if batterySoc <= 25.0
            and batterySoc >= 15.0
            and commLossTime <= 10.0
            and not propulsionCriticalFailure;
    }}

  KEY RULES for the SafetyArbiter:
  1. ALL override command sends MUST be in the arbiter's entry actions, NEVER in monitors.
  2. List higher-priority transitions BEFORE lower-priority ones in the state def.
  3. Each lower-priority guard MUST contain `and not <higher-priority-condition>`.
  4. Use the same runtime attributes (batterySoc, commLossTime, etc.) as the monitors;
     do NOT introduce separate flag attributes.

GUARD CONDITION RULES — the `if <faultCondition>` expression decides whether the
fault transition can ever fire.  A guard that is logically impossible produces a
dead state machine.  Follow these rules exactly:
- Use a threshold-CROSSING comparison operator: `<`, `<=`, `>`, `>=`.
  NEVER use `==` or `!=` for a numeric condition — a fault is "value crossed a
  limit", not "value exactly equals a number".  A continuously-changing quantity
  almost never lands on an exact value, so an `==` guard never triggers.
  ENUM GUARD EXCEPTION — `==` IS allowed when the left-hand side is an enum-typed
  attribute and the right-hand side is a qualified `EnumName::Value` literal.
  The type prefix is MANDATORY — bare `== HOVER` is not recognised:
    ✓  if operatingRegion == RegionMode::URBAN   // external enum attribute — allowed
    ✗  if operatingRegion == URBAN               // missing EnumType:: prefix
    ✗  if operatingRegion == "URBAN"             // string form — not valid SysML v2
  ENUM GUARD SELF-REFERENCE BAN — do NOT use a part's own current-mode attribute
  as a guard inside THAT SAME part's mode machine (see MODE MACHINE GUARD RULE):
    ✗  if flightMode == FlightMode::HOVER        // circular — mode machine owns flightMode
- The LEFT operand must be a DYNAMIC measured / sensed state variable
  (e.g. batteryCharge, commLossTime, obstacleDistance, tiltAngle).
  The RIGHT operand is the threshold. PREFER a dynamic threshold — another
  attribute, or an arithmetic expression of attributes — whenever the limit
  actually depends on operating conditions; use a bare numeric literal only
  when the limit is a genuinely fixed constant.
    e.g. `batteryCharge <= returnEnergyRequired`   (limit depends on distance)
         `commLossTime > heartbeatInterval + 5.0`  (limit derived from a param)
  NEVER compare a threshold against its own value
  (e.g. `rtbBatteryThreshold == 120.0` is meaningless — it is always true/false).
- For a boolean fault flag, the fault fires when the flag is TRUE.  Name flags
  affirmatively (sensorSelfTestFailed, collisionDetected, linkLost) and write the
  guard as the bare flag name:  `if sensorSelfTestFailed`
  (NOT `== false`, NOT `== true` — the extractor only recognises the bare flag).
- Every variable named in a guard must read as a runtime state of the owner part
  (it will be declared as a backing attribute in the assembly step).

  ✓ CORRECT guards (safety monitors — external sensor/flag owned by THIS or ANOTHER part):
      if batteryCharge < 15.0                   // fixed numeric threshold
      if commLossTime > 10.0
      if sensorSelfTestFailed                   // boolean flag — fires when true
      if batteryCharge <= returnEnergyRequired  // dynamic threshold (RHS is an attribute)
      if commLossTime > heartbeatInterval + 5.0 // arithmetic threshold
  ✗ WRONG guards:
      if batteryLevel == 15.0          // `==` on a swept value never fires
      if rtbBatteryThreshold == 120.0  // comparing a threshold to itself
      if sensorStatus == false         // use an affirmative flag instead
      if flightMode == HOVER           // missing EnumType:: prefix — not recognised
      if flightMode == FlightMode::HOVER  // CIRCULAR — mode machine must not guard on
                                          // its own owner's current-mode attribute;
                                          // use `accept <CommandDef>` instead

CRITICAL OWNERSHIP ANNOTATION — you MUST prefix every action def, state def, and enum def
with a comment naming where the element belongs.  The assembly step uses these annotations
to place each element correctly.  Missing annotations cause elements to be lost.

  • Part-owned elements (action def, state def): `// OWNER: <PartName>`
  • Package-scoped elements (enum def):          `// OWNER: package`
  • Mode attribute hints:                        `// ATTR OWNER: <PartName>`
    followed by:  `// attribute <name> : <EnumType> = <EnumType>::<initial>;`
    ENUM QUALIFIER RULE — enum values ALWAYS use `::`, never `.`:
      ✓  attribute flightMode : FlightPhase = FlightPhase::POWER_ON;
      ✗  attribute flightMode : FlightPhase = FlightPhase.POWER_ON;   // dot is NOT valid SysML v2
      ✓  if operatingMode == DronePhaseMode::CRUISE                   // guard: :: required
      ✗  if operatingMode == DronePhaseMode.CRUISE                    // dot fails at parse time

Format (use exactly this style — no other comment form).  All transitions and
entry actions follow canonical SysML v2 syntax (verified against the official
examples corpus):

  // OWNER: <PartName>
  action def doSomething {{ }}

  // OWNER: <PartName>
  action def <name>EmergencyResponse {{ }}     // emergency action lives at this level

  // OWNER: <PartName>
  state def <Name>SafetyBehavior {{
      state <Name>Nominal;                       // nominal = no entry action
      state <Name>Fault {{
          entry action onFault : <name>EmergencyResponse;     // reference, not inline def
      }}
      transition initial then <Name>Nominal;     // ALWAYS point to the NOMINAL state, NEVER the fault state
      transition <name>Fault                     // named fault transition
          first <Name>Nominal                    // source state (NOT `from`)
          if measuredVar < limitValue            // threshold-crossing guard (NOT `==`, NOT `when`)
          then <Name>Fault;                      // target state (NOT `to`)
  }}

SAFETY MONITOR STRUCTURAL RULES (violations produce dead state machines):
  RULE 1 — Initial state MUST be the nominal state (no entry action).
    ✓  transition initial then <Name>Nominal;     // Nominal has NO entry action
    ✗  transition initial then <Name>Fault;       // WRONG — machine starts in fault, can never observe the transition
    Rationale: the state machine is designed to DETECT a transition from normal to faulty.
    If it starts in the fault state, the fault transition has no source to fire from.

  RULE 2 — Every attribute referenced in a guard MUST be declared in the owning part def.
    ✓  // in part def: attribute batteryCharge : Real = 100.0;
       if batteryCharge < 15.0                   // declared → extractor can drive it
    ✗  if distanceToWaypoint > 1.0               // WRONG — if 'distanceToWaypoint' is not declared
                                                 // as an attribute, syside reports a sema error
                                                 // and the simulator cannot sweep the variable
    Action: for every variable name that appears in a guard condition, verify it exists
    as `attribute <name> : Real = <initial_value>;` in the owning part def before writing the guard.

CRITICAL — DO NOT USE these non-canonical forms (Syside may parse them but they
are not SysML v2 standard):
  ✗  transition X from <state> to <state> when <guard>;     // wrong keywords
  ✗  transition X -> Y when Z;                              // `->` is succession, not transition
  ✗  state Y {{ entry action def localAction {{ }} }}            // inline action def in entry
  ✗  if someValue == <number>;                              // `==` numeric guard never fires — use < <= > >=
  ✗  if someFlag == false;                                  // use an affirmative flag: `if someFlagFailed`
  ✗  if flightMode == HOVER;                               // bare identifier — missing `EnumType::` prefix
  ✗  enum def FlightMode {{ FlightMode::IDLE; ... }}        // values inside enum def use plain names, not qualified form
  ✗  enum def FlightMode {{ IDLE = 0; ARMED = 1; }}         // no integer assignments in SysML v2 enum def

PARAMETRIC CONSTRAINT RULE (MANDATORY):
  For every (readonly design parameter, runtime state variable) pair in the structural
  fragment, generate a matching `assert constraint` block INSIDE the owning part def body.
  Place it after the action defs, before the state defs.

  Pattern:
    // OWNER: <PartName>
    assert constraint <descriptiveName>Bound {{
        <runtime_var> <= <readonly_limit>    // for maximum constraints
    }}
    assert constraint <descriptiveName>Min {{
        <runtime_var> >= <readonly_limit>    // for minimum constraints
    }}

  Required constraint pairs (generate ALL that apply — maximum bounds only):
    currentAltitude_m   <= maxAltitude_m        (altitude fence)
    actualReleaseTime_s <= releaseTime_s         (payload timing)
    currentAirspeed     <= maxAirspeed           (speed envelope)
    payloadMass         <= maxPayloadMass        (payload limit)
    currentWeight       <= maxTakeoffWeight      (MTOW limit)

  Example output:
    // OWNER: FlightController
    assert constraint altitudeBound {{
        currentAltitude_m <= maxAltitude_m
    }}

  Do NOT generate constraints for safety guard variables (batteryCharge_pct,
  commLossTime_s, etc.) — those are already constrained by state machine guards.

  Do NOT generate `assert constraint` for these two categories — they are NOT
  structural invariants and will always fail at system initialisation:

  (a) Time-cumulative quantities: variables that start at 0 and accumulate over
      the mission (e.g. currentFlightTime, missionDuration, flightElapsed).
      A constraint like `currentFlightTime >= minFlightTime` is meaningless as an
      invariant — at t=0 the flight has just started.  These are end-to-end
      performance requirements verified by SITL, not instantaneous bounds.

  (b) Operational-phase-only quantities: variables that are only meaningful
      during a specific phase (e.g. currentGroundSpeed, currentForwardSpeed).
      The system is at rest before takeoff; `currentGroundSpeed >= minGroundSpeed`
      is false on the ground and must not be asserted as an always-true invariant.

  For both categories, add a comment instead:
      // REQ-PERF-NNN: verified by SITL — not an instantaneous invariant

  SUMMARY — assert constraint IS appropriate for:
    ✓  maximum bounds that must NEVER be exceeded at any time
       (currentAltitude <= maxAltitude, currentAirspeed <= maxAirspeed,
        payloadMass <= maxPayloadMass, currentWeight <= maxTakeoffWeight)
  assert constraint is NOT appropriate for:
    ✗  minimum endurance / throughput goals (flightTime >= minFlightTime)
    ✗  minimum speed during a specific flight phase (groundSpeed >= minGroundSpeed)

Output a single ```sysml code block containing ONLY the behavioral fragment (with OWNER comments).
No prose after the block.
"""

INTEGRATION_TEMPLATE = """Assemble the complete SysML v2 model from the fragments below.

System: {system_name}

Structural fragment (part defs, ports, attributes):
{parts_fragment}

Interface & flow fragment (item defs, typed port defs — use these types when annotating ports):
{interfaces_fragment}

Behavioral fragment (each element is annotated with // OWNER: <PartName>):
{behavior_fragment}

All requirements (every REQ ID must appear in exactly one satisfy statement):
{requirements}

Assembly rules:
1. Wrap everything in: package {package_name} {{ ... }}
   At the top of the package body, declare:
     (a) All item defs and typed port defs from the Interface & flow fragment.
     (b) All requirement defs (one per REQ ID using underscore form).
         Use the CORRECT SysML v2 doc-comment syntax:
           requirement def REQ_FUNC_001 {{ doc /* The drone shall ... */ }}
         NEVER use the string-assignment form — it is NOT valid SysML v2:
           ✗  requirement def REQ_FUNC_001 {{ doc = "The drone shall ..."; }}
     (c) All `enum def` elements from the behavioral fragment that are annotated
         `// OWNER: package`.  Enum defs are shared types — they MUST be at package
         scope, NOT inside any part def.
         Example — behavioral fragment contains:
             // OWNER: package
             enum def FlightMode {{ enum IDLE; enum ARMED; enum HOVER; }}
         → Place `enum def FlightMode {{ ... }}` at the top of the package alongside item defs.
2. Apply typed port defs from the Interface & flow fragment:
   Re-annotate every port in each part def to use the typed port def where applicable.
   Example — interfaces fragment contains `port def GNSSPort {{ ... }}`:
     → Change `in port gnssIn : DataPort;` to `in port gnssIn : GNSSPort;`
   If no typed port def matches a particular port, keep its original type.
3. Embed EVERY action def and state def from the behavioral fragment inside its owner part def.
   Use the // OWNER: <PartName> comment to identify which part def each element belongs to.
   CRITICAL: Do NOT drop, skip, or omit any state def from the behavioral fragment.
             Every state def in the behavioral fragment MUST appear in the final model.
   Elements annotated `// OWNER: package` go at package scope (rule 1c), not in any part def.
   For every `// ATTR OWNER: <PartName>` hint line in the behavioral fragment, inject the
   accompanying attribute declaration into that part def if it is not already present.
   Example — behavioral fragment contains:
       // OWNER: SafetyMonitor
       state def BatterySafetyBehavior {{ state BattNominal; state BattFault {{ entry; }} ... }}
   → Embed inside part def SafetyMonitor {{ ... }}.
   Example — behavioral fragment contains:
       // ATTR OWNER: FlightController
       // attribute flightMode : FlightMode = FlightMode::IDLE;
   → If part def FlightController does not already have `attribute flightMode`, add it.
4. Add connect statements for ALL inter-component data flows — not just INTF requirements.
     connect <partA>.<portA> to <partB>.<portB>;
   SYNTAX (critical — SysML v2 uses dot notation, NOT `::`):
     ✓  connect engine.drivePort to transmission.clutchPort;
     ✗  connect engine::drivePort to transmission::clutchPort;   (wrong — `::` is for
         qualified namespace names like `Package::Element`, not connect endpoints)
   MINIMUM COVERAGE (mandatory — every item below must have a connect statement):
   (a) Every INTF requirement must map to at least one connect.
   (b) Every safety-related port pair MUST be connected:
         commStatus     : CommunicationSystem.commStatus     → SafetyMonitor.commStatus
         sensorStatus   : PerceptionSystem.sensorStatus      → SafetyMonitor.sensorStatus
         overrideCmd    : SafetyMonitor.overrideCmd          → <MainController>.overrideCmd
   (c) Connect all semantically paired out→in ports (matching base names, e.g.
       `gnssOut → gnssIn`, `battStatusOut → battStatusIn`, `telemetryOut → telemetryIn`).
   (d) Connect the main controller to the communication system for uplink/downlink.
   RULE: For a system with N part defs, include at least max(N − 1, INTF_req_count) connects.
   Do NOT leave any out port without at least one connect to a consumer.
   FAN-IN PROHIBITION (critical): Each `in port` must receive from exactly ONE source.
   Never write two connect statements that both target the same port.
     ✗ WRONG (fan-in):
         connect flightController.telemetryData to commSystem.telemetryData;
         connect payloadManager.payloadStatus   to commSystem.telemetryData;   ← same target!
     ✓ RIGHT (route through aggregator):
         connect payloadManager.payloadStatus   to flightController.payloadStatusIn;
         connect flightController.telemetryData to commSystem.telemetryData;
   Sub-system status ports (payloadStatus, batteryStatus, etc.) MUST connect to the
   main controller or a dedicated aggregator — NOT directly to a communication port.
5. Add one satisfy statement PER requirement INSIDE the owning part def, using underscore form:
     satisfy requirement REQ_<CATEGORY>_<NNN>;
   NOT "satisfy REQ_X by PartName" — that form is forbidden and breaks the parser.
6. Every part def must retain its ports and attributes from the structural fragment unchanged.
7. Do not introduce new part defs — use exactly those from the structural fragment.

Pre-write checklist:
  □ Item defs and typed port defs from the interface fragment are declared at package level
  □ All `// OWNER: package` enum defs from the behavioral fragment are declared at package scope
  □ Every `// ATTR OWNER: <Part>` attribute hint has been injected into the corresponding part def
  □ Every part def has ≥ 1 port (annotated with typed port def where available)
  □ Every part def has ≥ 1 attribute with numeric value and unit
  □ Every state def from the behavioral fragment is embedded in its owner part def (NONE dropped)
  □ Every SAFE requirement maps to a state def with fault transition inside the safety part def
  □ Every INTF requirement maps to a typed port def and a semantically consistent connect usage
  □ Safety port connects present: commStatus, sensorStatus, overrideCmd each have a connect
  □ Total connect count ≥ max(N_parts − 1, INTF_req_count) — no output port left unconnected
  □ Every REQ ID (underscore form) appears in exactly one satisfy statement (inside a part def)

Output the complete model in a single ```sysml code block. No prose after the block.
"""


MCTS_REDUNDANCY_GROUNDING_TEMPLATE = """You have ONE task: add {redundancy_level} hardware redundancy to the SysML v2 model below.

Rules (strict):
- DO NOT remove, rename, or restructure any existing element.
- DO NOT change any port, attribute, action def, existing state def, satisfy link, or connect.
- ONLY add new content inside the target part def.
- Use valid SysML v2 syntax (doc /* */ not doc = ""; state names globally unique).

Target part def: {target_part}

What to add (copy exactly, then adjust state/transition names if needed to avoid duplicates):
{redundancy_instructions}

Current SysML model:
```sysml
{sysml_text}
```

Return the COMPLETE updated model in a single ```sysml code block. No prose before or after.
"""


@dataclass
class ThoughtStep:
    """A single step in a Chain of Thought reasoning process."""
    step_number: int
    description: str
    content: str
    confidence: float = 1.0


@dataclass
class CoTResult:
    """Result of a Chain of Thought reasoning process."""
    final_answer: str
    thought_steps: List[ThoughtStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    extracted_json: Optional[Dict[str, Any]] = None
    extracted_sysml: Optional[str] = None

    def get_scores(self) -> Optional[Dict[str, float]]:
        """Extract evaluation scores from the response."""
        if self.extracted_json:
            return self.extracted_json
        return None


class ChainOfThoughtPrompter:
    """
    Implements Chain of Thought prompting for MBSE design tasks.

    Supports multiple CoT patterns:
    - Zero-shot CoT: "Think step by step"
    - Few-shot CoT: Provide examples with reasoning
    - Tree of Thought: Explore multiple reasoning paths
    - Self-consistency: Generate multiple solutions and vote
    """

    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.system_prompt = SYSML_EXPERT_SYSTEM_PROMPT

    def extract_requirements(
        self,
        system_description: str,
        system_name: str = "system",
        context: str = "",
        fixed_requirements: Optional[List[str]] = None,
    ) -> CoTResult:
        """
        Use CoT prompting to extract structured requirements from a description.

        fixed_requirements: manually written requirements that MUST appear verbatim
        in the output.  The LLM is instructed to include them as-is and continue
        numbering new requirements from the next available ID per category.
        """
        description_block = system_description
        if context:
            description_block += f"\n\nAdditional context: {context}"

        fixed_block = ""
        if fixed_requirements:
            import re as _re
            # Compute next available ID per category from fixed list
            from collections import defaultdict
            max_num: dict = defaultdict(int)
            for req in fixed_requirements:
                m = _re.match(r"REQ-([A-Z]+)-(\d+):", req.strip())
                if m:
                    max_num[m.group(1)] = max(max_num[m.group(1)], int(m.group(2)))

            next_ids = "\n".join(
                f"  {cat}: used up to {n:03d}, your new reqs start at {n+1:03d}"
                for cat, n in sorted(max_num.items())
            )
            fixed_lines = "\n".join(f"  {r}" for r in fixed_requirements)
            fixed_block = (
                "\n\nFIXED REQUIREMENTS — copy these into your output VERBATIM "
                "(exact text, exact ID). Do NOT rephrase, merge, split, or omit any:\n"
                f"{fixed_lines}\n\n"
                f"Next available IDs for NEW requirements you add:\n{next_ids}\n"
            )

        prompt = REQUIREMENTS_COT_TEMPLATE.format(
            description=description_block,
            system_name=system_name,
            fixed_block=fixed_block,
        )
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=prompt),
        ]
        response = self.llm.complete(messages, temperature=0.9)
        return self._parse_cot_response(response.content)

    def generate_design(
        self,
        system_name: str,
        requirements: List[str],
        context: str = "",
        temperature: float = 0.5,
    ) -> CoTResult:
        """
        Use CoT prompting to generate a SysML v2 design.
        """
        req_text = "\n".join(f"  {r}" for r in requirements)
        context_block = (
            f"\nRelevant domain context:\n{context}\n"
            if context and context.strip()
            else ""
        )
        prompt = DESIGN_COT_TEMPLATE.format(
            system_name=system_name,
            requirements=req_text,
            context_block=context_block,
        )
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=prompt),
        ]
        response = self.llm.complete(messages, temperature=temperature)
        return self._parse_cot_response(response.content)

    def evaluate_design(
        self,
        model_text: str,
        requirements: List[str],
    ) -> CoTResult:
        """
        Use CoT prompting to evaluate a SysML v2 model against requirements.
        """
        req_text = "\n".join(f"  - {r}" for r in requirements)
        prompt = EVALUATION_COT_TEMPLATE.format(
            model=model_text,
            requirements=req_text,
        )
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=prompt),
        ]
        response = self.llm.complete(messages, temperature=0.2)
        return self._parse_cot_response(response.content)

    def refine_design(
        self,
        model_text: str,
        feedback: str,
        issues: List[str],
    ) -> CoTResult:
        """
        Use CoT prompting to refine a design based on evaluation feedback.
        """
        issues_text = "\n".join(f"  - {i}" for i in issues)
        prompt = REFINEMENT_COT_TEMPLATE.format(
            model=model_text,
            feedback=feedback,
            issues=issues_text,
        )
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=prompt),
        ]
        response = self.llm.complete(messages, temperature=0.4, max_tokens=65536)
        return self._parse_cot_response(response.content)

    # ------------------------------------------------------------------
    # Multi-step generation methods (Phase 2-a)
    # ------------------------------------------------------------------

    def decompose_architecture(
        self,
        system_name: str,
        requirements: List[str],
        context: str = "",
    ) -> CoTResult:
        """Step 1: Produce a structured component list (plain text, no SysML)."""
        req_text = "\n".join(f"  {r}" for r in requirements)
        context_block = (
            f"\nRelevant domain context:\n{context}\n"
            if context and context.strip()
            else ""
        )
        prompt = ARCHITECTURE_DECOMPOSITION_TEMPLATE.format(
            system_name=system_name,
            requirements=req_text,
            context_block=context_block,
        )
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=prompt),
        ]
        response = self.llm.complete(messages, temperature=0.7)
        return self._parse_cot_response(response.content)

    def generate_part_definitions(
        self,
        system_name: str,
        architecture: str,
        requirements: List[str],
        context: str = "",
    ) -> CoTResult:
        """Step 2: Generate structural SysML fragment (part def / port / attribute)."""
        req_text = "\n".join(f"  {r}" for r in requirements)
        context_block = (
            f"\nRelevant domain context:\n{context}\n"
            if context and context.strip()
            else ""
        )
        prompt = PART_DEFINITIONS_TEMPLATE.format(
            system_name=system_name,
            architecture=architecture,
            requirements=req_text,
            context_block=context_block,
        )
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=prompt),
        ]
        response = self.llm.complete(messages, temperature=0.3)
        return self._parse_cot_response(response.content)

    def generate_behavior(
        self,
        system_name: str,
        architecture: str,
        behavioral_requirements: List[str],
        parts_fragment: str,
        context: str = "",
        platform_profile: Optional[Dict[str, Any]] = None,
    ) -> CoTResult:
        """Step 3: Generate behavioral SysML fragment (action def / state def)."""
        req_text = "\n".join(f"  {r}" for r in behavioral_requirements)
        profile_block = _build_profile_block(platform_profile)
        prompt = BEHAVIOR_TEMPLATE.format(
            system_name=system_name,
            architecture=architecture,
            parts_fragment=parts_fragment,
            behavioral_requirements=req_text,
            platform_profile_block=profile_block,
        )
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=prompt),
        ]
        response = self.llm.complete(messages, temperature=0.4)
        return self._parse_cot_response(response.content)

    def generate_interfaces_and_flows(
        self,
        system_name: str,
        architecture: str,
        parts_fragment: str,
        intf_requirements: List[str],
        context: str = "",
    ) -> CoTResult:
        """Step 3: Generate interface & flow fragment (item def / typed port def)."""
        req_text = "\n".join(f"  {r}" for r in intf_requirements) if intf_requirements else "  (none)"
        prompt = INTERFACE_FLOW_TEMPLATE.format(
            system_name=system_name,
            architecture=architecture,
            parts_fragment=parts_fragment,
            intf_requirements=req_text,
        )
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=prompt),
        ]
        response = self.llm.complete(messages, temperature=0.3)
        return self._parse_cot_response(response.content)

    def assemble_model(
        self,
        system_name: str,
        parts_fragment: str,
        interfaces_fragment: str,
        behavior_fragment: str,
        requirements: List[str],
    ) -> CoTResult:
        """Step 5: Assemble complete SysML package with connections and satisfy links."""
        req_text = "\n".join(f"  {r}" for r in requirements)
        package_name = re.sub(r"[^A-Za-z0-9]", "", system_name) or "System"
        prompt = INTEGRATION_TEMPLATE.format(
            system_name=system_name,
            parts_fragment=parts_fragment,
            interfaces_fragment=interfaces_fragment if interfaces_fragment else "(none — use generic port types)",
            behavior_fragment=behavior_fragment,
            requirements=req_text,
            package_name=package_name,
        )
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=prompt),
        ]
        response = self.llm.complete(messages, temperature=0.2, max_tokens=65536)
        return self._parse_cot_response(response.content)

    def mcts_structural_grounding(
        self,
        redundancy_level: str,
        target_part: str,
        sysml_text: str,
    ) -> CoTResult:
        """Focused single-purpose LLM call: add redundancy structure to an existing model.

        This is intentionally narrow — it does exactly one thing so the LLM
        cannot "get distracted" by other quality concerns.  Called unconditionally
        from the MCTS grounding pass regardless of the model's quality score.
        """
        if redundancy_level == "triple":
            # Canonical SysML v2 form (verified against the official examples
            # corpus): transitions use `first / if / then`; emergencyStop is
            # declared at part-def top level and *referenced* from the entry,
            # not inline-defined inside an entry.  Channel failure signals are
            # grounded as Boolean attributes so transition guards have a real
            # producer; the fail conditions then drive 2-of-3 majority voting.
            instructions = (
                f"Add the following INSIDE part def {target_part} (before its closing brace):\n\n"
                f"    attribute redundancyChannels : Integer = 3;\n"
                f"    attribute channelAFailed : Boolean = false;\n"
                f"    attribute channelBFailed : Boolean = false;\n"
                f"    attribute channelCFailed : Boolean = false;\n\n"
                f"    action def emergencyStop {{ }}\n\n"
                f"    state def TripleChannelRedundancy {{\n"
                f"        state Active;\n"
                f"        state FailsafeActive {{\n"
                f"            entry action stop : emergencyStop;\n"
                f"        }}\n"
                f"        transition initial then Active;\n"
                f"        transition majorityFailAB\n"
                f"            first Active\n"
                f"            if channelAFailed and channelBFailed\n"
                f"            then FailsafeActive;\n"
                f"        transition majorityFailBC\n"
                f"            first Active\n"
                f"            if channelBFailed and channelCFailed\n"
                f"            then FailsafeActive;\n"
                f"        transition majorityFailAC\n"
                f"            first Active\n"
                f"            if channelAFailed and channelCFailed\n"
                f"            then FailsafeActive;\n"
                f"    }}"
            )
        elif redundancy_level == "dual":
            # Canonical SysML v2 form: see the triple-redundancy comment above.
            instructions = (
                f"Add the following INSIDE part def {target_part} (before its closing brace):\n\n"
                f"    attribute redundancyChannels : Integer = 2;\n"
                f"    attribute primaryChannelFailed : Boolean = false;\n"
                f"    attribute backupChannelFailed  : Boolean = false;\n\n"
                f"    action def emergencyStop {{ }}\n\n"
                f"    state def DualChannelRedundancy {{\n"
                f"        state PrimaryActive;\n"
                f"        state BackupActive;\n"
                f"        state FailsafeActive {{\n"
                f"            entry action stop : emergencyStop;\n"
                f"        }}\n"
                f"        transition initial then PrimaryActive;\n"
                f"        transition primaryFail\n"
                f"            first PrimaryActive\n"
                f"            if primaryChannelFailed\n"
                f"            then BackupActive;\n"
                f"        transition backupFail\n"
                f"            first BackupActive\n"
                f"            if backupChannelFailed\n"
                f"            then FailsafeActive;\n"
                f"    }}"
            )
        else:
            return self._parse_cot_response(sysml_text)

        prompt = MCTS_REDUNDANCY_GROUNDING_TEMPLATE.format(
            redundancy_level=redundancy_level,
            target_part=target_part,
            redundancy_instructions=instructions,
            sysml_text=sysml_text,
        )
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=prompt),
        ]
        response = self.llm.complete(messages, temperature=0.2)
        return self._parse_cot_response(response.content)

    #TODO 检查是否需要
    def self_consistency_generate(
        self,
        system_name: str,
        requirements: List[str],
        num_samples: int = 3,
    ) -> CoTResult:
        """
        Generate multiple design candidates and return the most consistent one.

        Implements the self-consistency CoT technique for more reliable results.
        """
        candidates = []
        for _ in range(num_samples):
            result = self.generate_design(
                system_name=system_name,
                requirements=requirements,
                temperature=0.7,
            )
            candidates.append(result)

        # Select the candidate with the most SysML content (heuristic for completeness)
        best = max(
            candidates,
            key=lambda r: len(r.extracted_sysml or ""),
        )
        best.metadata["num_candidates"] = num_samples
        best.metadata["consistency_method"] = "max_sysml_length"
        return best

    def _parse_cot_response(self, response_text: str) -> CoTResult:
        """Parse an LLM response to extract CoT steps, SysML, and JSON."""
        result = CoTResult(final_answer=response_text)

        # Extract SysML code blocks
        sysml_pattern = r"```sysml\n(.*?)```"
        sysml_matches = re.findall(sysml_pattern, response_text, re.DOTALL)
        if sysml_matches:
            result.extracted_sysml = sysml_matches[-1].strip()
        elif re.search(r"```sysml\n", response_text):
            print("  ⚠ [TOKEN LIMIT] LLM response truncated before closing ``` — "
                  "increase max_tokens for this step.")

        # Extract JSON code blocks
        json_pattern = r"```json\n(.*?)```"
        json_matches = re.findall(json_pattern, response_text, re.DOTALL)
        if json_matches:
            try:
                result.extracted_json = json.loads(json_matches[-1].strip())
            except json.JSONDecodeError:
                pass

        # Extract numbered steps
        step_pattern = r"(\d+)\.\s+([A-Z][^:]+):\s*(.*?)(?=\n\d+\.|$)"
        step_matches = re.findall(step_pattern, response_text, re.DOTALL)
        for num, description, content in step_matches:
            result.thought_steps.append(
                ThoughtStep(
                    step_number=int(num),
                    description=description.strip(),
                    content=content.strip(),
                )
            )

        #TODO：Metadata can be added into CoTResult

        return result
