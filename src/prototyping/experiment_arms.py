"""Unambiguous experiment namespaces for the legacy and revised Option 2 work.

The external-contract pilot used ``B0/B1/B2``; the revised Blackboard/A-G study
uses ``R*`` labels so archived runs from the two designs cannot be pooled by
name.
"""
from __future__ import annotations

from enum import Enum


LEGACY_EXPERIMENT_NAMESPACE = "LEGACY_EXTERNAL_CONTRACT_V1"
REVISED_EXPERIMENT_NAMESPACE = "BLACKBOARD_AG_V1"
# v10 ran the five generation steps as one shared conversation: qualification
# 3/3 -> 0/3 on three paired seeds at 2.1x prompt cost, a different check failing
# each seed. Reverted. v11 restores v9's independent single-turn calls and adds
# board instrumentation only (A/G-planning session, archived drafts, per-turn
# token accounting). The three v10 runs are a negative result, pooled with
# neither neighbour.
# v12 fixes the attribute materialiser, which could not see a unit-suffixed type
# (`: LengthValue [m]`) and appended a duplicate instead of normalising the
# existing declaration - the one NOT_QUALIFIED run in pilot_v11b_20260731.
# Prompts unchanged, terminal model not, so v11 and v12 are not poolable.
# v13 puts A/G Boolean concepts under the typed plan: nothing owned their
# declared type, so generation wrote `attribute airborne : Real = 0.0;` in two
# runs and only the terminal gate saw it. The planned-attribute materialiser now
# enforces the type during generation. v12 and v13 are not poolable.
# v14 corrects v13, which planned guaranteed concepts as attributes too. A
# guarantee is an output realised as a directed port, so one run had
# recoveryActuationPowerAvailable as both port and attribute. Only consumed
# concepts are planned now. v13 and v14 are not poolable.
# v15 closes the second writer of those attributes: both materialisers tested
# "already present?" with a pattern blind to a unit-suffixed type, so the
# duplicate returned through the one v12 did not fix. The semantic binder now
# also sees a declaration with no initializer. v14 and v15 are not poolable.
# v16 lets the plan own a planned port's type. The existence test read only the
# name, so a wrongly typed port was neither added nor corrected and conformance
# called it both missing and unplanned - four such pairs failed one run. A wrong
# direction is still reported rather than rewritten,
# because it changes what the connections mean. v15 and v16 are not poolable.
# v17 rejects a planned attribute whose value type the pipeline cannot emit: one
# baseline run committed `lockState : StateEnum = Locked` and failed Syside on a
# type declared nowhere, while validation had only checked the name was an
# identifier. Failing closed at plan time keeps the retry inside the bounded
# attempt budget. v16 and v17 are not poolable.
# v18 rewrites C-style boolean negation in the syntax gate. SysML v2 spells
# negation `not`, and `if !sensorFailure` was the committed model's only error in
# pilot_n6_20260802/seed-3/R0-CURRENT, costing that run its qualification. A
# model the gate previously rejected can now be repaired, so v17 and v18 are not
# poolable; `!=` is untouched, being a legal inequality the emitter produces.
# v19 stops the plan promising execution evidence the executor cannot produce. A
# STATE_ACTIVE constraint had to claim STATE_EXECUTION, but the executor probes
# the satisfaction boundary by perturbing the right-hand value and needs one side
# to satisfy and one not, which `==` cannot do, so every equality state
# constraint reported "boundary is not live". They now claim INSPECTION: still
# emitted, no longer counted as discharged execution evidence. This moves the
# qualification outcome and the obligation count, so v18 and v19 are not
# poolable.
# v20 relaxes the source-anchor gate, which intersected the stemmed words of a
# phrase copied from a requirement with those of a model identifier. The two
# vocabularies diverge: identifiers abbreviate (`navState` for `'navigate'`) or
# use the domain synonym (`obstacleData` for `'collision threat'`), and an
# adjunct clause (`'with a circular error probable (CEP) of less than 1.0
# metre'`) names no behaviour to represent. On the failed
# authoritative attempt of 2026-08-01, 10 of 13 issues over 11 realizations were
# the gate's own lexical reach; the 3 endpoint mistakes still fail. Abbreviation
# and synonym now count, an adjunct-headed phrase need not be represented, and
# the planning prompt states the rule. v19 and v20 are not poolable.
# v21 makes the response a functional requirement obliges a planned decision
# rather than a keyword inference: extraction probes (2026-08-16) had the keyword
# table read "delivery waypoint" as a release obligation, so the plan failed
# closed for a behaviour the requirement never asked for. Each FUNC realization
# records `response_intent` from a closed set
# (release|return|land|navigate|report|self_test|none), rationale required for
# `none`; the plan validator and terminal closure gate read it and fall back to
# the keyword table only when blank, so archived runs keep their verdicts. The
# planning prompt gained the field and its rule, moving the user-prompt digest of
# every planning call on every arm (golden_refactor_call_sequence.json ordinals
# 1/2/4/5; system-prompt digests unchanged). Riding along: the unit vocabulary
# accepts the degree symbol and Celsius, numeric bounds accept U+2212, "deliver"
# alone no longer marks a release intent, and return-intent markers gained the
# launch/RTL spellings. The validator looks for a requirement's response in the
# behaviour its realization names, falling back to provenance only when none is
# named - requiring provenance to equal the requirement left one state machine
# unable to answer two requirements (navigate and return as two states of one
# behaviour). On the closure side, the extractor's "no measurable criterion" flag
# travels on the requirement input artefact (`unmeasurable_req_ids`), and the
# closure audit no longer treats a FUNC row as a model gap when the plan recorded
# response_intent=none and the extractor flagged it unmeasurable; the matrix
# records those rows under a `planned_no_response` tier with status UNASSIGNED.
# The surgical repair prompt gained an inhibition-shaped branch beside the
# power-on one. None of this fires on the frozen path: no frozen requirement is
# unmeasurable.
# A RUNTIME_MEASUREMENT attribute used in a constraint carries an input binding
# at every verification tier. The STATE_EXECUTION exemption predated v19 and was
# wrong - the state executor reads a bound input like any tier - and on the
# extraction path it left scenarios failing "runtime subject X is not bound to an
# input" plus a closure repair the plan-conformance gate had to refuse as an
# unplanned element. Every archived plan already satisfies the tightened rule, so
# no archived verdict moves; the golden sequence moves on R2 ordinal 5 only.
# `_state_execution_obstacle` is resolved in the executor's favour: an unbound
# subject for a STATE_ACTIVE constraint is an obstacle and the constraint claims
# INSPECTION. This touches one archived verdict - the authoritative run's plan
# carries one such constraint (FlightController.attitudeDeviationRms, claiming
# STATE_EXECUTION) recorded as an ADVISORY, which under v21 would be returned for
# correction. Its matrix row (REQ_FUNC_003, partial) is anchored by the datasheet
# tier and unaffected.
# The verification matrix's initial-state rule (anchoring a "power-on"/"default"
# requirement by the machine whose initial state the text names) now yields to a
# recorded response intent: an extracted "execute a power-on self-check" carried
# intent self_test, but "poweron" matched the flight-phase manager's initial
# state PowerOn and that machine's initialisation failure was reported as the
# self-check requirement's evidence. A requirement obliging a response is
# anchored by the state that produces it. The frozen set's only initial-state
# requirement (REQ_SAFE_008) has no response intent and keeps its verdict.
# The post-flight-report rule in functional_behavior._has_required_trigger
# demanded both "land" and "complet" in the trigger context, from the frozen
# wording "upon completion of the automated landing sequence"; it now also
# accepts a completion trigger that is not a start-of-flight event, because an
# extracted requirement said "upon mission completion" and the model wrote
# MissionCompletion.
# A component may be declared `passive` in the plan (with a passive_rationale): a
# structural body - an airframe - that carries other parts but exchanges no
# signals, commands or power. It may plan no ports, is materialised as a
# `// PLAN-PASSIVE <Def>: <reason>` marker inside its part def, and takes no part
# in any reachability scenario, including the power-to-structure one. Before this
# the validator forced a port onto every component and the reachability simulator
# failed the power path into it, costing every model two scenarios. Archived
# models carry no marker.
# Accepting the degree sign had a model-side consequence: `[°]` and `[°C]` are
# not SysML tokens and reparent every later declaration, as `[%]` did before it
# was mapped. Generation committed such a model with the syntax check recorded as
# a qualification verdict rather than a gate, and the variation injector, which
# needs a clean parse, then rejected the catalogue seed as "could not form two
# variants" (two archival attempts stopped there). Plan units are now serialised
# through sysml_unit_name(), which maps ° -> deg and °C -> degC (both resolve
# under `import SI::*`). The frozen set's units (s, m, kg, percent) were already
# legal tokens.
# The self-test rule in _has_required_trigger required "selftest" in the trigger
# context as well as a start-of-life word, which held only for a transition named
# startSelfTest; a transition named powerOn firing accept PowerOnEvent into a
# state whose entry action is selfTest was refused. The marker match already
# establishes the response, so the rule now checks only the start-of-life
# trigger. This changes what the model is asked, so v20 and v21 are not poolable;
# every archived pilot ran at v20 or earlier.
# v22 opens the v21 response-intent vocabulary at both ends. The closed set tied
# what the gate can check to what the plan can express, so a requirement obliging
# a response the table does not name (an alert, an unlock) had to be recorded as
# `none`, contradicting its own text. Two additions, no removals: (1) a FUNC
# realization may record an out-of-vocabulary intent by declaring
# `response_markers`, the lowercase action-name fragments a reachable state's
# action must show, each sharing a content word with the copied effect_concept
# (the anti-self-grading rule); the plan validator and closure gate then run the
# ordinary reachable-action check against them, while built-in intents ignore
# declared markers. (2) `response_intent` may record `unverifiable` with a
# rationale when a response is obliged but no reachable-action name can evidence
# it; the gate holds the model to nothing and the matrix reports the row under a
# `planned_unverifiable_response` tier instead of `none`. The prompt gained the
# field and both rules, moving the same planning-call digests v21 moved. Nothing
# fires on the frozen path - every frozen FUNC requirement's response is named by
# the built-in table - so v22 plans of frozen inputs differ from v21 only in
# prompt text, and are still not poolable.
COMMON_GENERATION_PIPELINE_VERSION = "typed-whole-model-plan-v22"
R2_DETERMINISTIC_GENERATION_MODE = "DETERMINISTIC_SPEC_EMITTER"
R2_DETERMINISTIC_INTERVENTION_VERSION = (
    "r2-bbag-whole-model-guided-deterministic-v3"
)
# LLM-authored A/G is a separate intervention (design §15): the LLM authors the
# bounded A/G decomposition instead of the deterministic emitter. Version 7
# freezes that decomposition before ordinary architecture/behavior generation, so
# its response vocabulary is a design input implemented downstream rather than
# inferred from already-generated arbiter behavior. Authoring and post-commit
# repair budgets are fixed and independently audited. Not pooled with
# authored-v1/v2 or the deterministic mode.
# v9 changes the retry seam, not the budgets: the structured-decision turns are
# a multi-turn conversation, so a rejected decision object comes back as the
# model's own assistant turn instead of a paraphrase in a fresh prompt. That
# changes what the model sees on every retry, so v8 and v9 are not pooled.
R2_LLM_AUTHORED_GENERATION_MODE = "LLM_AUTHORED_AG"
R2_LLM_AUTHORED_INTERVENTION_VERSION = (
    "r2-bbag-whole-model-guided-authored-v10"
)

# LLM-decided specs are a third frozen intervention: the LLM emits the
# engineering decisions - pattern, timing origin and apportionment, discharge
# wiring, response ordering - and the deterministic emitter renders the SysML.
# Motivation: the LLM-authored mode matched gold on allocation 6/6 and discharge
# 4/6 without ever reaching a PASS, losing rounds to the notation rather than the
# engineering. Here conformance holds by construction and only the decisions are
# judged. Separate frozen config, not pooled with either other mode.
# v6 makes the bounded validation retry a multi-turn conversation: the rejected
# decision object is resent as the model's own assistant turn, so "keep
# everything that was already valid" refers to something it can read. The 9/9
# evidence was produced under v5, where every retry was a fresh single-turn
# prompt carrying a paraphrase of the rejection. v5 and v6 are not pooled; a v6
# claim needs its own runs.
# v8 changes no prompt or budget here. It records that the shared generation path
# moved: between 03ce19b and 30dd9f9, fifteen commits changed what the model is
# asked and shown - planned events retried when invalid and preserved through
# assembly (0ff752a, e1a7adb), bounded high-thinking assembly output (fa62f85),
# truncated Step 1 plans separated from missing ones (b3c5f32), units emitted as
# SysML names and recognised however spelled (abc9462, fa4cca8, 7ed60a1), and a
# hard wall-clock timeout on Vertex calls (e5464a2, 02f62f8). A recheck on seeds
# 2/3/4 at 30dd9f9: R0 and R1 both scored lower than on 2 August and became
# indistinguishable from each other, while R2 scored higher and held pattern
# conformance at 9/9 chains. So v7 and v8 are not pooled; `code_revision` in each
# pilot_config is the field that separates them.
R2_LLM_DECIDED_GENERATION_MODE = "LLM_DECIDED_SPEC"
R2_LLM_DECIDED_INTERVENTION_VERSION = (
    "r2-bbag-whole-model-guided-decided-v8"
)

R2_GENERATION_MODES = (
    R2_DETERMINISTIC_GENERATION_MODE,
    R2_LLM_AUTHORED_GENERATION_MODE,
    R2_LLM_DECIDED_GENERATION_MODE,
)
R2_INTERVENTION_VERSION_BY_MODE = {
    R2_DETERMINISTIC_GENERATION_MODE: R2_DETERMINISTIC_INTERVENTION_VERSION,
    R2_LLM_AUTHORED_GENERATION_MODE: R2_LLM_AUTHORED_INTERVENTION_VERSION,
    R2_LLM_DECIDED_GENERATION_MODE: R2_LLM_DECIDED_INTERVENTION_VERSION,
}


class RevisedExperimentArm(str, Enum):
    """Arms reserved for the supervisor-directed redesign."""

    CURRENT = "R0-CURRENT"
    BLACKBOARD_CONTEXT = "R1-BBCTX"
    SEMANTIC_ASSURANCE = "R2-BBAG"
    LONG_SESSION_DIAGNOSTIC = "R1-LONG"

    @classmethod
    def parse(cls, value: "RevisedExperimentArm | str") -> "RevisedExperimentArm":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().upper().replace("_", "-")
        aliases = {
            "R0": cls.CURRENT,
            "CURRENT": cls.CURRENT,
            "R1": cls.BLACKBOARD_CONTEXT,
            "BB-CTX": cls.BLACKBOARD_CONTEXT,
            "R2": cls.SEMANTIC_ASSURANCE,
            "BB-AG": cls.SEMANTIC_ASSURANCE,
            "R1-LONG": cls.LONG_SESSION_DIAGNOSTIC,
        }
        if normalized in aliases:
            return aliases[normalized]
        for arm in cls:
            if normalized == arm.value:
                return arm
        raise ValueError(f"unknown revised experiment arm: {value!r}")

    @property
    def implemented(self) -> bool:
        """The arm's intervention is wired and runnable end-to-end."""
        return self in {
            self.CURRENT, self.BLACKBOARD_CONTEXT, self.SEMANTIC_ASSURANCE
        }

    @property
    def evaluation_ready(self) -> bool:
        """The arm can be scored and pooled into the controlled comparison.

        R0/R1 rest on deterministic coordination/context metrics. R2-BBAG's runtime
        loop exists (``ag_emitter``/``ag_chains``, extractor/checker, orchestrator
        wiring, ``ag_evaluation.py``), but post-hoc readiness also needs the frozen
        configuration/boundary/gold/taxonomy/blind-label chain; until that manifest
        clears, R2 accuracy/F1 cannot be pooled, so R2 is excluded here.
        """
        return self in {self.CURRENT, self.BLACKBOARD_CONTEXT}

    @property
    def uses_blackboard(self) -> bool:
        return self in {self.BLACKBOARD_CONTEXT, self.SEMANTIC_ASSURANCE}


def revised_arm_metadata(
    arm: RevisedExperimentArm, r2_generation_mode: str | None = None
) -> dict[str, object]:
    """Arm metadata for a run result.

    ``r2_generation_mode`` is the mode the run actually executed. It was once
    hardcoded to the deterministic intervention, so a run in another mode reported
    itself as deterministic and the pooling gates, which compare the recorded mode
    rather than the behaviour, accepted it.
    """
    metadata: dict[str, object] = {
        "experiment_namespace": REVISED_EXPERIMENT_NAMESPACE,
        "configuration": arm.value,
        "implemented": arm.implemented,
        "evaluation_ready": arm.evaluation_ready,
        "common_generation_pipeline_version": COMMON_GENERATION_PIPELINE_VERSION,
        "blackboard_context_intervention": arm.uses_blackboard,
        "semantic_assurance_intervention": arm is RevisedExperimentArm.SEMANTIC_ASSURANCE,
        "global_long_session_diagnostic": (
            arm is RevisedExperimentArm.LONG_SESSION_DIAGNOSTIC
        ),
    }
    if arm is RevisedExperimentArm.SEMANTIC_ASSURANCE:
        mode = r2_generation_mode or R2_DETERMINISTIC_GENERATION_MODE
        if mode not in R2_INTERVENTION_VERSION_BY_MODE:
            raise ValueError(
                f"unknown r2_generation_mode {mode!r}; a run may not report an "
                "intervention that has no frozen version"
            )
        metadata.update({
            "r2_generation_mode": mode,
            # derived, not passed in: the recorded version has to match the mode
            # actually executed, or two interventions pool under one version
            "r2_intervention_version": R2_INTERVENTION_VERSION_BY_MODE[mode],
        })
    return metadata
