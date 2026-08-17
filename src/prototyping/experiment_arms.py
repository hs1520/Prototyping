"""Unambiguous experiment namespaces for the legacy and revised Option 2 work.

The implemented external-contract pilot historically used ``B0/B1/B2``.  The
revised Blackboard/A-G study deliberately uses ``R*`` labels so archived runs
from the two designs cannot be pooled by name.
"""
from __future__ import annotations

from enum import Enum


LEGACY_EXPERIMENT_NAMESPACE = "LEGACY_EXTERNAL_CONTRACT_V1"
REVISED_EXPERIMENT_NAMESPACE = "BLACKBOARD_AG_V1"
# v10 ran the five generation steps as one shared conversation. Measured on
# three paired seeds it took qualification from 3/3 to 0/3 at 2.1x the prompt
# cost, failing a different check each seed, and was reverted. v11 restores v9's
# independent single-turn generation calls; what it adds over v9 is board
# instrumentation only (an A/G-planning session, archived generation drafts,
# per-turn token accounting), none of which changes what the model is asked to
# produce. The three archived v10 runs are a negative result and must not be
# pooled with either neighbour.
# v12 fixes the attribute materialiser, which could not see a
# unit-suffixed type (`: LengthValue [m]`) and appended a duplicate
# declaration instead of normalising the existing one — the sole cause of
# the one NOT_QUALIFIED run in pilot_v11b_20260731. Generation prompts are
# unchanged; the terminal model is not, so v11 and v12 are not poolable.
# v13 puts A/G Boolean concepts under the typed plan. They are typed by the
# contract but were never planned attributes, so nothing owned their declared
# type and generation was free to write `attribute airborne : Real = 0.0;` —
# which it did in two measured runs, leaving the terminal gate as the first
# thing to see it and able only to fail the run. The gate stays fail-closed;
# what changes is that the existing planned-attribute materialiser now
# enforces the type during generation. v12 and v13 are not poolable.
# v14 corrects v13: it planned every Boolean A/G concept as an attribute,
# including the ones a component guarantees. Those are outputs, generation
# realises them as directed ports, and one measured run then had
# recoveryActuationPowerAvailable as both a port and an attribute. Only
# consumed concepts are planned now. v13 and v14 are not poolable.
# v15 closes the second writer of the same attributes. Two independent
# materialisers write the semantic-anchor attributes, and both decided
# "already present?" with a pattern blind to a unit-suffixed type; v12
# fixed one, and the duplicate came back through the other. The semantic
# binder now also sees a declaration with no initializer. v14 and v15 are
# not poolable.
# v16 also lets the plan own a planned port's TYPE. "Does this port exist?"
# looked only at the name, so a port written with the wrong type was neither
# added nor corrected, and conformance reported it as both a missing planned
# port and an unplanned one — four such pairs failed one measured run. A
# wrong DIRECTION is still reported rather than rewritten: that changes what
# the connections mean. v15 and v16 are not poolable.
# v17 rejects a planned attribute whose value type the pipeline cannot emit.
# One baseline run committed `lockState : StateEnum = Locked` and failed
# Syside on a type that is declared nowhere; the plan has no way to declare
# one, and validation had only checked the name was a well-formed
# identifier. The plan now fails closed, which puts the retry inside the
# bounded attempt budget. This changes generation, so v16 and v17 are not
# poolable.
# v18 rewrites C-style boolean negation deterministically in the syntax gate.
# `if !sensorFailure` is a parser error — SysML v2 spells negation `not` — and
# in pilot_n6_20260802/seed-3/R0-CURRENT it was the committed model's only
# error, which cost that run its qualification. One occurrence in eight
# archived pilots, and the A/G emitter already wrote `not`, so the convention
# was known to the system and simply never enforced on provider output. A model
# the gate previously rejected can now be repaired, so v17 and v18 are not
# poolable. `!=` is untouched: it is a legal inequality the emitter produces.
# v19 stops the plan promising execution evidence the executor cannot produce.
# A STATE_ACTIVE constraint had to claim STATE_EXECUTION, but the executor
# probes a satisfaction boundary by perturbing the right-hand value and needs
# one side to satisfy and the other not to — which `==` fails by construction,
# so every equality state constraint was reported "boundary is not live"
# whatever the model said. In pilot_n6_20260802 seed-3 was the only seed to
# plan two of them and the only run in its arm to lose qualification; it had
# committed to more checking than any other seed, not done worse work. Such
# constraints are now required to claim INSPECTION: still emitted, still
# inspectable, no longer counted as discharged execution evidence. This moves
# both the qualification outcome and the committed-obligation count, so v18 and
# v19 are not poolable.
# v20 stops the source-anchor gate demanding a lexical match the two vocabularies
# cannot produce. It intersects the stemmed words of a phrase copied verbatim out
# of a requirement with the stemmed words of a model identifier, and those diverge
# by construction: identifiers abbreviate (`navState` for `'navigate'`), they use
# the domain synonym (`obstacleData` for `'collision threat'`), and a requirement
# often offers only an adjunct clause where an effect is wanted (`'with a circular
# error probable (CEP) of less than 1.0 metre'`), which names no behaviour for any
# component to represent. Measured on the failed authoritative attempt of
# 2026-08-01: 13 issues over 11 realizations, of which 10 were the gate's own
# lexical reach and 3 were real endpoint mistakes that still fail. Abbreviation
# and synonym now count, an adjunct-headed phrase is not required to be
# represented, and the planning prompt states the rule instead of leaving it to
# be discovered by rejection. Plans the gate previously rejected can now pass, so
# v19 and v20 are not poolable.
# v21 makes the response a functional requirement obliges a planned decision
# instead of a keyword inference. Three probes on the extraction path (2026-08-16,
# frozen_requirements=None) showed the keyword table reading "delivery
# waypoint" as a release obligation and "receive a waypoint sequence" as a
# navigate one, and the plan then failing closed for a behaviour the requirement
# never asked for. Each FUNC realization now records `response_intent` from a
# closed set (release|return|land|navigate|report|self_test|none) with a
# rationale required for `none`; the plan validator and the terminal closure
# gate both read that field and fall back to the keyword table only when it is
# blank, so archived runs keep their verdicts. The planning prompt gained the
# field and its rule, which moves the user-prompt digest of every planning call
# on every arm (golden_refactor_call_sequence.json ordinals 1/2/4/5;
# system-prompt digests unchanged). Two smaller repairs ride with it: the unit
# vocabulary accepts the degree symbol and Celsius, numeric bounds accept the
# typographic minus U+2212 (each a spelling an extracted set actually used and
# the frozen set never had), and "deliver" alone no longer marks a release
# intent. The plan validator now looks for a requirement's response in the
# behaviour its realization names, falling back to provenance only when none is
# named: requiring the provenance tag to equal the requirement made one state
# machine unable to answer two requirements (navigate and return as two states
# of one behaviour), which is the natural shape and what the LLM kept planning.
# The return-intent markers gained the launch/RTL spellings for the same reason
# the units did. Two closure-side changes complete the intent mechanism: the
# extractor's "no measurable criterion" flag now travels on the requirement
# input artefact (`unmeasurable_req_ids`), and the terminal closure audit no
# longer treats a FUNC row as a model gap when the plan recorded
# response_intent=none for it AND the extractor flagged it unmeasurable -- two
# independent signals that the requirement offers nothing to anchor to. The
# verification matrix records such rows under a `planned_no_response` tier and
# leaves their status UNASSIGNED; the gap is real but is the requirement's, not
# the model's. The surgical repair prompt also gained an inhibition-shaped
# guidance branch beside the existing power-on one. On the frozen path none of
# this fires: no frozen requirement is unmeasurable.
# One further constraint-plan repair: a RUNTIME_MEASUREMENT attribute used in a
# constraint must carry an input binding at EVERY verification tier. The
# STATE_EXECUTION exemption dated from before v19 and was wrong -- the state
# executor reads a bound input like any other tier -- and its effect on the
# extraction path was a plan that froze, a model whose scenarios failed with
# "runtime subject X is not bound to an input", and a closure repair that the
# plan-conformance gate then had to refuse (a new binding is an unplanned
# element). Every archived plan (18 pilot cells + the authoritative run) already
# satisfies the tightened rule, so no archived verdict moves; the golden
# sequence moves on R2 ordinal 5 only, whose fixture retries a plan that trips
# this check and so re-embeds its reworded issue text.
# And the disagreement `_state_execution_obstacle` recorded and deliberately
# left open -- the executor demands a bound subject for a STATE_ACTIVE
# constraint, the validator did not -- is now resolved in the executor's favour:
# an unbound subject is an obstacle, and the constraint must claim INSPECTION.
# The probes settled which side was wrong. NOTE the one place this touches an
# archived verdict: the authoritative run's plan carries exactly one such
# constraint (FlightController.attitudeDeviationRms, claiming STATE_EXECUTION),
# which that run recorded as an ADVISORY; under v21 the same plan would be
# returned to the LLM for correction. The archived matrix row (REQ_FUNC_003,
# partial, anchored by the datasheet tier) is unaffected because the datasheet
# evidence, not the state execution, is what anchored it. Two tests whose
# fixtures had pinned the unresolved behaviour now claim INSPECTION.
# Last, the verification matrix's initial-state rule (a "power-on"/"default"
# requirement is anchored by an owner machine whose initial state the text
# names) now yields to a recorded response intent. An extracted "execute a
# power-on self-check" carried intent self_test, but the substring "poweron"
# matched the flight-phase manager's initial state PowerOn, and that machine's
# initialisation failure was reported as the self-check requirement's evidence
# while the self-check machine itself passed. A requirement that obliges a
# response is anchored by the state that produces it. The frozen set's only
# initial-state requirement (REQ_SAFE_008, "default to the mechanically locked
# state") is a safety requirement with no response intent and is unaffected;
# its archived verdict stands.
# And the post-flight-report trigger rule in functional_behavior._has_required_
# trigger, which demanded both "land" and "complet" in the trigger context
# because the frozen requirement said "upon completion of the automated landing
# sequence", now also accepts a completion trigger that is not a start-of-flight
# event, because an extracted requirement said "upon mission completion" and the
# model wrote MissionCompletion. The existing test that fires the same state via
# GenericCommand still fails, as it must.
# Finally, a component may be declared `passive` in the plan (with a
# passive_rationale): a purely structural body -- an airframe -- that carries
# other parts but exchanges no signals, commands or power. A passive component
# may plan no ports (the "has no planned ports" rule yields to it), is
# materialised as a `// PLAN-PASSIVE <Def>: <reason>` marker inside its part
# def, and takes no part in any reachability scenario, including the
# power-to-structure one. Before this the plan validator forced a port onto
# every component, the LLM obliged with a port nobody connected, and the
# reachability simulator then failed the power path into it -- on the
# authoritative run and on every extraction probe alike, so the airframe cost
# every model two failed scenarios by construction. Whether a structural body
# is passive is now a recorded design decision rather than a default the
# simulator assumes and the planner cannot express. Archived models carry no
# marker and keep their scenarios and scores; only a plan that declares
# passivity changes anything.
# One consequence of accepting the degree sign in the unit vocabulary had to be
# closed on the model side: an extracted plan then wrote `[°]` and `[°C]` into
# the model, which are not SysML tokens (a parse error, and one that reparents
# every later declaration, exactly as `[%]` did before it was mapped). The
# generation stage committed such a model with the syntax check failing as a
# qualification verdict rather than a gate, and the exploration stage's
# variation injector -- which requires a clean parse -- then rejected the
# catalogue seed and reported it as "could not form two variants" (two archival
# attempts on the extraction path stopped exactly there). Every place a plan
# unit is serialised into model text now goes through sysml_unit_name(), and
# the table maps ° -> deg and °C -> degC (both resolve under `import SI::*`).
# The frozen set's units (s, m, kg, percent) were already legal tokens, which
# is why the missing mapping was never observed.
# The self-test trigger rule in _has_required_trigger required the word
# "selftest" in the TRIGGER context as well as a start-of-life word, which held
# only when the transition itself was named startSelfTest; a transition named
# powerOn firing accept PowerOnEvent into a state whose entry action is
# selfTest -- the same design -- was refused. The response being a self-test is
# already established by the marker match; the rule now checks only that the
# trigger is a start-of-life event. This changes what the model is asked, so v20 and v21 are not
# poolable; every archived pilot in the artefact tree ran at v20 or earlier.
COMMON_GENERATION_PIPELINE_VERSION = "typed-whole-model-plan-v21"
R2_DETERMINISTIC_GENERATION_MODE = "DETERMINISTIC_SPEC_EMITTER"
R2_DETERMINISTIC_INTERVENTION_VERSION = (
    "r2-bbag-whole-model-guided-deterministic-v3"
)
# LLM-authored A/G is a SEPARATE intervention (design §15): the LLM authors the
# bounded A/G decomposition instead of the deterministic emitter. Version 7
# freezes that decomposition before ordinary architecture/behavior generation;
# its response vocabulary is therefore a design input implemented downstream,
# rather than being inferred post-hoc from already-generated arbiter behavior.
# It retains the fixed authoring and post-commit repair budgets.
# The post-commit repair budget remains fixed and independently audited.
# It must never be pooled with authored-v1/v2 or the deterministic mode.
# v9 changes the retry seam, not the budgets: the structured-decision turns are a
# real multi-turn conversation, so a rejected decision object comes back as the
# model's own assistant turn instead of being paraphrased into a fresh prompt.
# That changes what the model sees on every retry, so v8 and v9 are not pooled.
R2_LLM_AUTHORED_GENERATION_MODE = "LLM_AUTHORED_AG"
R2_LLM_AUTHORED_INTERVENTION_VERSION = (
    "r2-bbag-whole-model-guided-authored-v10"
)

# LLM-decided specs are a THIRD frozen intervention. The LLM emits the engineering
# decisions — pattern, timing origin and apportionment, discharge wiring, response
# ordering — and the deterministic emitter renders the SysML from them. Measurement
# motivated it: the LLM-authored mode agreed with gold on allocation 6/6 and
# discharge 4/6 while never once reaching a PASS, because it kept losing rounds to
# the notation rather than to the engineering. Here conformance holds by
# construction and only the decisions are judged. Separate frozen config; never
# pooled with either other mode.
# v6 makes the bounded validation retry a real multi-turn conversation: the
# rejected decision object is resent as the model's own assistant turn, so
# "keep everything that was already valid" refers to something it can read. The
# 9/9 evidence was produced under v5, where every retry was a fresh single-turn
# prompt carrying only a paraphrase of the rejection. v5 and v6 must never be
# pooled, and a v6 claim requires its own runs.
# v8 carries no change to this intervention's own prompts or budgets. It records
# that the shared generation path underneath it moved: between 03ce19b and
# 30dd9f9, fifteen commits changed what the model is asked and what it is shown
# — planned events are retried when invalid and preserved through assembly
# (0ff752a, e1a7adb), high-thinking assembly output is bounded (fa62f85),
# truncated Step 1 plans are separated from missing ones (b3c5f32), units are
# emitted as SysML names and recognised however spelled (abc9462, fa4cca8,
# 7ed60a1), and Vertex calls acquired a hard wall-clock timeout (e5464a2,
# 02f62f8). A targeted recheck on seeds 2/3/4 at 30dd9f9 confirmed the effect is
# not cosmetic: R0 and R1 both scored lower than on 2 August and became
# indistinguishable from each other, while R2 scored higher and held pattern
# conformance at 9/9 chains. So v7 and v8 must never be pooled: a v8 claim
# requires its own runs, and `code_revision` in each pilot_config is the field
# that actually separates them.
R2_LLM_DECIDED_GENERATION_MODE = "LLM_DECIDED_SPEC"
R2_LLM_DECIDED_INTERVENTION_VERSION = (
    "r2-bbag-whole-model-guided-decided-v8"
)

R2_GENERATION_MODES = (
    R2_DETERMINISTIC_GENERATION_MODE,
    R2_LLM_AUTHORED_GENERATION_MODE,
    R2_LLM_DECIDED_GENERATION_MODE,
)
# The one accepted (mode, version) pair per intervention; nothing else may run.
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
        """The arm's intervention is wired and runnable end-to-end.

        R2-BBAG became runnable once the SysML A/G profile, extractor, and bounded
        compositional checker landed (Increment 2). Runnability is distinct from
        :attr:`evaluation_ready`.
        """
        return self in {
            self.CURRENT, self.BLACKBOARD_CONTEXT, self.SEMANTIC_ASSURANCE
        }

    @property
    def evaluation_ready(self) -> bool:
        """The arm can be scored and pooled into the controlled comparison.

        R0/R1 rest on deterministic coordination/context metrics. The R2-BBAG code
        runtime loop exists — A/G-aware generation (``ag_emitter``/``ag_chains``), the
        extractor/checker, orchestrator wiring, and the independent evaluator
        (``ag_evaluation.py``) all exist. Post-hoc readiness additionally requires
        the complete frozen configuration/boundary/gold/taxonomy/blind-label evidence
        chain; until that manifest clears, pooling R2 accuracy/F1 is not valid, so R2 stays
        runnable-but-not-poolable and is deliberately excluded here.
        """
        return self in {self.CURRENT, self.BLACKBOARD_CONTEXT}

    @property
    def uses_blackboard(self) -> bool:
        return self in {self.BLACKBOARD_CONTEXT, self.SEMANTIC_ASSURANCE}


def revised_arm_metadata(
    arm: RevisedExperimentArm, r2_generation_mode: str | None = None
) -> dict[str, object]:
    """Arm metadata for a run result.

    ``r2_generation_mode`` must be the mode the run actually executed. It used to
    be hardcoded to the deterministic intervention, which meant a run executing a
    different mode would report itself as deterministic — mislabelled evidence that
    the pooling gates would then accept, because they compare the recorded mode
    rather than observe the behaviour.
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
            # derived, never passed in: the recorded version must be the one bound
            # to the mode actually executed, or results from different
            # interventions could be pooled under one version
            "r2_intervention_version": R2_INTERVENTION_VERSION_BY_MODE[mode],
        })
    return metadata
