"""
requirement_linker.py

Maps SysML requirements to ArduPilot parameters and SITL test specs.

核心设计原则（重构后）：
  以 guard / attribute 内容作为语义锚点，不依赖 REQ ID 字符串。

  REQ ID 是 LLM 自由命名的标签，跨系统、跨运行不稳定。
  guard 内容（operator + variable keywords + threshold）才是不变的语义。

映射流程：
  part.satisfy(REQ_X)
    → 取该 part 的状态机 guard（e.g. batterySoc <= 25.0）
    → 按 guard 内容（operator=<= + var 含 battery）匹配 _CONTENT_CATALOGUE
    → 得到 ArduPilot 参数（BATT_FS_LOW_PCT = 25.0）
    → REQ ID 仅用于测试脚本的标签，不参与匹配逻辑

五层 Fallback（内容优先）：
  层1  guard/attr 内容匹配（_CONTENT_CATALOGUE）← 最健壮
  层2  AST 合成器（guard 变量名关键词）
  层3  LLM 状态机语义分类
  层3b LLM 需求文本直接推断
  层4  全部失败 → unmapped
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Dict, List, Optional, Any, Mapping, Tuple

from src.utils.digest import sha256_text
from src.sysml.lite_model import SysMLLiteModel
from src.sitl.sitl_specs import InjectSpec, VerifySpec

try:
    import syside as _syside
    _SYSIDE_OK = True
except ImportError:
    _syside = None      # type: ignore
    _SYSIDE_OK = False


from src.sitl.sitl_catalogue import (
    _CONTENT_CATALOGUE,
    _TAG_TO_ENTRY,
    ContentEntry,
    GuardMatcher,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedParam:
    req_id: str
    part_name: str
    param_name: str
    value: Any
    source: str = ""       # 描述值的来源，例如 "guard:<=:25.0" 或 "attr:maxAltitude"


@dataclass(frozen=True)
class SITLTestSpec:
    req_id: str
    tier: str
    inject: InjectSpec
    verify: VerifySpec
    notes: str
    params: tuple[ResolvedParam, ...] = ()


@dataclass(frozen=True, slots=True)
class GuardEvidence:
    part_name: str
    attribute: str
    kind: str
    operator: str
    threshold: Any
    signature: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class AcceptEventGuard:
    """Pseudo-guard for an accept-event transition (no boolean guard).

    Event-driven and guard-driven state machines are two legal SysML spellings
    of the same behavioral semantics; representing accepts as guard-shaped
    records lets the exclusive-claim machinery treat both uniformly.
    ``attribute`` carries the accepted event type name (e.g.
    ``AbortConditionActive``) so keyword matching and claim keys work unchanged.
    """
    attribute: str
    kind: str = "accept_event"
    operator: str = "accept"
    threshold: Any = None


@dataclass(frozen=True, slots=True)
class RequirementEvidenceBundle:
    """Immutable, revision-bound input for SITL and verification consumers."""

    model_digest: str
    requirement_texts: Mapping[str, str]
    satisfying_parts: Mapping[str, tuple[str, ...]]
    guard_assignments: Mapping[str, GuardEvidence]
    test_specs: tuple[SITLTestSpec, ...]
    resolved_params: tuple[ResolvedParam, ...]
    parm_file: str
    traceability_mismatches: tuple[Mapping[str, str], ...]
    coverage: Mapping[str, Any]

    def coverage_payload(self) -> Dict[str, Any]:
        """Return the legacy JSON shape without exposing mutable bundle state."""
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in self.coverage.items()
        }


# ---------------------------------------------------------------------------
# RequirementLinker
# ---------------------------------------------------------------------------

class RequirementLinker:
    """
    通过 AST（satisfy_relationships + state_extractor guard）将 SysML
    需求映射到 ArduPilot 参数，不依赖属性名字符串。

    可选传入 llm 启用语义标签分类，使得不在 _REQ_CATALOGUE key 里的 req ID
    也能通过状态机语义找到匹配的模板（三层 fallback：req ID → 语义标签 →
    unmapped）。
    """

    def __init__(
        self,
        model: SysMLLiteModel,
        llm: Optional[Any] = None,
        verbose: bool = False,
        plan_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._model = model
        self._llm = llm
        self._verbose = verbose
        # Requirement → model-identity bindings from the frozen plan. The
        # sent-command traceability check consumes the ROUTE: a response that
        # sends through the requirement's own causal-path command leg matches
        # structurally, whatever the payload item is spelled like (run3 sends
        # RecoveryCmdData through recoveryCmd — the plan's SAFE_005 route —
        # and the CHUTE-substring check rejected it).
        from src.simulation.verification_binding import plan_bindings
        if plan_payload is None:
            metadata = getattr(model, "metadata", None) or {}
            candidate = metadata.get("whole_model_generation_plan")
            plan_payload = candidate if isinstance(candidate, dict) else None
        self._requirement_bindings = plan_bindings(plan_payload)
        self._contract_bundle = None
        self._contracts: Dict[str, Any] = {}
        self._contract_trace_findings: Dict[str, List[Any]] = {}
        # req_id → part_name
        self._satisfy_map: Dict[str, List[str]] = self._build_satisfy_map()
        # part_name → List[GuardCondition]
        self._guard_map: Dict[str, List[Any]] = self._build_guard_map()
        # part_name → {attr_name: float}
        self._attr_map: Dict[str, Dict[str, float]] = self._build_attr_map()
        # part_name → [port_name, ...]（attr 缺席时的接口/配置类匹配面）
        self._port_map: Dict[str, List[str]] = self._build_port_map()
        # semantic_tag → ContentEntry（供 Layer 2/3 反查）
        self._tag_to_entry: Dict[str, ContentEntry] = _TAG_TO_ENTRY
        # part_name → List[SynthesizedSpec]（AST 合成，主层）
        self._ast_specs = self._build_ast_specs()
        # req_id → tag（LLM 状态机语义分类，第三层 fallback）
        self._semantic_map: Dict[str, str] = self._build_semantic_map()
        # req_id → {param, value, tier}（LLM 需求文本直接推断，第四层 fallback）
        self._direct_param_map: Dict[str, Dict] = self._build_direct_param_map()
        # req_id → requirement doc text; used only as a traceability cross-check
        # against the guard/action-derived semantic tag. The guard remains the
        # primary matching anchor, but a text/tag family mismatch means the model
        # is satisfying a requirement with the wrong behavior.
        self._req_texts: Dict[str, str] = self._extract_requirement_texts()
        self._traceability_mismatches: Dict[str, Dict[str, str]] = {}
        # req_id → (part_name, guard, ContentEntry)：独占 guard 分配
        # 同一 guard 只能分配给一个 req，防止多 REQ 共享 part 时全部映射到同一 guard
        self._guard_assignment: Dict[str, Any] = self._assign_guards_exclusive()
        # _lookup_catalogue 结果缓存：避免 _resolve_all 多次调用时重复打印 [CONTENT]
        self._catalogue_cache: Dict[str, Optional[Dict]] = {}
        self._resolved_cache: Optional[tuple[ResolvedParam, ...]] = None
        self._evidence_bundle: Optional[RequirementEvidenceBundle] = None

    def _assign_guards_exclusive(self) -> Dict[str, Any]:
        """
        预分配：将 part 的每个 guard 独占地分配给一个 req。

        问题根因：多个 REQ 共享同一 part（如 SafetyMonitor satisfy 5 个 SAFE REQ）时，
        若不加独占控制，对每个 REQ 调用 _match_by_content 都会返回相同的第一个匹配 guard，
        导致 BATT_FS_LOW_PCT=25 被写多次而 BATT_FS_CRT_PCT=15 从未被写。

        算法：
          按 req_id 排序（确定性），对每个 req 贪心地分配第一个未被占用的 guard。
          返回 {req_id: {"part": str, "guard": GuardCondition, "entry": ContentEntry}}
          或    {req_id: None}（无 guard 可分配）
        """
        assignment: Dict[str, Any] = {}
        claimed: set = set()  # (part_name, guard_attr, guard_op)

        for req_id in sorted(self._covered_req_ids()):
            part_names = self._satisfy_map.get(req_id, [])
            expected_family = self._requirement_family(req_id)
            # Doc text present but NO safety-family signal (IP54, regulatory,
            # environmental...) → this requirement must not claim any safety guard
            # merely because it satisfies the same part (guard-side analog of the
            # attr req_text_kws gate). Docless requirements keep legacy matching.
            if expected_family is None and self._req_texts.get(req_id, "").strip():
                continue
            entries = self._family_guard_entries(
                expected_family, self._preferred_semantic_tags(req_id)
            )

            if self._claim_guard(req_id, entries, part_names, claimed, assignment):
                continue
            # Multiple same-family requirements can legitimately point to the
            # same physical fault guard (e.g. self-test inhibit + alert). Let
            # them share a same-family guard rather than falling through to an
            # unrelated AST/LLM tag that traceability then has to block.
            if expected_family is not None and self._claim_guard(
                req_id, entries, part_names, claimed, assignment,
                allow_claimed=True,
            ):
                continue
            # Cross-family fallback (reached for docless requirements, and for
            # family-known requirements with no same-family guard — where the
            # wrong-family match must surface as a visible TRACE block per A4,
            # not silently vanish).
            all_guard_entries = [
                entry for entry in _CONTENT_CATALOGUE
                if entry.guard_matcher is not None
            ]
            self._claim_guard(
                req_id, all_guard_entries, part_names, claimed, assignment
            )

        return assignment

    def _family_guard_entries(
        self,
        expected_family: Optional[str],
        preferred_tags: List[str],
    ) -> List[ContentEntry]:
        """Guard-matcher catalogue entries for one requirement: filtered to the
        requirement's family (when known) and sorted so its preferred semantic
        tags come first."""
        entries = [
            entry for entry in _CONTENT_CATALOGUE
            if entry.guard_matcher is not None
            and (
                expected_family is None
                or self._tag_family(entry.semantic_tag) == expected_family
            )
        ]
        if preferred_tags:
            entries.sort(key=lambda e: (
                preferred_tags.index(e.semantic_tag)
                if e.semantic_tag in preferred_tags else len(preferred_tags)
            ))
        return entries

    def _claim_guard(
        self,
        req_id: str,
        entries: List[ContentEntry],
        part_names: List[str],
        claimed: set,
        assignment: Dict[str, Any],
        allow_claimed: bool = False,
    ) -> bool:
        """Greedily claim the first matching (part, guard) for *req_id* from
        *entries*, recording it in *assignment* and marking the guard claimed.

        A guard already claimed by another requirement is skipped unless
        ``allow_claimed`` — the caller enables that only for same-family
        sharing.  Returns True when an assignment was made.

        Entries may carry ``req_text_kws`` / ``req_text_exclude_kws`` gates
        (the guard-side mirror of the attr-path text gate): a same-family
        requirement about the OPPOSITE response (e.g. "release the payload")
        must not claim a lock-semantics guard merely because both satisfy the
        same part."""
        req_text = self._requirement_match_text(req_id)
        req_blob = f"{req_id} {req_text}".lower()
        for entry in entries:
            if entry.req_text_kws and req_text and not any(
                kw in req_blob for kw in entry.req_text_kws
            ):
                continue
            if entry.req_text_exclude_kws and any(
                kw in req_blob for kw in entry.req_text_exclude_kws
            ):
                continue
            gm = entry.guard_matcher
            for pname in part_names:
                for guard in self._guard_map.get(pname, []):
                    if not self._guard_satisfies(guard, gm, pname):
                        continue
                    gkey = (pname,
                            getattr(guard, "attribute", ""),
                            getattr(guard, "operator", ""))
                    if gkey in claimed and not allow_claimed:
                        continue
                    if gkey not in claimed:
                        claimed.add(gkey)
                    assignment[req_id] = {
                        "part":  pname,
                        "guard": guard,
                        "entry": entry,
                    }
                    return True
        return False

    def _build_ast_specs(self):
        """
        用 AST 合成器从 guard 变量名 + entry action 名直接推导
        InjectSpec/VerifySpec，建立 part_name → List[SynthesizedSpec] 索引。

        不依赖 LLM，<1ms，Phase 2 标准命名后接近 100% 覆盖。
        """
        try:
            from src.sitl.ast_synthesizer import synthesize_specs
            return synthesize_specs(self._model, verbose=self._verbose)
        except Exception as e:
            if self._verbose:
                print(f"  [AST-SYN] failed: {e}")
            return {}

    def _build_semantic_map(self) -> Dict[str, str]:
        """
        用 LLM 给状态机打语义标签，建立 req_id → tag 映射。

        核心逻辑：一个 part 可能有多个状态机（多个安全行为），不能简单地
        用 part_name → tag（后写的会覆盖前面）。正确做法是把每个 req 的
        guard 变量与该 part 的状态机 guard 做交叉匹配，选最接近的 tag。
        """
        if self._llm is None:
            return {}
        try:
            from src.sitl.semantic_classifier import classify_state_machines
        except ImportError:
            return {}
        try:
            sm_to_tag = classify_state_machines(
                self._model, self._llm, verbose=self._verbose
            )
        except Exception as e:
            if self._verbose:
                print(f"  [SEMANTIC] classify failed: {e}")
            return {}

        # 建立 state_def_name → (tag, owner_part, guard_vars) 的详细表
        try:
            from src.simulation.state_extractor import extract_state_machines
            text = self._model.to_sysml_text() or ""
            sms = extract_state_machines(text)
        except Exception as exc:
            from src.utils.suppressed import record_suppressed
            record_suppressed("sitl.requirement_linker.semantic_state_extract", exc)
            sms = []

        # sm_name → (owner_part, guard 涉及的所有属性名)
        sm_detail: Dict[str, Tuple[str, List[str]]] = {}
        for sm in sms:
            guard_attrs: List[str] = []
            for t in sm.fault_transitions():
                for g in t.guards:
                    attr = getattr(g, "attribute", None)
                    if attr:
                        guard_attrs.append(attr.lower())
            sm_detail[sm.name] = (sm.owner_part, guard_attrs)

        # 为每个 req 找最匹配的 tag
        # 策略：req 满足的所有 part 中，找 guard 变量与该 req 最接近的 sm
        req_tag: Dict[str, str] = {}
        for req_id in self._satisfy_map:
            part_names = self._satisfy_map[req_id]  # 现在是 list

            # 收集所有满足该 REQ 的 part 下的候选 sm
            candidate_sms = [
                (sm_name, sm_to_tag[sm_name], detail)
                for sm_name, detail in sm_detail.items()
                if detail[0] in part_names and sm_name in sm_to_tag
            ]
            if not candidate_sms:
                continue

            # 如果只有一个 sm，直接用
            if len(candidate_sms) == 1:
                req_tag[req_id] = candidate_sms[0][1]
                continue

            # 多个 sm：用 req_id 关键词与 sm guard 属性做最佳匹配
            # 把 req_id 拆成小写词（REQ_FUNC_007 → ["func","007"]）
            req_tokens = set(req_id.lower().replace("_", " ").split())

            best_tag, best_score = None, -1
            for sm_name, tag, (owner, guard_attrs) in candidate_sms:
                # 先用 tag 关键词打分
                tag_tokens = set(tag.lower().replace("_", " ").split())
                score = len(req_tokens & tag_tokens & set(
                    ["battery","batt","gcs","comm","link","sensor","parachute",
                     "propulsion","engine","payload","abort"]
                ))
                # 如果 guard 属性明确出现在 req_id 里，加分
                for attr in guard_attrs:
                    if any(tok in attr for tok in req_tokens):
                        score += 2
                if score > best_score:
                    best_score, best_tag = score, tag

            # 只有打到分才采用，否则不误判
            if best_tag and best_score > 0:
                req_tag[req_id] = best_tag

        return req_tag

    def _build_direct_param_map(self) -> Dict[str, Dict]:
        """
        Layer 3b（第四层 fallback）：用 LLM 从需求文本直接推断 ArduPilot 参数。

        只处理层1-3均未命中的需求（FUNC/PERF/CONS/INTF 类）。
        输入：需求文本（从 SysML doc 注释提取）。
        输出：{req_id: {param: str, value: float, tier: str}}。
        """
        if self._llm is None:
            return {}
        try:
            from src.sitl.semantic_classifier import (
                suggest_params_from_requirements,
                extract_req_texts_from_model,
            )
        except ImportError:
            return {}

        # 找出层1-3均未命中的 req（有 satisfy 关系但无 catalogue/AST/LLM 匹配）
        unmapped_reqs: Dict[str, str] = {}
        all_req_texts = extract_req_texts_from_model(self._model)

        for req_id in self._covered_req_ids():
            # 层2：AST 合成命中
            parts = self._satisfy_map.get(req_id, [])
            ast_hit = any(self._ast_specs.get(p) for p in parts)
            if ast_hit:
                continue
            # 层3：LLM 语义标签命中
            if req_id in self._semantic_map:
                continue
            # 三层全未命中 → 送给 Layer 3b
            text = all_req_texts.get(req_id, "")
            if text:
                unmapped_reqs[req_id] = text

        if not unmapped_reqs:
            return {}

        result = suggest_params_from_requirements(
            unmapped_reqs, self._llm, verbose=self._verbose
        )
        return result

    # ------------------------------------------------------------------
    # 内容匹配核心方法（Layer 1 — 完全不依赖 REQ ID）
    # ------------------------------------------------------------------

    def _entry_to_dict(self, entry: ContentEntry,
                       guard_val: Optional[float] = None,
                       guard_src: str = "",
                       attr_val: Optional[float] = None,
                       attr_src: str = "") -> Dict[str, Any]:
        """
        将 ContentEntry 转换为 _resolve_all 期望的 dict 格式。

        guard_val / attr_val：匹配时已解析的数值，存入 _resolved_* 键，
        供 _resolve_all 跳过 threshold_slot 机制直接使用。
        """
        result = {
            "semantic_tag":        entry.semantic_tag,
            "threshold_slot":      None,          # 内容匹配不再需要
            "ardu_params":         entry.ardu_params,
            "_resolved_guard_val": guard_val,
            "_resolved_guard_src": guard_src,
            "_resolved_attr_val":  attr_val,
            "_resolved_attr_src":  attr_src,
            "_attr_multiplier":    entry.attr_matcher.multiplier
                                   if entry.attr_matcher else 1.0,
            "sitl_test": {
                "tier":    entry.tier,
                "inject":  entry.inject or InjectSpec(kind="noop"),
                "verify":  entry.verify or VerifySpec(kind="noop"),
                "notes":   entry.notes,
            },
        }
        if entry.l2_inject is not None and entry.l2_verify is not None:
            # 附加 L2 行为检查（主 tier 的参数一致性证据保持不变）
            result["l2_test"] = {
                "tier":   "L2",
                "inject": entry.l2_inject,
                "verify": entry.l2_verify,
                "notes":  entry.l2_notes,
            }
        return result

    def _guard_satisfies(self, guard, gm: GuardMatcher,
                          part_name: str) -> bool:
        """判断一条 guard（或 accept 伪 guard）是否匹配 GuardMatcher。"""
        kind = getattr(guard, "kind", "")
        var  = getattr(guard, "attribute", "").lower()

        if kind == "accept_event":
            # accept 事件转移：仅显式声明 "event" 的条目可匹配；
            # 按事件类型名做关键词匹配，action_kws 检查目标态 entry action。
            if "event" not in gm.operators:
                return False
            if not any(kw in var for kw in gm.var_keywords):
                return False
            if gm.action_kws:
                action = self._find_guard_action(part_name, guard)
                if not action or not any(
                    kw in action.lower() for kw in gm.action_kws
                ):
                    return False
            return True

        if "bool" in gm.operators and kind == "bool_true":
            if not any(kw in var for kw in gm.var_keywords):
                return False
            # 可选：检查 entry action 关键词
            if gm.action_kws:
                action = self._find_guard_action(part_name, guard)
                if not action or not any(
                    kw in action.lower() for kw in gm.action_kws
                ):
                    return False
            return True
        if "bool" in gm.operators and "event" not in gm.operators:
            # bool 专属条目遇到非 bool guard：保持原有拒绝语义
            return False

        # comparison guard
        if kind != "comparison":
            return False
        op = getattr(guard, "operator", "")
        if op not in gm.operators:
            return False
        if not any(kw in var for kw in gm.var_keywords):
            return False
        th = getattr(guard, "threshold", 0.0)
        if not (gm.threshold_min <= th <= gm.threshold_max):
            return False
        return True

    def _find_guard_action(self, part_name: str, guard) -> Optional[str]:
        """
        在 part 的状态机里找到包含该 guard 的转移，返回其 target state 的
        entry action 名。用于 GuardMatcher.action_kws 区分语义。
        """
        try:
            from src.simulation.state_extractor import extract_state_machines
            text = self._model.to_sysml_text() or ""
            wanted_event = (
                getattr(guard, "attribute", "")
                if getattr(guard, "kind", "") == "accept_event" else None
            )
            for sm in extract_state_machines(text):
                if sm.owner_part != part_name:
                    continue
                if wanted_event is not None:
                    for tr in sm.transitions:
                        if tr.is_initial or tr.accept_trigger != wanted_event:
                            continue
                        action = sm.entry_action_for_state(tr.target or "")
                        if action:
                            return action
                    continue
                for tr in sm.fault_transitions():
                    for g in tr.guards:
                        if (getattr(g, "attribute", "") ==
                                getattr(guard, "attribute", "") and
                                getattr(g, "operator", "") ==
                                getattr(guard, "operator", "")):
                            action = sm.entry_action_for_state(tr.target or "")
                            if action:
                                return action
        except Exception as exc:
            from src.utils.suppressed import record_suppressed
            record_suppressed("sitl.requirement_linker.entry_action_lookup", exc)
            pass
        return None

    def _guard_response_commands(self, part_name: str, guard) -> tuple[str, set[str]]:
        """Return entry-action usage and explicit sends for a guard target state."""
        try:
            from src.simulation.state_extractor import extract_state_machines
            text = self._model.to_sysml_text() or ""
            fallback_action = ""
            wanted_event = (
                getattr(guard, "attribute", "")
                if getattr(guard, "kind", "") == "accept_event" else None
            )
            for sm in extract_state_machines(text):
                if sm.owner_part != part_name:
                    continue
                if wanted_event is not None:
                    matched = [
                        tr for tr in sm.transitions
                        if not tr.is_initial and tr.accept_trigger == wanted_event
                    ]
                else:
                    matched = [
                        tr for tr in sm.fault_transitions()
                        if any(
                            getattr(g, "attribute", "") == getattr(guard, "attribute", "")
                            and getattr(g, "operator", "") == getattr(guard, "operator", "")
                            for g in tr.guards
                        )
                    ]
                for tr in matched:
                    state = next((s for s in sm.states if s.name == (tr.target or "")), None)
                    if state is None:
                        continue
                    fallback_action = fallback_action or (state.entry_action or "")
                    commands = {str(cmd).upper() for cmd, _port in state.sends}
                    if commands:
                        return state.entry_action or "", commands
            return fallback_action, set()
        except Exception as exc:
            from src.utils.suppressed import record_suppressed
            record_suppressed("sitl.requirement_linker.guard_response", exc)
            return "", set()

    def _guard_response_send_ports(self, part_name: str, guard) -> set[str]:
        """Ports the guard's response state sends through (route identity)."""
        try:
            from src.simulation.state_extractor import extract_state_machines
            text = self._model.to_sysml_text() or ""
            wanted_event = (
                getattr(guard, "attribute", "")
                if getattr(guard, "kind", "") == "accept_event" else None
            )
            ports: set[str] = set()
            for sm in extract_state_machines(text):
                if sm.owner_part != part_name:
                    continue
                if wanted_event is not None:
                    matched = [
                        tr for tr in sm.transitions
                        if not tr.is_initial and tr.accept_trigger == wanted_event
                    ]
                else:
                    matched = [
                        tr for tr in sm.fault_transitions()
                        if any(
                            getattr(g, "attribute", "") == getattr(guard, "attribute", "")
                            and getattr(g, "operator", "") == getattr(guard, "operator", "")
                            for g in tr.guards
                        )
                    ]
                for tr in matched:
                    state = next((s for s in sm.states if s.name == (tr.target or "")), None)
                    if state is None:
                        continue
                    ports.update(
                        str(port) for _cmd, port in state.sends if port
                    )
            return ports
        except Exception as exc:
            from src.utils.suppressed import record_suppressed
            record_suppressed("sitl.requirement_linker.guard_send_ports", exc)
            return set()

    def _terminal_route_ports(self, req_id: str) -> set[str]:
        """Ports of the requirement's causal path's FINAL hop, per its plan.

        The last hop is the terminal effect leg — for the parachute
        requirement, SafetyMonitor.recoveryCmd → RecoverySystem.recoveryCmd.
        A response that sends through it is on the requirement's own route.
        """
        from src.simulation.verification_binding import binding_for
        binding = binding_for(self._requirement_bindings, req_id)
        if binding is None or not binding.route:
            return set()
        last = binding.route[-1]
        return {port for port in (last[1], last[3]) if port}

    def _action_traceability_issue(
        self, req_id: str, tag: str
    ) -> Optional[Dict[str, str]]:
        """Reject an explicit response command that contradicts its semantic tag."""
        expected_commands = {
            "BATTERY_RTB": {"CMD_RTL"},
            "BATTERY_LAND": {"CMD_LAND"},
            "GCS_LOSS": {"CMD_LAND", "CMD_RTL"},
            "GCS_LOSS_LAND": {"CMD_LAND"},
            "GCS_LOSS_RTL": {"CMD_RTL"},
            "PARACHUTE_DEPLOY": {"CMD_PARACHUTE"},
        }
        expected = expected_commands.get(tag)
        assigned = self._guard_assignment.get(req_id)
        if not expected or not assigned:
            return None
        action, commands = self._guard_response_commands(
            assigned["part"], assigned["guard"]
        )
        # Empty action bodies remain abstract behavior allocations.  But once a
        # command is explicit, checking the wrong command is mandatory.
        semantic_match = bool(commands & expected)
        if tag == "PARACHUTE_DEPLOY":
            # Accept-side widening only — a CHUTE-spelled command is never a
            # ground for rejection, merely for acceptance.
            semantic_match = semantic_match or any(
                "PARACHUTE" in command or "CHUTE" in command
                for command in commands
            )
        if not semantic_match:
            # Route identity from the plan: a response sending through the
            # requirement's own terminal causal-path leg is the commanded
            # response, whatever the payload item is named. run3 sends
            # RecoveryCmdData through recoveryCmd (the SAFE_005 route) and
            # the substring check rejected it, collapsing L2 generation
            # from ~9 tests to 3.
            route_ports = self._terminal_route_ports(req_id)
            if route_ports:
                send_ports = self._guard_response_send_ports(
                    assigned["part"], assigned["guard"]
                )
                semantic_match = bool(send_ports & route_ports)
        if not commands or semantic_match:
            return None
        return {
            "req_id": req_id,
            "expected_family": tag,
            "matched_family": ",".join(sorted(commands)),
            "matched_tag": tag,
            "requirement_text": self._req_texts.get(req_id, ""),
            "message": (
                f"traceability mismatch: {tag} expects one of "
                f"{sorted(expected)}, but action {action} sends {sorted(commands)}"
            ),
        }

    def _match_by_content(self, req_id: str) -> Optional[Dict[str, Any]]:
        """
        Layer 1（内容驱动）：按 guard/attr 内容匹配，与 REQ ID 完全无关。

        guard 匹配优先使用 _guard_assignment 预分配结果（独占，防止多 REQ
        抢占同一 guard）。attr 匹配仍在运行时动态进行（属性不存在争用问题）。

        返回 _entry_to_dict(entry, ...) 格式的 dict，或 None。
        """
        part_names = self._satisfy_map.get(req_id)
        if not part_names:
            return None

        # ── Guard 匹配：使用独占预分配结果 ────────────────────────────
        assigned = getattr(self, "_guard_assignment", {}).get(req_id)
        if assigned is not None:
            pname = assigned["part"]
            guard = assigned["guard"]
            entry = assigned["entry"]
            th    = getattr(guard, "threshold", None)
            attr  = getattr(guard, "attribute", "?")
            op    = getattr(guard, "operator", "?")
            g_val = float(th) if th is not None else None
            g_src = f"guard:{op}:{th} (attr:{attr})"
            if self._verbose:
                print(f"  [CONTENT] {req_id} → {entry.semantic_tag} "
                      f"guard={attr!r} {op} {th} (part={pname})")
            return self._entry_to_dict(entry, guard_val=g_val, guard_src=g_src)

        # ── Attr 匹配：动态搜索（属性不存在争用）─────────────────────
        # req_text_kws gate：attr 匹配只证明 "satisfy 的 part 上有这个属性"，
        # 不证明 "这条需求关于这个量"。有需求文本时要求文本命中条目关键词，
        # 否则 MTOW/温度/法规类需求会被首个 attr 命中的条目错误标为 L1 PASS
        # （曾发生：FENCE_ALT_MAX "验证" 姿态 RMS、WPNAV_SPEED "验证" MTOW）。
        req_text = self._requirement_match_text(req_id)
        req_blob = f"{req_id} {req_text}".lower()
        for entry in _CONTENT_CATALOGUE:
            if entry.attr_matcher is None:
                continue
            if entry.req_text_kws and req_text and not any(
                kw in req_blob for kw in entry.req_text_kws
            ):
                continue
            if entry.req_text_exclude_kws and any(
                kw in req_blob for kw in entry.req_text_exclude_kws
            ):
                continue
            am = entry.attr_matcher
            for pname in part_names:
                if am.part_keywords and not any(
                    kw in pname.lower() for kw in am.part_keywords
                ):
                    continue
                attrs = self._attr_map.get(pname, {})
                for aname, aval in attrs.items():
                    if not any(kw in aname.lower() for kw in am.attr_keywords):
                        continue
                    resolved = (round(aval * am.multiplier)
                                if am.multiplier != 1.0 else aval)
                    src = (f"attr:{pname}.{aname}"
                           + (f"*{int(am.multiplier)}"
                              if am.multiplier != 1.0 else ""))
                    if self._verbose:
                        print(f"  [CONTENT] {req_id} → {entry.semantic_tag} "
                              f"attr={pname}.{aname}={aval}"
                              f"{'×'+str(int(am.multiplier)) if am.multiplier!=1.0 else ''}"
                              f"={resolved}")
                    return self._entry_to_dict(
                        entry, attr_val=resolved, attr_src=src
                    )
                # ── Port 匹配（attr 未命中时）：接口类 part 合法地只声明
                # port。port 没有数值，动态 token（@guard/@attr_match/@attr:）
                # 无从解析，因此静态 ardu_params 是硬前提——即使条目误开
                # allow_port_match 也不放行，落入诚实的 no-mapping。
                if am.allow_port_match and not self._has_dynamic_params(entry):
                    for port_name in self._port_map.get(pname, []):
                        if not any(
                            kw in port_name.lower() for kw in am.attr_keywords
                        ):
                            continue
                        if self._verbose:
                            print(f"  [CONTENT] {req_id} → {entry.semantic_tag} "
                                  f"port={pname}.{port_name} (static params)")
                        return self._entry_to_dict(
                            entry, attr_val=None,
                            attr_src=f"port:{pname}.{port_name}",
                        )

        return None

    @staticmethod
    def _has_dynamic_params(entry: ContentEntry) -> bool:
        """True when any ardu_params value needs a model-resolved number."""
        return any(
            isinstance(value, str) and value.startswith("@")
            for value in entry.ardu_params.values()
        )

    def _requirement_match_text(self, req_id: str) -> str:
        """Requirement prose used for semantic matching, without method tags.

        Verification annotations such as ``[V: hardware-in-the-loop]`` describe
        how evidence will be collected.  Letting their words participate in the
        matcher caused a waypoint CEP requirement to hit CONTROL_LOOP_RATE solely
        because the annotation contained the word ``loop``.
        """
        text = self._req_texts.get(req_id, "")
        return re.sub(r"\[(?:V|SEV)\s*:[^\]]*\]", " ", text,
                      flags=re.IGNORECASE).strip()

    # ------------------------------------------------------------------

    def _lookup_catalogue(self, req_id: str) -> Optional[Dict[str, Any]]:
        """
        五层 Fallback（内容优先，REQ ID 无关）：

          层1  _CONTENT_CATALOGUE 内容匹配（guard/attr）← 新，最健壮
          层2  AST 合成器（guard 变量名关键词）
          层3  LLM 状态机语义分类
          层3b LLM 需求文本直接推断
          层4  全部失败 → None
        """
        if req_id in self._catalogue_cache:
            return self._catalogue_cache[req_id]

        part_names = self._satisfy_map.get(req_id)
        if not part_names:
            self._catalogue_cache[req_id] = None
            return None

        result: Optional[Dict[str, Any]] = None

        # ── 层1：内容匹配（_CONTENT_CATALOGUE）────────────────────────
        result = self._match_by_content(req_id)

        # Part-level fallbacks (AST synthesis / LLM state-machine tags) map the
        # PART's safety behavior. With doc text present but no safety-family
        # signal, attributing that behavior to THIS requirement is unjustified —
        # same rule as the guard/attr text gates (IP54 must not inherit the
        # SafetyMonitor's GCS spec just because both satisfy the same part).
        _text_without_family = (
            bool(self._req_texts.get(req_id, "").strip())
            and self._requirement_family(req_id) is None
        )

        # ── 层2：AST 合成（遍历所有 satisfying parts）──────────────────
        if result is None and not _text_without_family:
            result = self._lookup_from_ast(req_id, part_names)

        # ── 层3：LLM 状态机语义标签 ────────────────────────────────────
        if result is None and not _text_without_family:
            tag = self._semantic_map.get(req_id)
            if tag and tag in self._tag_to_entry:
                if self._verbose:
                    print(f"  [SEMANTIC] {req_id} → tag={tag} (parts={part_names})")
                result = self._entry_to_dict(self._tag_to_entry[tag])

        # ── 层3b：LLM 需求文本直接推断 ─────────────────────────────────
        if result is None:
            result = self._lookup_from_direct_param(req_id)

        result = self._apply_traceability_gate(req_id, result)
        self._catalogue_cache[req_id] = result
        return result

    def _lookup_from_ast(
        self, req_id: str, part_names: List[str]
    ) -> Optional[Dict[str, Any]]:
        """层2 fallback：AST 合成器。

        遍历 satisfying parts 的合成 spec，把匹配 guard 的阈值解析进
        ``@guard`` 占位符——避免模型里明明有 guard 却漏成
        <unresolved:guard>。AST fallback 可以从 guard/action 名推断 inject，
        但 S4 已把这些目录 tag 固定为确定性的 MAVLink conformance verify，
        不让旧的 wait_statustext 模板重新引入 run-to-run flaky 验证。
        """
        for pname in part_names:
            ast_candidates = self._ast_specs.get(pname, [])
            if not ast_candidates:
                continue
            best = self._best_ast_spec(req_id, pname, ast_candidates)
            if best is None:
                continue
            base_entry = self._tag_to_entry.get(best.tag)
            if not base_entry:
                continue
            g_val, g_src = self._threshold_for_guard_var(
                pname, best.guard_var,
                base_entry.guard_matcher.operators
                if base_entry.guard_matcher else None,
            )
            if self._verbose:
                print(f"  [AST-SYN] {req_id} → tag={best.tag} "
                      f"guard={best.guard_var!r} (part={pname})"
                      + (f" thr={g_val}" if g_val is not None else ""))
            d = self._entry_to_dict(base_entry, guard_val=g_val, guard_src=g_src)
            verify = best.verify
            if (
                base_entry.verify is not None
                and base_entry.verify.kind in {
                    "assert_servo_pwm",
                    "assert_sensor_unhealthy",
                    "assert_arm_rejected",
                    "wait_mode",
                }
            ):
                verify = base_entry.verify
            d["sitl_test"] = {
                "tier":   base_entry.tier,
                "inject": best.inject,
                "verify": verify,
                "notes":  base_entry.notes,
            }
            return d
        return None

    def _lookup_from_direct_param(self, req_id: str) -> Optional[Dict[str, Any]]:
        """层3b fallback：LLM 从需求文本直接推断的参数建议。"""
        direct = self._direct_param_map.get(req_id)
        if not direct:
            return None
        param_name = direct["param"]
        value      = direct["value"]
        tier       = direct.get("tier", "L1")
        if self._verbose:
            print(f"  [REQ-PARAM] {req_id} → {param_name}={value} [{tier}]")
        return {
            "semantic_tag":        f"LLM_DIRECT:{param_name}",
            "threshold_slot":      None,
            "ardu_params":         {param_name: value},
            "_resolved_guard_val": None,
            "_resolved_attr_val":  None,
            "sitl_test": {
                "tier":   tier,
                "inject": InjectSpec(kind="noop"),
                "verify": VerifySpec(kind="noop"),
                "notes":  "LLM-suggested param from requirement text.",
            },
        }

    def _apply_traceability_gate(
        self, req_id: str, result: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Reject semantically inconsistent mappings without breaking flow.

        The linker intentionally derives tests from model behavior (guard/action
        content). A4 adds the missing cross-check: if the requirement text says
        GCS loss but the model behavior maps to PARACHUTE, do not emit the
        parachute test under that requirement ID. Surface a deterministic
        traceability mismatch instead.
        """
        contract_issue = self._contract_traceability_issue(req_id)
        if contract_issue is not None:
            self._traceability_mismatches[req_id] = contract_issue
            return {
                "semantic_tag": "TRACEABILITY_MISMATCH:CONTRACT_TRACE",
                "threshold_slot": None,
                "ardu_params": {},
                "_resolved_guard_val": None,
                "_resolved_attr_val": None,
                "_traceability_mismatch": contract_issue,
                "sitl_test": {
                    "tier": "TRACE",
                    "inject": InjectSpec(kind="skip", notes=contract_issue["message"]),
                    "verify": VerifySpec(kind="skip", notes=contract_issue["message"]),
                    "notes": contract_issue["message"],
                },
            }
        if result is None:
            self._traceability_mismatches.pop(req_id, None)
            return None

        tag = str(result.get("semantic_tag", ""))
        expected = self._requirement_family(req_id)
        matched = self._tag_family(tag)
        detail = None
        if expected is not None and matched is not None and expected != matched:
            detail = {
                "req_id": req_id,
                "expected_family": expected,
                "matched_family": matched,
                "matched_tag": tag,
                "requirement_text": self._req_texts.get(req_id, ""),
                "message": (
                    f"traceability mismatch: requirement text implies {expected}, "
                    f"but guard/action mapping selected {tag} ({matched})"
                ),
            }
        if detail is None:
            detail = self._response_traceability_issue(req_id, tag)
        if detail is None:
            detail = self._action_traceability_issue(req_id, tag)
        if detail is None:
            self._traceability_mismatches.pop(req_id, None)
            return result

        self._traceability_mismatches[req_id] = detail
        if self._verbose:
            print(f"  [TRACE-MISMATCH] {req_id}: {detail['message']}")
        return {
            "semantic_tag": f"TRACEABILITY_MISMATCH:{tag}",
            "threshold_slot": None,
            "ardu_params": {},
            "_resolved_guard_val": None,
            "_resolved_attr_val": None,
            "_traceability_mismatch": detail,
            "sitl_test": {
                "tier": "TRACE",
                "inject": InjectSpec(kind="skip", notes=detail["message"]),
                "verify": VerifySpec(kind="skip", notes=detail["message"]),
                "notes": detail["message"],
            },
        }

    def _contract_traceability_issue(self, req_id: str) -> Optional[Dict[str, str]]:
        """Expose contract-first trace faults before a model-derived test is emitted."""
        findings = self._contract_trace_findings.get(req_id, ())
        if not findings:
            return None
        first = findings[0]
        if isinstance(first, dict):
            code = str(first.get("finding_code", "SEMANTIC_TRACE_FAILED"))
            expected = first.get("expected", {})
            observed = first.get("observed", {})
        else:
            code = str(getattr(first, "finding_code", "SEMANTIC_TRACE_FAILED"))
            expected = getattr(first, "expected", {})
            observed = getattr(first, "observed", {})
        return {
            "req_id": req_id,
            "expected_family": "CONTRACT_TRACE",
            "matched_family": "MODEL_TRACE",
            "matched_tag": code,
            "requirement_text": self._req_texts.get(req_id, ""),
            "message": (
                f"contract-first trace blocked test generation: {code}; "
                f"expected={dict(expected)!r}; observed={dict(observed)!r}"
            ),
        }

    def traceability_mismatches(self) -> List[Dict[str, str]]:
        """Return SAFE requirement text/tag family mismatches discovered so far."""
        for req_id in sorted(self._covered_req_ids()):
            self._lookup_catalogue(req_id)
        return [dict(v) for _, v in sorted(self._traceability_mismatches.items())]

    @classmethod
    def static_traceability_issues(cls, model_text: str,
                                   model_name: str = "Model") -> List[str]:
        """Refinement-actionable issues for response-traceability mismatches.

        Runs the SAME deterministic gate that later withholds SITL evidence
        (no LLM, no SITL process) against the in-session model text, so a
        response action that emits the wrong command family — measured on
        8 of 24 archived runs as a blocked matrix row discovered only at
        Phase 9 — reaches the refinement loop while the author can still
        repair it.  Silence on any parse/link failure: this is an advisory
        projection, never a new failure mode for generation.
        """
        try:
            from src.sysml.lite_model import build_lite_model
            model = build_lite_model(str(model_text or ""), model_name=model_name)
            mismatches = cls(model, llm=None).traceability_mismatches()
        except Exception as exc:  # advisory projection must never break the loop
            from src.utils.suppressed import record_suppressed
            record_suppressed("sitl.requirement_linker.static_trace_issues", exc)
            return []
        issues: List[str] = []
        for item in mismatches:
            message = str(item.get("message") or "traceability mismatch")
            issues.append(
                f"[SITL-TRACE] {item.get('req_id')}: {message}. The flight-stack "
                "test for this requirement is derived from the response the "
                "model actually emits, so this mapping will be withheld at "
                "verification. Repair the response action so its send matches "
                "the requirement's commanded response (e.g. a parachute "
                "requirement's response must send the parachute-command "
                "payload through its command port, not the detected-failure "
                "event or a generic mode command); keep the triggering "
                "guard/accept unchanged."
            )
        return issues

    def _extract_requirement_texts(self) -> Dict[str, str]:
        try:
            text = self._model.to_sysml_text() if self._model else ""
        except Exception:
            return {}
        if not text:
            return {}
        result: Dict[str, str] = {}
        for m in re.finditer(
            r"requirement\s+def\s+([A-Za-z_][\w]*)\s*\{(?P<body>.*?)\}",
            text,
            re.S,
        ):
            req_id = m.group(1)
            body = m.group("body")
            doc = re.search(r"doc\s*/\*(.*?)\*/", body, re.S)
            result[req_id] = (doc.group(1).strip() if doc else body.strip())
        return result

    @staticmethod
    def _tag_family(tag: str) -> Optional[str]:
        if tag.startswith("LLM_DIRECT:") or tag.startswith("TRACEABILITY_MISMATCH:"):
            return None
        tag = tag.upper()
        if tag.startswith("BATTERY_"):
            return "BATTERY"
        if tag.startswith("GCS_LOSS"):
            return "GCS"
        if tag.startswith("SENSOR_"):
            return "SENSOR"
        if tag == "PARACHUTE_DEPLOY":
            return "PARACHUTE"
        if tag.startswith("PAYLOAD_"):
            return "PAYLOAD"
        if tag in {"ALTITUDE_FENCE", "RADIUS_FENCE"}:
            return "GEOFENCE"
        return None

    def _requirement_family(self, req_id: str) -> Optional[str]:
        families = self._requirement_families(req_id)
        return next(iter(families)) if len(families) == 1 else None

    def _requirement_families(self, req_id: str) -> set[str]:
        """Infer fault-trigger families, excluding ordinary component mentions.

        A requirement that merely uploads data to the GCS is not a GCS-loss
        requirement. A compound contingency naming several trigger families
        cannot be proven by exercising only one guard.
        """
        contract = self._contracts.get(req_id)
        if contract is not None and contract.obligations:
            families = {
                {
                    "battery_state_of_charge": "BATTERY",
                    "sensor_self_test_failure": "SENSOR",
                    "gcs_link_absent": "GCS",
                    "delivery_abort_condition": "PAYLOAD",
                    "delivery_waypoint_proximity": "PAYLOAD",
                    "delivery_coordinate_condition_satisfied": "PAYLOAD",
                    "critical_propulsion_failure": "PARACHUTE",
                }.get(obligation.trigger.concept)
                for obligation in contract.obligations
                if obligation.trigger is not None
            }
            return {family for family in families if family}

        text = self._req_texts.get(req_id, "")
        low = f"{req_id} {text}".lower()
        families: set[str] = set()

        def has(*terms: str) -> bool:
            return any(re.search(
                rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])", low
            ) for term in terms)

        if has(
            "battery", "state-of-charge", "state of charge", "soc",
            "low voltage", "charge",
        ):
            families.add("BATTERY")
        sensor_entity = has("sensor", "gps", "gnss", "imu", "magnetometer")
        sensor_fault = has(
            "self-test", "self test", "prearm", "pre-arm", "arming",
            "failure", "failed", "fault", "unhealthy",
        )
        if has("self-test", "self test", "prearm", "pre-arm") or (
            sensor_entity and sensor_fault
        ):
            families.add("SENSOR")
        gcs_entity = has(
            "gcs", "ground control", "uplink", "communication", "comm",
            "data link", "datalink", "telemetry", "heartbeat",
        )
        gcs_fault = has(
            "loss", "lost", "absent", "interrupted", "disconnect",
            "unavailable", "outage", "timeout", "failsafe", "fail-safe",
        )
        if gcs_entity and gcs_fault:
            families.add("GCS")
        if has(
            "delivery abort", "delivery-abort", "abort condition", "gripper",
            "payload lock", "lock payload", "locked payload",
            "payload in the mechanically locked",
            "payload-release actuator shall default",
            "default to the mechanically locked",
        ):
            families.add("PAYLOAD")
        if has(
            "geofence", "geo-fence", "fence", "airspace", "segregated airspace",
            "boundary", "boundaries", "outside designated", "deviates",
            "operational radius", "flight radius",
        ):
            families.add("GEOFENCE")
        if has(
            "parachute", "propulsion failure", "propulsion subsystem failure",
            "engine failure", "motor failure", "critical propulsion",
        ):
            families.add("PARACHUTE")
        return families

    def _response_traceability_issue(
        self, req_id: str, tag: str
    ) -> Optional[Dict[str, str]]:
        """Detect a no-response boundary mapped to a positive failsafe test."""
        if not tag.upper().startswith("GCS_LOSS"):
            return None
        low = self._requirement_match_text(req_id).lower()
        forbids_response = any(phrase in low for phrase in (
            "without initiating", "shall not initiate", "must not initiate",
            "shall not enter", "must not enter",
        ))
        bounded_below_trigger = any(phrase in low for phrase in (
            "or less", "no more than", "less than", "before the", "until the",
        ))
        if not (forbids_response and bounded_below_trigger):
            return None
        return {
            "req_id": req_id,
            "expected_family": "GCS_NO_RESPONSE_BOUNDARY",
            "matched_family": "GCS_POSITIVE_FAILSAFE",
            "matched_tag": tag,
            "requirement_text": self._req_texts.get(req_id, ""),
            "message": (
                "traceability mismatch: requirement forbids a GCS failsafe "
                "response inside the boundary, but the selected test only proves "
                "a positive response after link loss"
            ),
        }

    def _preferred_semantic_tags(self, req_id: str) -> List[str]:
        """Return deterministic tag preferences within a broad requirement family."""
        contract = self._contracts.get(req_id)
        if contract is not None and contract.obligations:
            preferences: list[str] = []
            for obligation in contract.obligations:
                trigger = obligation.trigger.concept if obligation.trigger else ""
                response = obligation.response.concept if obligation.response else ""
                tag = {
                    ("battery_state_of_charge", "return_to_base"): "BATTERY_RTB",
                    ("battery_state_of_charge", "controlled_landing"): "BATTERY_LAND",
                    ("gcs_link_absent", "controlled_landing"): "GCS_LOSS_LAND",
                    ("gcs_link_absent", "return_to_base"): "GCS_LOSS_RTL",
                    ("sensor_self_test_failure", "prevent_arming"): "SENSOR_ARMING_INHIBIT",
                    ("sensor_self_test_failure", "alert_gcs"): "SENSOR_GROUND_ALERT",
                    ("critical_propulsion_failure", "deploy_parachute"): "PARACHUTE_DEPLOY",
                    ("delivery_abort_condition", "lock_payload"): "PAYLOAD_ABORT_LOCK",
                }.get((trigger, response))
                if tag and tag not in preferences:
                    preferences.append(tag)
            if preferences:
                return preferences
        text = self._req_texts.get(req_id, "")
        low = f"{req_id} {text}".lower()
        family = self._requirement_family(req_id)
        if family == "SENSOR":
            if any(kw in low for kw in (
                "not transition", "shall not transition", "not arm",
                "armed or airborne", "prevent arming",
            )):
                return ["SENSOR_ARMING_INHIBIT", "SENSOR_GROUND_ALERT"]
            if any(kw in low for kw in (
                "issue a failure alert", "failure alert", "alert", "notify",
                "ground control", "gcs",
            )):
                return ["SENSOR_GROUND_ALERT", "SENSOR_ARMING_INHIBIT"]
            if any(kw in low for kw in (
                "inhibit", "pre-flight startup inhibit",
                "preflight startup inhibit", "prevent arming", "arming",
            )):
                return ["SENSOR_ARMING_INHIBIT", "SENSOR_GROUND_ALERT"]
            return ["SENSOR_ARMING_INHIBIT", "SENSOR_GROUND_ALERT"]
        if family == "BATTERY":
            if any(kw in low for kw in ("land", "landing", "descent", "below 15", "15%")):
                return ["BATTERY_LAND", "BATTERY_RTB"]
            if any(kw in low for kw in ("return", "rtb", "rtl", "base")):
                return ["BATTERY_RTB", "BATTERY_LAND"]
        if family == "GCS":
            # Action from the requirement TEXT: land-only → the failsafe action must
            # be LAND (RTL is a wrong action, not a pass); return-only → RTL. Text
            # naming both (or neither) keeps the lenient legacy entry — either
            # failsafe reaction satisfies such a requirement.
            has_land = any(kw in low for kw in (
                "land", "landing", "descend", "descent", "current position",
            ))
            has_rtl = any(kw in low for kw in ("return", "rtl", "rtb", "base", "home"))
            if has_land and not has_rtl:
                return ["GCS_LOSS_LAND", "GCS_LOSS", "GCS_LOSS_RTL"]
            if has_rtl and not has_land:
                return ["GCS_LOSS_RTL", "GCS_LOSS", "GCS_LOSS_LAND"]
            return ["GCS_LOSS", "GCS_LOSS_RTL", "GCS_LOSS_LAND"]
        return []

    def _best_ast_spec(self, req_id: str, part_name: str, candidates):
        """
        从同一 part 的多个 AST 合成结果中，选与该 req 的 guard 最匹配的那个。
        利用 _guard_map 里该 part 的 guard 属性集合做交叉比对。
        """
        if len(candidates) == 1:
            return candidates[0]

        # 用该 req part 的 guard 属性名做匹配
        part_guards = self._guard_map.get(part_name, [])
        part_guard_attrs = {
            getattr(g, "attribute", "").lower()
            for g in part_guards
        }

        best, best_score = None, -1
        for cand in candidates:
            score = 0
            if cand.guard_var.lower() in part_guard_attrs:
                score += 3   # guard 变量直接匹配
            # req_id 词元 vs candidate tag 词元
            req_tokens = set(req_id.lower().replace("_", " ").split())
            tag_tokens = set(cand.tag.lower().replace("_", " ").split())
            score += len(req_tokens & tag_tokens)
            if score > best_score:
                best_score, best = score, cand

        return best if best_score >= 0 else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # 这些参数在 SITL 里设置会破坏仿真（如 GPS、EKF），
    # 只做 L1 验证（检 resolved 值），不写进 .parm 文件
    _SITL_SKIP_PARAMS: set = {
        "SCHED_LOOP_RATE",  # 改调度频率会破坏 SITL GPS 仿真，让 SITL 用默认 400Hz
    }

    def compile_evidence(self) -> RequirementEvidenceBundle:
        """Compile one immutable evidence bundle for this model revision."""
        if self._evidence_bundle is not None:
            return self._evidence_bundle

        model_text = self._model.to_sysml_text() or ""
        assignments: Dict[str, GuardEvidence] = {}
        for req_id, assigned in self._guard_assignment.items():
            if not assigned:
                continue
            guard = assigned.get("guard")
            assignments[req_id] = GuardEvidence(
                part_name=str(assigned.get("part", "")),
                attribute=str(getattr(guard, "attribute", "?")),
                kind=str(getattr(guard, "kind", "")),
                operator=str(getattr(guard, "operator", "")),
                threshold=getattr(guard, "threshold", None),
                signature=(
                    getattr(guard, "kind", None),
                    getattr(guard, "attribute", None),
                    getattr(guard, "operator", None),
                    getattr(guard, "threshold", None),
                    getattr(guard, "enum_type", None),
                    getattr(guard, "enum_value", None),
                ),
            )

        specs = tuple(self.generate_test_specs())
        resolved = tuple(self._resolve_all())
        mismatches = tuple(
            MappingProxyType(dict(item))
            for item in self.traceability_mismatches()
        )
        coverage = MappingProxyType({
            key: tuple(value) if isinstance(value, list) else value
            for key, value in self.coverage_stats().items()
        })
        self._evidence_bundle = RequirementEvidenceBundle(
            model_digest=sha256_text(model_text),
            requirement_texts=MappingProxyType(dict(self._req_texts)),
            satisfying_parts=MappingProxyType({
                req_id: tuple(parts)
                for req_id, parts in self._satisfy_map.items()
            }),
            guard_assignments=MappingProxyType(assignments),
            test_specs=specs,
            resolved_params=resolved,
            parm_file=self.generate_parm_file(),
            traceability_mismatches=mismatches,
            coverage=coverage,
        )
        return self._evidence_bundle

    def generate_parm_file(self) -> str:
        resolved = self._resolve_all()
        lines = [
            "# Auto-generated by requirement_linker.py (AST-based)",
            f"# Source model: {self._model.name}",
            "",
        ]
        seen: Dict[str, str] = {}
        for rp in resolved:
            val = self._format_value(rp.value)
            # 跳过未解析的值（ArduCopter 解析到非数字会行为异常）
            if isinstance(rp.value, str) and rp.value.startswith("<"):
                lines.append(f"# SKIPPED {rp.param_name}: {val}  [{rp.source}]")
                continue
            if isinstance(rp.value, str) and "@" in rp.value:
                lines.append(f"# SKIPPED {rp.param_name}: {val}  [{rp.source}]")
                continue
            # 跳过 SITL 不兼容参数（L1 仍验证 resolved 值，但不写进 .parm）
            if rp.param_name in self._SITL_SKIP_PARAMS:
                lines.append(f"# SITL-SKIP {rp.param_name}: {val}  (L1-validated, not loaded into SITL)")
                continue
            comment = f"  # {rp.req_id} ({rp.part_name}) [{rp.source}]"
            key = rp.param_name
            if key not in seen:
                lines.append(f"{key:<30} {val}{comment}")
                seen[key] = val
            elif seen[key] != val:
                lines.append(f"# CONFLICT {key}: {seen[key]} vs {val} ({rp.req_id})")

        # CHUTE_ALT_MIN=0：测试时不限制降落伞触发高度
        if "CHUTE_ENABLED" in seen and "CHUTE_ALT_MIN" not in seen:
            lines.append(f"{'CHUTE_ALT_MIN':<30} 0  # SITL test: disable alt threshold")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _bind_verify_args(verify, attr_val):
        """Resolve ``@attr_match`` inside a verify spec's args.

        A threshold a check compares against has to come from the model, the
        same way an ``ardu_params`` value does. When it cannot be resolved the
        spec is DROPPED rather than run against a default: a check that invents
        its own limit reports a verdict about nothing.
        """
        if verify is None:
            return verify
        args = dict(getattr(verify, "args", {}) or {})
        if not any(v == "@attr_match" for v in args.values()):
            return verify
        if attr_val is None or float(attr_val) <= 0:
            return None
        for key, value in list(args.items()):
            if value == "@attr_match":
                args[key] = float(attr_val)
        return replace(verify, args=args)

    def generate_test_specs(self) -> List[SITLTestSpec]:
        specs: List[SITLTestSpec] = []
        resolved_by_req: Dict[str, List[ResolvedParam]] = {}
        for rp in self._resolve_all():
            resolved_by_req.setdefault(rp.req_id, []).append(rp)

        for req_id in self._covered_req_ids():
            cat = self._lookup_catalogue(req_id)
            if cat is None:
                continue
            st = cat["sitl_test"]
            verify = self._bind_verify_args(
                st.get("verify"), cat.get("_resolved_attr_val"))
            if verify is not None:
                specs.append(SITLTestSpec(
                    req_id=req_id,
                    tier=st["tier"],
                    inject=st.get("inject"),
                    verify=verify,
                    notes=st.get("notes", ""),
                    params=tuple(resolved_by_req.get(req_id, [])),
                ))
            l2 = cat.get("l2_test")
            if l2 is not None:
                # 同一需求的附加 L2 行为检查（如 FENCE 缩尺执法）；参数
                # 已由主 spec 携带并写入启动 .parm，这里不重复。
                l2_verify = self._bind_verify_args(
                    l2.get("verify"), cat.get("_resolved_attr_val"))
                if l2_verify is not None:
                    specs.append(SITLTestSpec(
                        req_id=req_id,
                        tier="L2",
                        inject=l2.get("inject"),
                        verify=l2_verify,
                        notes=l2.get("notes", ""),
                        params=(),
                    ))
        return specs

    # ------------------------------------------------------------------
    # SITL → LLM feedback
    # ------------------------------------------------------------------

    def unresolved_feedback(self) -> List[Dict[str, Any]]:
        """Turn every unresolved SITL parameter into an actionable model-fix
        instruction for the design LLM.

        After the AST-synthesis threshold fix, an unresolved parameter is a
        trustworthy "model defect" signal: the requirement matched a catalogue
        tag, but the model genuinely lacks the guard/attribute the tag needs to
        supply a value.  Each item names the satisfying part, what element is
        missing, and which ArduPilot parameter depends on it.

        Returns a list of dicts: {req_id, tag, kind, part, param, message}.
        """
        items: List[Dict[str, Any]] = []
        for spec in self.generate_test_specs():
            cat = self._lookup_catalogue(spec.req_id)
            tag = (cat or {}).get("semantic_tag", "")
            entry = self._tag_to_entry.get(tag)
            parts = self._satisfy_map.get(spec.req_id, [])
            part = parts[0] if parts else "<the satisfying part>"
            for p in spec.params:
                if not (isinstance(p.value, str) and p.value.startswith("<unresolved")):
                    continue
                kind = p.value[len("<unresolved:"):].rstrip(">")
                items.append({
                    "req_id":  spec.req_id,
                    "tag":     tag,
                    "kind":    kind,
                    "part":    part,
                    "param":   p.param_name,
                    "message": self._unresolved_message(
                        spec.req_id, tag, entry, kind, part, p.param_name
                    ),
                })
        return items

    @staticmethod
    def _unresolved_message(req_id, tag, entry, kind, part, param_name) -> str:
        """State what the model is missing and why it matters — WITHOUT
        prescribing SysML syntax.

        The generation prompt already teaches canonical SysML v2 transition/
        attribute syntax; re-teaching it here is redundant and risky (a
        hand-written fragment that drifts from the canonical form would
        actively mislead the LLM, like the earlier `readonly` mistake).  So the
        feedback gives only semantic facts — which part, what quantity must be
        monitored/declared, which ArduPilot parameter depends on it — and lets
        the LLM apply its own (prompt-grounded, syntax-gate-validated) code.
        """
        gm = getattr(entry, "guard_matcher", None) if entry else None
        am = getattr(entry, "attr_matcher", None) if entry else None

        if kind.startswith("guard") and gm is not None:
            kws = "/".join(gm.var_keywords[:3]) or "the monitored quantity"
            return (
                f"{req_id} ({tag}): the part `{part}` that satisfies this "
                f"requirement defines no state-machine guard that compares a "
                f"{kws} variable against a numeric threshold. ArduPilot parameter "
                f"{param_name} is derived from that threshold, so it cannot be "
                f"set. Add the missing threshold-based guard."
            )

        if kind.startswith("attr") or kind == "chute_delay":
            if am is not None and am.attr_keywords:
                kws = "/".join(am.attr_keywords[:3])
            elif kind.startswith("attr:"):
                kws = kind.split(":", 1)[1]
            elif kind == "chute_delay":
                kws = "parachute deploy-time"
            else:
                kws = "the required"
            return (
                f"{req_id} ({tag}): the part `{part}` that satisfies this "
                f"requirement declares no numeric attribute representing the "
                f"{kws} value. ArduPilot parameter {param_name} is derived from "
                f"it, so it cannot be set. Add the missing attribute."
            )

        return (
            f"{req_id} ({tag}): parameter {param_name} is unresolved — the model "
            f"is missing the guard/attribute it maps from on part `{part}`."
        )

    def coverage_stats(self) -> Dict[str, Any]:
        """Machine-readable coverage classification (the numbers behind coverage_report).

        "unmapped" is the honest-gap bucket: requirements the model satisfies but
        that no SITL check verifies at any tier. Reports must surface this count so
        "traceability blocked = 0" is not over-read as "everything verified".
        """
        covered = self._covered_req_ids()
        mapped: set = set()
        mismatched: set = set()
        for r in covered:
            entry = self._lookup_catalogue(r)
            if entry is None:
                continue
            tag = str(entry.get("semantic_tag", ""))
            (mismatched if tag.startswith("TRACEABILITY_MISMATCH:") else mapped).add(r)
        unmapped = covered - mapped - mismatched
        return {
            "satisfied": len(covered),
            "mapped": len(mapped),
            "traceability_mismatched": len(mismatched),
            "unmapped": len(unmapped),
            "unmapped_req_ids": sorted(unmapped),
        }

    def coverage_report(self) -> str:
        covered = self._covered_req_ids()
        matched_content, matched_ast, matched_llm, matched_direct = (
            set(), set(), set(), set()
        )
        mismatched = set()

        for r in covered:
            entry = self._lookup_catalogue(r)
            if entry is None:
                continue
            tag = entry.get("semantic_tag", "")
            if tag.startswith("TRACEABILITY_MISMATCH:"):
                mismatched.add(r)
            elif tag.startswith("LLM_DIRECT:"):
                matched_direct.add(r)
            elif r in self._semantic_map:
                matched_llm.add(r)
            elif any(self._ast_specs.get(p) for p in self._satisfy_map.get(r, [])):
                matched_ast.add(r)
            else:
                matched_content.add(r)

        matched   = matched_content | matched_ast | matched_llm | matched_direct
        unmatched = covered - matched - mismatched

        lines = [
            f"Coverage report — {self._model.name}",
            f"  Satisfied reqs in model  : {len(covered)}",
            f"  Mapped via content match : {len(matched_content)}",
            f"  Mapped via AST synth     : {len(matched_ast)}",
            f"  Mapped via LLM tag       : {len(matched_llm)}",
            f"  Mapped via LLM req-text  : {len(matched_direct)}",
            f"  Mapped total             : {len(matched)}",
            f"  Traceability mismatches  : {len(mismatched)}",
            "",
            "Mapped requirements:",
        ]
        for r in sorted(matched_content):
            entry = self._lookup_catalogue(r)
            tier = entry["sitl_test"]["tier"] if entry else "?"
            tag  = entry.get("semantic_tag", "?") if entry else "?"
            lines.append(f"  ✓ {r:<22} [{tier}] (content:{tag})")
        for r in sorted(matched_ast):
            entry = self._lookup_catalogue(r)
            tier = entry["sitl_test"]["tier"] if entry else "?"
            lines.append(f"  ✓ {r:<22} [{tier}] (ast-synth)")
        for r in sorted(matched_llm):
            entry = self._lookup_catalogue(r)
            tier = entry["sitl_test"]["tier"] if entry else "?"
            tag  = self._semantic_map.get(r, "?")
            lines.append(f"  ✓ {r:<22} [{tier}] (llm:{tag})")
        for r in sorted(matched_direct):
            entry = self._lookup_catalogue(r)
            tier = entry["sitl_test"]["tier"] if entry else "?"
            param = entry.get("semantic_tag", "?").replace("LLM_DIRECT:", "") if entry else "?"
            lines.append(f"  ✓ {r:<22} [{tier}] (llm-direct:{param})")
        if mismatched:
            lines.append("\nTraceability mismatches (not verified):")
            details = {d["req_id"]: d for d in self.traceability_mismatches()}
            for r in sorted(mismatched):
                d = details.get(r, {})
                lines.append(
                    f"  ✗ {r:<22} expected={d.get('expected_family', '?')} "
                    f"matched={d.get('matched_tag', '?')}"
                )
        if unmatched:
            lines.append("\nSatisfied but no mapping (skipped):")
            for r in sorted(unmatched):
                lines.append(f"  – {r}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Build maps from AST
    # ------------------------------------------------------------------

    def _build_satisfy_map(self) -> Dict[str, List[str]]:
        """
        req_id → [part_name, ...]

        SysML v2 允许多个 part 同时 satisfy 同一 REQ（联合满足、分层满足、冗余满足）。
        保留所有满足关系；后续解析按"谁有 guard 就用谁"选取，不靠迭代顺序。
        """
        m: Dict[str, List[str]] = {}
        for part in self._model.part_definitions:
            for sat in part.satisfy_relationships:
                m.setdefault(sat.target.name, []).append(part.name)
        return m

    def _build_guard_map(self) -> Dict[str, List[Any]]:
        """
        part_name → 该 part 的状态机 guard 列表。
        使用 state_extractor 提取，与 behavioral_sim 共享同一 AST 路径。
        """
        from src.simulation.state_extractor import extract_state_machines
        guard_map: Dict[str, List[Any]] = {}
        try:
            # extract_state_machines 接收 SysML 原始文本，不是 LiteModel 对象
            sysml_text = self._model.to_sysml_text()
            if not sysml_text:
                return guard_map
            state_machines = extract_state_machines(sysml_text)
            for sm in state_machines:
                for trans in sm.fault_transitions():
                    guard_map.setdefault(sm.owner_part, []).extend(trans.guards)
                # accept 事件驱动的转移没有 guard，fault_transitions() 收不到；
                # 以伪 guard 形式并入，让事件驱动写法与 guard 写法共享同一条
                # 独占分配/匹配管线（只有显式声明 "event" operator 的条目会
                # 匹配它们，存量 bool/comparison 条目行为不变）。
                for trans in sm.transitions:
                    if trans.is_initial or trans.guards or not trans.accept_trigger:
                        continue
                    guard_map.setdefault(sm.owner_part, []).append(
                        AcceptEventGuard(attribute=str(trans.accept_trigger))
                    )
        except Exception as exc:
            from src.utils.suppressed import record_suppressed
            record_suppressed("sitl.requirement_linker.guard_map", exc)
            pass
        return guard_map

    def _build_syside_attr_map(self) -> Dict[str, Dict[str, float]]:
        """
        Walk the syside AST and evaluate every AttributeUsage expression,
        grouped by owner part name.  Returns {part_name: {attr_name: float}}.

        Handles arithmetic expressions and unit-bearing literals that
        _parse_attr_value (regex-based) cannot evaluate.
        Falls back to {} when syside is unavailable or parsing fails.
        """
        if not _SYSIDE_OK:
            return {}
        sysml_text = self._model.to_sysml_text() if self._model else ""
        if not sysml_text:
            return {}
        result: Dict[str, Dict[str, float]] = {}
        try:
            model, _ = _syside.try_load_model(sysml_source=sysml_text)
            compiler = _syside.Compiler()
            for attr in model.nodes(_syside.AttributeUsage):
                try:
                    owner = attr.owner
                    if owner is None:
                        continue
                    part_name = getattr(owner, "name", None)
                    if not part_name:
                        continue
                    # An AttributeUsage need not be named — a redefinition
                    # such as `attribute :>> mass = 5[kg];` is legal and
                    # reports name None.  Keying the map on that put a None
                    # into every keyword scan that later reads attr names.
                    attr_name = getattr(attr, "name", None)
                    if not attr_name:
                        continue
                    expr = attr.feature_value_expression
                    if expr is None:
                        continue
                    val, report = compiler.evaluate(expr)
                    if not report.fatal:
                        from src.utils.syside_utils import coerce_static_number
                        number = coerce_static_number(val)
                        if number is not None:
                            result.setdefault(part_name, {})[attr_name] = number
                except Exception as exc:
                    from src.utils.suppressed import record_suppressed
                    record_suppressed("sitl.requirement_linker.syside_attr_node", exc)
                    pass
        except Exception as exc:
            from src.utils.suppressed import record_suppressed
            record_suppressed("sitl.requirement_linker.syside_attr_map", exc)
            pass
        return result

    def _build_port_map(self) -> Dict[str, List[str]]:
        """part_name → [port names]。

        接口/配置类需求的 satisfy 目标（CommunicationSystem、PerceptionSystem）
        在 SysML 里合法地只声明 port（``in port gnssCorrections``）而无属性；
        AttrMatcher 声明 ``allow_port_match`` 时以 port 名为匹配面。port 无
        数值，因此只允许全静态 ardu_params 的条目走这条路（_match_by_content
        强制检查）。
        """
        result: Dict[str, List[str]] = {}
        for part in self._model.part_definitions:
            names = [
                str(port.name) for port in getattr(part, "ports", [])
                if getattr(port, "name", None)
            ]
            if names:
                result[part.name] = names
        return result

    def _build_attr_map(self) -> Dict[str, Dict[str, float]]:
        """part_name → {attr_name: float}，用于 @attr: 直接读属性的情况。"""
        result: Dict[str, Dict[str, float]] = {}
        for part in self._model.part_definitions:
            attrs: Dict[str, float] = {}
            for attr in part.attributes:
                val = self._parse_attr_value(attr)
                if val is not None:
                    attrs[attr.name] = val
            result[part.name] = attrs

        # Augment with syside-evaluated values: handles expressions like
        # `= 10.0 [m/s]` or `= mass * g` that _parse_attr_value regex misses.
        # Syside values take precedence when they successfully evaluate.
        for part_name, syside_attrs in self._build_syside_attr_map().items():
            result.setdefault(part_name, {}).update(syside_attrs)

        return result

    # ------------------------------------------------------------------
    # 解析
    # ------------------------------------------------------------------

    def _covered_req_ids(self) -> set:
        return set(self._satisfy_map.keys())

    def _resolve_all(self) -> List[ResolvedParam]:
        """
        遍历所有已映射的 REQ，将 ardu_params 的 token 解析为实际数值。

        token 解析优先级（内容匹配后已预解析的值优先）：
          @guard       → cat["_resolved_guard_val"]（内容匹配已算好）
          @attr_match  → cat["_resolved_attr_val"]（内容匹配已算好）
          @attr:name   → 关键词搜索属性值
          @attr:name*N → 同上 × N
          @guard*N / @guard_attr*N → parachute delay 特殊处理
          number       → 直接使用
        """
        if self._resolved_cache is not None:
            return list(self._resolved_cache)
        results: List[ResolvedParam] = []
        for req_id in self._covered_req_ids():
            cat = self._lookup_catalogue(req_id)
            if cat is None:
                continue

            part_names = self._satisfy_map[req_id]
            part_name  = part_names[0]

            # 从 cat 取出预解析值（内容匹配路径已解析，其他路径为 None）
            pre_guard_val = cat.get("_resolved_guard_val")
            pre_guard_src = cat.get("_resolved_guard_src") or "guard"
            pre_attr_val  = cat.get("_resolved_attr_val")
            pre_attr_src  = cat.get("_resolved_attr_src") or "attr"

            # parachute delay：按关键词搜索属性（兼容新旧路径）
            chute_delay: Optional[float] = None
            if "CHUTE_DELAY_MS" in cat.get("ardu_params", {}):
                for pname in part_names:
                    chute_delay = self._extract_chute_delay(pname)
                    if chute_delay is not None:
                        part_name = pname
                        break

            for param_name, raw_value in cat["ardu_params"].items():
                if raw_value == "@guard":
                    val = pre_guard_val if pre_guard_val is not None \
                          else "<unresolved:guard>"
                    src = pre_guard_src

                elif raw_value == "@attr_match":
                    # AttrMatcher 已预解析（含单位换算）
                    val = pre_attr_val if pre_attr_val is not None \
                          else "<unresolved:attr_match>"
                    src = pre_attr_src

                elif isinstance(raw_value, str) and (
                    raw_value.startswith("@guard*") or
                    raw_value.startswith("@guard_attr*")
                ):
                    # parachute: parachuteDeployTime × 1000
                    val = (round(chute_delay * 1000)
                           if chute_delay is not None
                           else "<unresolved:chute_delay>")
                    src = "attr:parachute*1000" if chute_delay else "unresolved"

                elif isinstance(raw_value, str) and raw_value.startswith("@attr:"):
                    # @attr:name 或 @attr:name*N
                    rest = raw_value[6:]
                    if "*" in rest:
                        attr_name, factor_str = rest.split("*", 1)
                        factor = float(factor_str)
                    else:
                        attr_name, factor = rest, 1.0
                    val, src = self._extract_attr(part_name, attr_name)
                    if isinstance(val, (int, float)) and factor != 1.0:
                        val = round(val * factor)
                        src = f"{src}*{int(factor)}"

                else:
                    val = raw_value
                    src = "static"

                results.append(ResolvedParam(
                    req_id=req_id,
                    part_name=part_name,
                    param_name=param_name,
                    value=val,
                    source=src,
                ))
        self._resolved_cache = tuple(results)
        return list(self._resolved_cache)

    def _threshold_for_guard_var(
        self,
        part_name: str,
        guard_var: str,
        operators: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], str]:
        """Resolve the numeric threshold of the model guard whose attribute
        matches *guard_var* (the variable the AST synthesizer matched).

        Used by the AST-synthesis layer so `@guard` placeholders are filled
        from the guard that layer actually found — instead of leaking as
        `<unresolved:guard>` even though the model contains the guard.

        When *operators* is given (the tag's expected operators), a guard
        whose operator is in that set is preferred.  Returns (None, "") when
        no matching guard carries a numeric threshold (e.g. a boolean guard,
        which needs no threshold).
        """
        gv = (guard_var or "").lower()
        cands = [
            g for g in self._guard_map.get(part_name, [])
            if (getattr(g, "attribute", "") or "").lower() == gv
            and getattr(g, "threshold", None) is not None
        ]
        if not cands:
            return None, ""
        if operators:
            preferred = [g for g in cands if getattr(g, "operator", None) in operators]
            if preferred:
                cands = preferred
        g = cands[0]
        op = getattr(g, "operator", "?")
        th = float(getattr(g, "threshold"))
        return th, f"guard:{op}:{th} (attr:{getattr(g, 'attribute', '?')}, ast)"

    def _extract_chute_delay(self, part_name: str) -> Optional[float]:
        """
        在 part_name 的属性里，找 parachute/deploy 相关的时间属性。
        使用语义关键词匹配，不依赖精确属性名。
        """
        attrs = self._attr_map.get(part_name, {})
        keywords = ["parachute", "deploy", "chute"]
        for name, val in attrs.items():
            nl = name.lower()
            if any(k in nl for k in keywords) and 0.1 <= val <= 5.0:
                return val
        # fallback: 找值在 [0.1, 5.0] 范围内的时间属性（秒）
        time_keywords = ["time", "delay", "timeout"]
        for name, val in attrs.items():
            nl = name.lower()
            if any(k in nl for k in time_keywords) and 0.1 <= val <= 2.0:
                return val
        return None

    def _extract_attr(
        self,
        part_name: str,
        attr_name: str,
    ) -> Tuple[Any, str]:
        """
        在 part_name 的所有属性里，优先精确名字匹配，
        否则做宽松关键词匹配。
        """
        # 先找与 satisfy 该需求的 part，再向其他 part 扩展
        all_parts = [p for p in self._model.part_definitions if p.name == part_name]
        all_parts += [p for p in self._model.part_definitions if p.name != part_name]

        for part in all_parts:
            attrs = self._attr_map.get(part.name, {})
            # 精确匹配
            if attr_name in attrs:
                return attrs[attr_name], f"attr:{part.name}.{attr_name}"
            # 宽松：attr_name 的关键词子集
            keywords = re.sub(r'([A-Z])', r' \1', attr_name).lower().split()
            for aname, aval in attrs.items():
                aname_lower = re.sub(r'([A-Z])', r' \1', aname).lower()
                if all(k in aname_lower for k in keywords):
                    return aval, f"attr:{part.name}.{aname}(~{attr_name})"

        return f"<unresolved:attr:{attr_name}>", "unresolved"

    def _parse_attr_value(self, attr) -> Optional[float]:
        raw = getattr(attr, "default_value", None)
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            if isinstance(raw, str):
                m = re.search(r"-?\d+\.?\d*", raw)
                if m:
                    return float(m.group())
        return None

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, float) and value == int(value):
            return str(int(value))
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)
