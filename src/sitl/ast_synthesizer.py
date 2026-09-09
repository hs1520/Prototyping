"""ast_synthesizer.py

从 SysML 状态机 AST 合成 InjectSpec + VerifySpec，不用 LLM。

匹配依据（比状态机名稳定）：
  - guard 变量名（Phase 2 已标准化）-> 决定 inject
  - entry action 名                  -> 决定 verify

guard 变量名能关键词匹配即生效：标准变量名覆盖率接近 100%，
非标准变量名回退到 LLM 分类或 unmapped。

公共 API：
  SynthesizedSpec   - AST 合成结果
  synthesize_specs(model) -> Dict[part_name, List[SynthesizedSpec]]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.sysml.lite_model import SysMLLiteModel

from src.sitl.sitl_specs import InjectSpec, VerifySpec


_GUARD_TO_TAG_INJECT: List[Tuple[List[str], str, Optional[InjectSpec]]] = [
    (["comm", "gcs", "link", "heartbeat"],
     "GCS_LOSS",
     InjectSpec(kind="disconnect_gcs", pre_takeoff_m=5.0)),

    (["propulsion", "engine", "motor", "thrust"],
     "PARACHUTE_DEPLOY",
     InjectSpec(kind="set_param",
                params={"SIM_ENGINE_FAIL": 1.0},
                pre_takeoff_m=10.0)),

    (["sensor", "selftest", "post", "prearm"],
     "SENSOR_ARMING_INHIBIT",
     InjectSpec(kind="set_param",
                params={"BATT_ARM_VOLT": 100.0, "_settle_s": 1.0})),

    # 电池 SOC（区分 RTB vs LAND 由 guard operator 决定，这里都先标 BATTERY_RTB）
    (["battery", "soc", "charge", "volt"],
     "BATTERY_RTB",
     None),

    (["payload", "abort", "delivery", "gripper"],
     "PAYLOAD_ABORT_LOCK",
     InjectSpec(
         kind="mavlink_command",
         params={"command": 211, "param1": 0, "param2": 1, "_settle_s": 1.5},
         pre_takeoff_m=5.0,
     )),
]


_ACTION_TO_VERIFY: List[Tuple[List[str], VerifySpec]] = [
    (["safelanding", "initiatesafelanding", "land", "descent", "controlleddescent"],
     VerifySpec(kind="wait_mode",
                args={"mode": "LAND", "fallback": "RTL"},
                timeout=25.0)),

    (["rtb", "returntobase", "return", "home", "initiaterth", "rtl"],
     VerifySpec(kind="wait_mode",
                args={"mode": "RTL"},
                timeout=25.0)),

    (["parachute", "chute", "recovery", "deployparachute", "deploy"],
     VerifySpec(kind="wait_statustext",
                # 匹配成功消息 "Parachute: Released"；用 "arachute" 会误匹配
                # 拒绝消息 "Parachute: Landed"（地面/已着陆拒绝部署）
                args={"keyword": "Parachute: Released"},
                timeout=15.0)),

    (["inhibit", "inhibitarming", "blockarm", "preventarm", "disarm"],
     VerifySpec(kind="assert_arm_rejected", timeout=6.0)),

    (["lockpayload", "payload", "lock", "gripper", "abort"],
     VerifySpec(
         kind="wait_statustext",
         args={"keyword": ["ripper", "grip"]},
         timeout=10.0,
     )),
]


@dataclass
class SynthesizedSpec:
    """AST 合成的一个测试 spec，绑定到某个 part 的某个状态机。"""
    owner_part: str
    sm_name: str
    guard_var: str
    entry_action: str
    tag: str
    inject: InjectSpec
    verify: VerifySpec


def synthesize_specs(
    model: "SysMLLiteModel",
    verbose: bool = False,
    identity_tags: Optional[Mapping[str, str]] = None,
) -> Dict[str, List[SynthesizedSpec]]:
    """从模型 AST 合成 (InjectSpec, VerifySpec) 对。

    identity_tags: {模型标识符(小写) -> semantic_tag}，来自冻结计划的绑定
    （linker 构建）。身份命中优先于关键词族，模型自选拼写的 guard 变量/事件
    也能合成到正确的 tag，关键词只在无计划时兜底。返回
    {part_name: [SynthesizedSpec, ...]}，一个 part 的每个状态机一条；未匹配
    的状态机跳过。
    """
    try:
        from src.simulation.state_extractor import extract_state_machines
        text = model.to_sysml_text() or ""
        state_machines = extract_state_machines(text)
    except Exception as exc:
        from src.utils.suppressed import record_suppressed
        record_suppressed("sitl.ast_synthesizer.state_extract", exc)
        return {}

    result: Dict[str, List[SynthesizedSpec]] = {}

    for sm in state_machines:
        for transition in sm.fault_transitions():
            for guard in transition.guards:
                guard_var = getattr(guard, "attribute", None) or ""
                inject_spec, tag = _match_guard(guard_var, identity_tags)
                if inject_spec is None and tag == "":
                    continue

                target_state = _find_target_state(sm, transition)
                entry_action = target_state or ""
                verify_spec = _match_action(entry_action)
                if verify_spec is None:
                    continue

                spec = SynthesizedSpec(
                    owner_part=sm.owner_part,
                    sm_name=sm.name,
                    guard_var=guard_var,
                    entry_action=entry_action,
                    tag=tag,
                    inject=inject_spec,
                    verify=verify_spec,
                )
                result.setdefault(sm.owner_part, []).append(spec)

                if verbose:
                    print(f"  [AST] {sm.owner_part}.{sm.name}: "
                          f"guard={guard_var!r} → {tag} "
                          f"inject={inject_spec.kind} verify={verify_spec.kind}")

                break

    return result


_TAG_TO_INJECT: Dict[str, Optional[InjectSpec]] = {
    tag: inject for _kws, tag, inject in _GUARD_TO_TAG_INJECT
}

AST_GUARD_TAGS: Tuple[str, ...] = tuple(_TAG_TO_INJECT)


def _match_guard(
    guard_var: str,
    identity_tags: Optional[Mapping[str, str]] = None,
) -> Tuple[Optional[InjectSpec], str]:
    var_lower = guard_var.lower()
    if identity_tags:
        tag = identity_tags.get(var_lower)
        if tag and tag in _TAG_TO_INJECT:
            return (_TAG_TO_INJECT[tag], tag)
    for keywords, tag, inject_spec in _GUARD_TO_TAG_INJECT:
        if any(kw in var_lower for kw in keywords):
            return (inject_spec, tag)
    return (None, "")


def _match_action(entry_action: str) -> Optional[VerifySpec]:
    action_lower = entry_action.lower()
    for keywords, verify_spec in _ACTION_TO_VERIFY:
        if any(kw in action_lower for kw in keywords):
            return verify_spec
    return None


def _find_target_state(sm, transition) -> Optional[str]:
    target_name = getattr(transition, "target", None)
    if not target_name:
        return None
    entry = sm.entry_action_for_state(target_name)
    return entry
