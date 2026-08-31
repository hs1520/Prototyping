"""
ast_synthesizer.py

从 SysML 状态机 AST 直接合成 InjectSpec + VerifySpec，无需 LLM。

匹配依据（比状态机名更可靠）：
  - guard 变量名（Phase 2 已标准化）→ 决定 inject
  - entry action 名                  → 决定 verify

覆盖率：只要 guard 变量名能关键词匹配就生效。
  标准变量名（Phase 2 约束后）→ 覆盖率接近 100%。
  非标准变量名 → 回退到 LLM 分类或 unmapped。

公共 API：
  SynthesizedSpec   — AST 合成结果
  synthesize_specs(model) → Dict[part_name, List[SynthesizedSpec]]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.sysml.lite_model import SysMLLiteModel

from src.sitl.sitl_specs import InjectSpec, VerifySpec


# ---------------------------------------------------------------------------
# 关键词表：guard 变量名 → (semantic_tag, InjectSpec)
#
# 每个条目是 (关键词集, tag, InjectSpec | None)
# 匹配规则：guard 变量名小写后，任一关键词是其子串
# InjectSpec=None 表示纯 L1（参数验证），不需要注入
# ---------------------------------------------------------------------------

_GUARD_TO_TAG_INJECT: List[Tuple[List[str], str, Optional[InjectSpec]]] = [
    # GCS / 通信丢失
    (["comm", "gcs", "link", "heartbeat"],
     "GCS_LOSS",
     InjectSpec(kind="disconnect_gcs", pre_takeoff_m=5.0)),

    # 引擎 / 推进失效
    (["propulsion", "engine", "motor", "thrust"],
     "PARACHUTE_DEPLOY",
     InjectSpec(kind="set_param",
                params={"SIM_ENGINE_FAIL": 1.0},
                pre_takeoff_m=10.0)),

    # 传感器自检 / POST 失败
    (["sensor", "selftest", "post", "prearm"],
     "SENSOR_ARMING_INHIBIT",
     InjectSpec(kind="set_param",
                params={"BATT_ARM_VOLT": 100.0, "_settle_s": 1.0})),

    # 电池 SOC（区分 RTB vs LAND 由 guard operator 决定，这里都先标 BATTERY_RTB）
    (["battery", "soc", "charge", "volt"],
     "BATTERY_RTB",
     None),       # 纯 L1，不需要 SITL 注入

    # 载荷中止 — MAV_CMD_DO_GRIPPER (211) RELEASE
    (["payload", "abort", "delivery", "gripper"],
     "PAYLOAD_ABORT_LOCK",
     InjectSpec(
         kind="mavlink_command",
         params={"command": 211, "param1": 0, "param2": 1, "_settle_s": 1.5},
         pre_takeoff_m=5.0,
     )),
]


# ---------------------------------------------------------------------------
# 关键词表：entry action 名 → VerifySpec
# ---------------------------------------------------------------------------

_ACTION_TO_VERIFY: List[Tuple[List[str], VerifySpec]] = [
    # 安全降落 / 受控下降
    (["safelanding", "initiatesafelanding", "land", "descent", "controlleddescent"],
     VerifySpec(kind="wait_mode",
                args={"mode": "LAND", "fallback": "RTL"},
                timeout=25.0)),

    # 返航 RTB/RTL
    (["rtb", "returntobase", "return", "home", "initiaterth", "rtl"],
     VerifySpec(kind="wait_mode",
                args={"mode": "RTL"},
                timeout=25.0)),

    # 降落伞
    (["parachute", "chute", "recovery", "deployparachute", "deploy"],
     VerifySpec(kind="wait_statustext",
                # 必须匹配成功消息 "Parachute: Released"，不能用 "arachute"——
                # 后者会误匹配拒绝消息 "Parachute: Landed"（地面/已着陆拒绝部署）
                args={"keyword": "Parachute: Released"},
                timeout=15.0)),

    # 阻止解锁
    (["inhibit", "inhibitarming", "blockarm", "preventarm", "disarm"],
     VerifySpec(kind="assert_arm_rejected", timeout=6.0)),

    # 载荷锁定 — STATUSTEXT "Gripper Released"
    (["lockpayload", "payload", "lock", "gripper", "abort"],
     VerifySpec(
         kind="wait_statustext",
         args={"keyword": ["ripper", "grip"]},
         timeout=10.0,
     )),
]


# ---------------------------------------------------------------------------
# 结果类
# ---------------------------------------------------------------------------

@dataclass
class SynthesizedSpec:
    """AST 合成的一个测试 spec，绑定到某个 part 的某个状态机。"""
    owner_part: str
    sm_name: str
    guard_var: str          # 触发 inject 匹配的 guard 变量名
    entry_action: str       # 触发 verify 匹配的 entry action 名
    tag: str                # 对应的 semantic_tag
    inject: InjectSpec
    verify: VerifySpec


# ---------------------------------------------------------------------------
# 主合成函数
# ---------------------------------------------------------------------------

def synthesize_specs(
    model: "SysMLLiteModel",
    verbose: bool = False,
    identity_tags: Optional[Mapping[str, str]] = None,
) -> Dict[str, List[SynthesizedSpec]]:
    """
    从模型 AST 合成 (InjectSpec, VerifySpec) 对。

    identity_tags: {模型标识符(小写) → semantic_tag}，来自冻结计划的绑定
    （linker 构建）。身份命中优先于英文关键词族——模型自选拼写的 guard
    变量/事件仍能合成到正确的 tag；关键词仅作无计划时的兜底。

    返回 {part_name: [SynthesizedSpec, ...]}。
    一个 part 可能有多个状态机，每个产生一个 SynthesizedSpec。
    未能匹配的状态机直接跳过（fail-safe，不产生误判）。
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
                    continue  # 未匹配，跳过

                # 找 target state 的 entry action
                target_state = _find_target_state(sm, transition)
                entry_action = target_state or ""
                verify_spec = _match_action(entry_action)
                if verify_spec is None:
                    continue  # 无法合成 verify，跳过

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

                break  # 每个 transition 只用第一个 guard 匹配

    return result


# ---------------------------------------------------------------------------
# 内部匹配函数
# ---------------------------------------------------------------------------

_TAG_TO_INJECT: Dict[str, Optional[InjectSpec]] = {
    tag: inject for _kws, tag, inject in _GUARD_TO_TAG_INJECT
}

#: The semantic tags this synthesizer can produce — the linker's identity
#: tier maps plan-declared identifiers onto exactly these.
AST_GUARD_TAGS: Tuple[str, ...] = tuple(_TAG_TO_INJECT)


def _match_guard(
    guard_var: str,
    identity_tags: Optional[Mapping[str, str]] = None,
) -> Tuple[Optional[InjectSpec], str]:
    """
    根据 guard 变量名匹配注入 spec 和语义 tag。
    计划身份优先(变量是计划为某需求声明的元素 → 直接取该需求的 tag),
    关键词族兜底。返回 (InjectSpec | None, tag)。tag="" 表示未匹配。
    """
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
    """
    根据 entry action 名（小写）匹配验证 spec。
    """
    action_lower = entry_action.lower()
    for keywords, verify_spec in _ACTION_TO_VERIFY:
        if any(kw in action_lower for kw in keywords):
            return verify_spec
    return None


def _find_target_state(sm, transition) -> Optional[str]:
    """
    从 transition 找 target state 的 entry_action 名。
    """
    target_name = getattr(transition, "target", None)
    if not target_name:
        return None
    entry = sm.entry_action_for_state(target_name)
    return entry
