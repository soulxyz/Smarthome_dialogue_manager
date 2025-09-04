"""设备状态机与默认预设系统

提供可编程的设备抽象、二室一厅默认设备预设、
以及统一的设备管理与执行接口，便于与对话引擎集成。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List, Callable
from datetime import datetime
from copy import deepcopy


# 基础设备抽象
@dataclass
class Device:
    device_type: str
    name: str
    room: str
    state: Dict[str, Any] = field(default_factory=dict)

    def turn_on(self) -> Tuple[bool, str]:
        self.state["on"] = True
        return True, f"{self.room}{self.name}已开启"

    def turn_off(self) -> Tuple[bool, str]:
        self.state["on"] = False
        return True, f"{self.room}{self.name}已关闭"

    def adjust(self, attribute: str, value: Any) -> Tuple[bool, str]:
        # 缺省实现：直接写入
        self.state[attribute] = value
        return True, f"已将{self.room}{self.name}的{attribute}设置为{value}"

    def status_text(self) -> str:
        on_text = "已开启" if self.state.get("on") else "已关闭"
        return f"{self.room}{self.name}{on_text}"


# 具体设备
class LightDevice(Device):
    def __init__(self, name: str, room: str):
        super().__init__(device_type="灯", name=name, room=room, state={"on": False, "brightness": 50})

    def _parse_int_like(self, value: Any, err_msg: str) -> Tuple[bool, Optional[int]]:
        if value is None:
            return False, None
        try:
            if isinstance(value, (int, float)):
                return True, int(value)
            if isinstance(value, str):
                v = int(float(value.strip()))
                return True, v
        except Exception:
            return False, None
        return False, None

    def adjust(self, attribute: str, value: Any) -> Tuple[bool, str]:
        if attribute in ("亮度", "brightness"):
            ok, v = self._parse_int_like(value, "亮度取值无效")
            if not ok:
                return False, "亮度取值无效"
            v = max(0, min(100, int(v)))
            self.state["brightness"] = v
            return True, f"已将{self.room}{self.name}亮度设置为{v}%"
        return super().adjust(attribute, value)

    def status_text(self) -> str:
        base = super().status_text()
        return f"{base}，亮度{self.state.get('brightness', 0)}%"


class AirConditionerDevice(Device):
    def __init__(self, name: str, room: str):
        super().__init__(device_type="空调", name=name, room=room, state={"on": False, "temperature": 26, "mode": "自动", "fan_speed": 2})

    def _parse_int_like(self, value: Any) -> Tuple[bool, Optional[int]]:
        if value is None:
            return False, None
        try:
            if isinstance(value, (int, float)):
                return True, int(value)
            if isinstance(value, str):
                v = int(float(value.strip()))
                return True, v
        except Exception:
            return False, None
        return False, None

    def adjust(self, attribute: str, value: Any) -> Tuple[bool, str]:
        if attribute in ("温度", "temperature"):
            ok, v = self._parse_int_like(value)
            if not ok:
                return False, "温度取值无效"
            v = max(16, min(30, int(v)))
            self.state["temperature"] = v
            return True, f"已将{self.room}{self.name}温度设置为{v}度"
        if attribute in ("风速", "fan_speed"):
            ok, v = self._parse_int_like(value)
            if not ok:
                return False, "风速取值无效"
            v = max(1, min(5, int(v)))
            self.state["fan_speed"] = v
            return True, f"已将{self.room}{self.name}风速设置为{v}档"
        if attribute in ("模式", "mode"):
            self.state["mode"] = str(value)
            return True, f"已将{self.room}{self.name}模式设置为{value}"
        return super().adjust(attribute, value)

    def status_text(self) -> str:
        base = super().status_text()
        return (
            f"{base}，温度{self.state.get('temperature', 26)}度，"
            f"风速{self.state.get('fan_speed', 2)}档，模式{self.state.get('mode', '自动')}"
        )


class TVDevice(Device):
    def __init__(self, name: str, room: str):
        super().__init__(device_type="电视", name=name, room=room, state={"on": False, "volume": 20, "channel": 1})

    def _parse_int_like(self, value: Any) -> Tuple[bool, Optional[int]]:
        if value is None:
            return False, None
        try:
            if isinstance(value, (int, float)):
                return True, int(value)
            if isinstance(value, str):
                v = int(float(value.strip()))
                return True, v
        except Exception:
            return False, None
        return False, None

    def adjust(self, attribute: str, value: Any) -> Tuple[bool, str]:
        if attribute in ("音量", "volume"):
            ok, v = self._parse_int_like(value)
            if not ok:
                return False, "音量取值无效"
            v = max(0, min(100, int(v)))
            self.state["volume"] = v
            return True, f"已将{self.room}{self.name}音量设置为{v}"
        if attribute in ("频道", "channel"):
            ok, v = self._parse_int_like(value)
            if not ok:
                return False, "频道取值无效"
            v = max(1, int(v))
            self.state["channel"] = v
            return True, f"已将{self.room}{self.name}切换到第{v}频道"
        return super().adjust(attribute, value)

    def status_text(self) -> str:
        base = super().status_text()
        return f"{base}，音量{self.state.get('volume', 0)}"


class FanDevice(Device):
    def __init__(self, name: str, room: str):
        super().__init__(device_type="风扇", name=name, room=room, state={"on": False, "speed": 2})

    def _parse_int_like(self, value: Any) -> Tuple[bool, Optional[int]]:
        if value is None:
            return False, None
        try:
            if isinstance(value, (int, float)):
                return True, int(value)
            if isinstance(value, str):
                v = int(float(value.strip()))
                return True, v
        except Exception:
            return False, None
        return False, None

    def adjust(self, attribute: str, value: Any) -> Tuple[bool, str]:
        if attribute in ("风速", "speed", "fan_speed"):
            ok, v = self._parse_int_like(value)
            if not ok:
                return False, "风速取值无效"
            v = max(1, min(5, int(v)))
            self.state["speed"] = v
            return True, f"已将{self.room}{self.name}风速设置为{v}档"
        return super().adjust(attribute, value)

    def status_text(self) -> str:
        base = super().status_text()
        return f"{base}，风速{self.state.get('speed', 0)}档"


class DeviceManager:
    """设备管理器：维护设备状态并提供执行接口"""

    def __init__(self):
        # 设备注册表：按类型与房间索引
        self.devices: List[Device] = []
        self._index: Dict[Tuple[str, str], List[Device]] = {}
        self._build_default_preset()
        # 版本与事件回调
        self._version: int = 0
        self._last_updated_at: Optional[str] = None
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []
        # 最近一次快照（便于做默认对比）
        self._last_snapshot: Dict[str, Any] = self.snapshot()

    # ---------- 预设 ----------
    def _build_default_preset(self):
        # 客厅
        self._register(LightDevice(name="灯", room="客厅"))
        self._register(TVDevice(name="电视", room="客厅"))
        self._register(AirConditionerDevice(name="空调", room="客厅"))
        self._register(FanDevice(name="风扇", room="客厅"))
        # 主卧
        self._register(LightDevice(name="灯", room="主卧"))
        self._register(AirConditionerDevice(name="空调", room="主卧"))
        # 次卧
        self._register(LightDevice(name="灯", room="次卧"))
        self._register(FanDevice(name="风扇", room="次卧"))

    def _register(self, device: Device):
        self.devices.append(device)
        key = (device.device_type, device.room)
        self._index.setdefault(key, []).append(device)

    # ---------- 事件/版本 ----------
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """注册设备事件回调. 回调签名: fn(event: Dict[str, Any]) -> None"""
        if callback and callback not in self._callbacks:
            self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _bump_version(self) -> None:
        self._version += 1
        self._last_updated_at = datetime.now().isoformat(timespec="seconds")

    def _emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        event = {
            "event": event_type,
            "version": self._version,
            "timestamp": self._last_updated_at,
            **payload,
        }
        for cb in list(self._callbacks):
            try:
                cb(event)
            except Exception:
                # 单个回调出错不影响整体
                continue

    def get_version(self) -> int:
        return self._version

    def get_last_updated(self) -> Optional[str]:
        return self._last_updated_at

    # ---------- 查询与执行 ----------
    def find_device(self, device_type: str, room: Optional[str]) -> Optional[Device]:
        if room:
            # 精确匹配房间；若未找到则不回退
            devs = self._index.get((device_type, room))
            if devs:
                return devs[0]
            return None
        # 未指定房间，若只有一个同类设备则返回，否则优先客厅
        all_of_type = [d for d in self.devices if d.device_type == device_type]
        if len(all_of_type) == 1:
            return all_of_type[0]
        for cand in all_of_type:
            if cand.room == "客厅":
                return cand
        return all_of_type[0] if all_of_type else None

    def perform_action(
        self,
        action_keyword: str,
        device_type: Optional[str],
        room: Optional[str] = None,
        attribute: Optional[str] = None,
        number_value: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not device_type:
            return {"success": False, "message": "未指定设备类型"}
        device = self.find_device(device_type, room)
        if not device:
            return {"success": False, "message": f"未找到{room or ''}的{device_type}"}

        action = self._normalize_action(action_keyword)
        # 记录前快照（仅针对该设备）
        before_state = deepcopy(device.state)
        if action == "turn_on":
            ok, msg = device.turn_on()
        elif action == "turn_off":
            ok, msg = device.turn_off()
        elif action in ("set", "increase", "decrease"):
            # 对于调节类动作，需要明确属性
            if not attribute and device_type in ("空调", "风扇"):
                attribute = "风速" if device_type == "风扇" else "温度"
            if not attribute and device_type == "灯":
                attribute = "亮度"
            if not attribute and device_type == "电视":
                attribute = "音量"

            if number_value is None:
                # 兜底步进
                step = 5
                base = self._read_attribute(device, attribute)
                if isinstance(base, int):
                    number_value = base + (step if action == "increase" else -step)
                else:
                    number_value = step
            ok, msg = device.adjust(attribute, number_value)
        else:
            ok, msg = False, "暂不支持的操作"

        # 若状态变更成功：更新版本、触发事件并刷新最近快照
        if ok:
            self._bump_version()
            after_state = deepcopy(device.state)
            self._emit_event(
                "state_changed",
                {
                    "device": {
                        "device_type": device.device_type,
                        "room": device.room,
                        "name": device.name,
                    },
                    "action": action,
                    "attribute": attribute,
                    "before_state": before_state,
                    "after_state": after_state,
                    "message": msg,
                },
            )
            # 更新全局最近快照
            self._last_snapshot = self.snapshot()

        return {
            "success": ok,
            "message": msg,
            "device_type": device.device_type,
            "room": device.room,
            "state": device.state.copy(),
        }

    def query_status(self, device_type: Optional[str] = None, room: Optional[str] = None) -> Dict[str, Any]:
        targets: List[Device]
        if device_type:
            if room:
                dev = self.find_device(device_type, room)
                targets = [dev] if dev else []
            else:
                targets = [d for d in self.devices if d.device_type == device_type]
        else:
            # 当仅指定房间时，按房间过滤
            if room:
                targets = [d for d in self.devices if d.room == room]
            else:
                # 全量
                targets = self.devices

        if not targets:
            return {"success": False, "message": "未找到对应设备"}

        lines = [d.status_text() for d in targets]
        return {"success": True, "message": "；".join(lines), "states": [d.state.copy() for d in targets]}

    def snapshot(self) -> Dict[str, Any]:
        return {
            f"{d.room}-{d.device_type}": {"on": d.state.get("on"), **{k: v for k, v in d.state.items() if k != "on"}}
            for d in self.devices
        }

    def snapshot_with_meta(self) -> Dict[str, Any]:
        """返回带元信息的快照，不破坏 snapshot() 兼容性"""
        return {
            "version": self._version,
            "timestamp": self._last_updated_at,
            "data": self.snapshot(),
        }

    def snapshot_diff(self, old_snapshot: Optional[Dict[str, Any]] = None, new_snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """计算两个快照的差异。
        - 如果未提供任何参数，则默认对比 self._last_snapshot 与当前快照。
        - 兼容传入带有 {version,timestamp,data} 的结构或直接传入 snapshot() 的纯数据结构。
        """
        old_data = old_snapshot if old_snapshot is not None else self._last_snapshot
        new_data = new_snapshot if new_snapshot is not None else self.snapshot()
        if isinstance(old_data, dict) and "data" in old_data:
            old_data = old_data.get("data", {})
        if isinstance(new_data, dict) and "data" in new_data:
            new_data = new_data.get("data", {})
        diff = self._compute_diff(old_data, new_data)
        return {
            "base_version": self._version,  # 当前版本（用于参考）
            "added": diff["added"],
            "removed": diff["removed"],
            "changed": diff["changed"],
        }

    @staticmethod
    def _compute_diff(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        old_keys = set(old.keys())
        new_keys = set(new.keys())
        added = {k: new[k] for k in (new_keys - old_keys)}
        removed = {k: old[k] for k in (old_keys - new_keys)}
        common = old_keys & new_keys
        changed: Dict[str, Any] = {}
        for k in common:
            before = old[k] or {}
            after = new[k] or {}
            # 对比字段
            fields = set(before.keys()) | set(after.keys())
            field_changes = {}
            for f in fields:
                if before.get(f) != after.get(f):
                    field_changes[f] = {"from": before.get(f), "to": after.get(f)}
            if field_changes:
                changed[k] = {
                    "before": before,
                    "after": after,
                    "changed_fields": field_changes,
                }
        return {"added": added, "removed": removed, "changed": changed}

    # ---------- 工具 ----------
    @staticmethod
    def _normalize_action(action_keyword: str) -> str:
        mapping = {
            "打开": "turn_on",
            "开启": "turn_on",
            "启动": "turn_on",
            "开": "turn_on",
            "关闭": "turn_off",
            "关掉": "turn_off",
            "关": "turn_off",
            "停止": "turn_off",
            "调节": "set",
            "设置": "set",
            "调到": "set",
            "调整": "set",
            "增加": "increase",
            "提高": "increase",
            "调高": "increase",
            "加大": "increase",
            "减少": "decrease",
            "降低": "decrease",
            "调低": "decrease",
            "减小": "decrease",
        }
        return mapping.get(action_keyword, action_keyword)
    @staticmethod
    def _read_attribute(device: Device, attribute: Optional[str]) -> Any:
        if not attribute:
            return None
        aliases = {
            "亮度": "brightness",
            "音量": "volume",
            "温度": "temperature",
            "风速": "fan_speed" if isinstance(device, AirConditionerDevice) else "speed",
        }
        key = aliases.get(attribute, attribute)
        return device.state.get(key)

    def get_device_patterns(self) -> Dict[str, str]:
        """返回设备名称到类型的映射，用于意图识别"""
        patterns = {}
        for device in self.devices:
            patterns[device.name] = device.device_type
            # 同时注册带房间的名称，例如 "客厅灯"
            if device.room:
                patterns[f"{device.room}{device.name}"] = device.device_type
        return patterns
