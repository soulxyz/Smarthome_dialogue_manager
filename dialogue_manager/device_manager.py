"""设备状态机与默认预设系统

提供可编程的设备抽象、二室一厅默认设备预设、
以及统一的设备管理与执行接口，便于与对话引擎集成。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List


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

    def adjust(self, attribute: str, value: Any) -> Tuple[bool, str]:
        if attribute in ("亮度", "brightness"):
            try:
                v = int(value)
            except Exception:
                return False, f"亮度取值无效"
            v = max(0, min(100, v))
            self.state["brightness"] = v
            return True, f"已将{self.room}{self.name}亮度设置为{v}%"
        return super().adjust(attribute, value)

    def status_text(self) -> str:
        base = super().status_text()
        return f"{base}，亮度{self.state.get('brightness', 0)}%"


class AirConditionerDevice(Device):
    def __init__(self, name: str, room: str):
        super().__init__(device_type="空调", name=name, room=room, state={"on": False, "temperature": 26, "mode": "自动", "fan_speed": 2})

    def adjust(self, attribute: str, value: Any) -> Tuple[bool, str]:
        if attribute in ("温度", "temperature"):
            try:
                v = int(value)
            except Exception:
                return False, "温度取值无效"
            v = max(16, min(30, v))
            self.state["temperature"] = v
            return True, f"已将{self.room}{self.name}温度设置为{v}度"
        if attribute in ("风速", "fan_speed"):
            try:
                v = int(value)
            except Exception:
                return False, "风速取值无效"
            v = max(1, min(5, v))
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

    def adjust(self, attribute: str, value: Any) -> Tuple[bool, str]:
        if attribute in ("音量", "volume"):
            try:
                v = int(value)
            except Exception:
                return False, "音量取值无效"
            v = max(0, min(100, v))
            self.state["volume"] = v
            return True, f"已将{self.room}{self.name}音量设置为{v}"
        if attribute in ("频道", "channel"):
            try:
                v = int(value)
            except Exception:
                return False, "频道取值无效"
            v = max(1, v)
            self.state["channel"] = v
            return True, f"已将{self.room}{self.name}切换到第{v}频道"
        return super().adjust(attribute, value)

    def status_text(self) -> str:
        base = super().status_text()
        return f"{base}，音量{self.state.get('volume', 0)}"


class FanDevice(Device):
    def __init__(self, name: str, room: str):
        super().__init__(device_type="风扇", name=name, room=room, state={"on": False, "speed": 2})

    def adjust(self, attribute: str, value: Any) -> Tuple[bool, str]:
        if attribute in ("风速", "speed", "fan_speed"):
            try:
                v = int(value)
            except Exception:
                return False, "风速取值无效"
            v = max(1, min(5, v))
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

    # ---------- 预设 ----------
    def _build_default_preset(self):
        # 客厅
        self._register(LightDevice(name="客厅灯", room="客厅"))
        self._register(TVDevice(name="电视", room="客厅"))
        self._register(AirConditionerDevice(name="空调", room="客厅"))
        self._register(FanDevice(name="风扇", room="客厅"))
        # 主卧
        self._register(LightDevice(name="卧室灯", room="主卧"))
        self._register(AirConditionerDevice(name="空调", room="主卧"))
        # 次卧
        self._register(LightDevice(name="次卧灯", room="次卧"))
        self._register(FanDevice(name="风扇", room="次卧"))

    def _register(self, device: Device):
        self.devices.append(device)
        key = (device.device_type, device.room)
        self._index.setdefault(key, []).append(device)

    # ---------- 查询与执行 ----------
    def find_device(self, device_type: str, room: Optional[str]) -> Optional[Device]:
        if room:
            # 精确匹配房间
            devs = self._index.get((device_type, room))
            if devs:
                return devs[0]
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