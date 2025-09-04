"""Streamlit调试界面.

提供实时的对话调试、监控和可视化功能.
"""

import json
import os
import sys
import time
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dialogue_manager import DialogueEngine
from dialogue_manager.engine import EngineConfig  # noqa: E402

import pandas as pd
import streamlit as st
from dialogue_manager.logger import get_dialogue_logger, EventType, LogLevel


def init_session_state():
    """初始化会话状态."""
    if "dialogue_engine" not in st.session_state:
        # 从环境变量或Streamlit secrets读取API密钥
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            # 尝试从Streamlit secrets读取
            try:
                api_key = st.secrets.get("SILICONFLOW_API_KEY")
            except Exception:
                pass

        if not api_key:
            st.error("❌ 未配置API密钥！请设置环境变量 SILICONFLOW_API_KEY 或在 Streamlit secrets 中配置")
            st.info(
                '💡 配置方法：\n1. 环境变量：设置 SILICONFLOW_API_KEY=your_api_key\n2. Streamlit secrets：在 .streamlit/secrets.toml 中添加 SILICONFLOW_API_KEY = "your_api_key"'
            )
            st.stop()

        # 初始化引擎配置
        config = EngineConfig(max_turns=10, confidence_threshold=0.7, model_name="deepseek-chat", enable_clarification=True)
        st.session_state.dialogue_engine = DialogueEngine(api_key, config)
        st.session_state.engine_config = config
        # 复用MemoryManager实例
        st.session_state.memory_manager = st.session_state.dialogue_engine.memory_manager

    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None

    if "dialogue_history" not in st.session_state:
        st.session_state.dialogue_history = []

    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []

    if "user_id" not in st.session_state:
        st.session_state.user_id = "debug_user"

    # 用于存放从候选澄清按钮触发的待发送指令
    if "queued_input" not in st.session_state:
        st.session_state.queued_input = None

    # 设备事件与快照比较所需的会话状态
    if "device_events" not in st.session_state:
        st.session_state.device_events = []
    if "device_callback_registered" not in st.session_state:
        st.session_state.device_callback_registered = False
    if "snapshot_baseline" not in st.session_state:
        st.session_state.snapshot_baseline = None

    # 注册设备事件回调（只注册一次）
    if not st.session_state.device_callback_registered:
        try:
            engine = st.session_state.dialogue_engine
            dm = getattr(engine, "device_manager", None)
            if dm is not None:
                def _on_device_event(evt: dict):
                    # 只记录最近500条
                    st.session_state.device_events.append(evt)
                    if len(st.session_state.device_events) > 500:
                        st.session_state.device_events = st.session_state.device_events[-500:]
                dm.register_callback(_on_device_event)
                st.session_state.device_callback_registered = True
                # 保存引用，防止被GC
                st.session_state._device_event_callback = _on_device_event
        except Exception:
            pass


def display_header():
    """显示页面头部."""
    st.title("🏠 智能家居多轮对话管理引擎")
    st.markdown("### 调试与监控界面")

    # 显示系统状态
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("当前会话ID", st.session_state.current_session_id or "未开始")

    with col2:
        engine = st.session_state.dialogue_engine
        state = engine.current_state.value if engine else "未初始化"
        st.metric("系统状态", state)

    with col3:
        turn_count = len(st.session_state.dialogue_history)
        st.metric("对话轮数", turn_count)

    with col4:
        # 简单的健康检查
        health_status = "🟢 正常" if st.session_state.dialogue_engine else "🔴 异常"
        st.metric("系统健康", health_status)


def display_sidebar():
    """显示侧边栏控制面板."""
    st.sidebar.header("🎛️ 控制面板")

    # 用户设置
    st.sidebar.subheader("用户设置")
    user_id = st.sidebar.text_input("用户ID", value=st.session_state.user_id)
    if user_id != st.session_state.user_id:
        st.session_state.user_id = user_id

    # 会话控制
    st.sidebar.subheader("会话控制")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🚀 开始新会话", use_container_width=True):
            engine = st.session_state.dialogue_engine
            session_id = engine.start_session(st.session_state.user_id)
            st.session_state.current_session_id = session_id
            st.session_state.dialogue_history.clear()
            st.session_state.debug_logs.clear()
            st.success(f"已开始新会话: {session_id}")
            st.rerun()

    with col2:
        if st.button("🛑 结束会话", use_container_width=True):
            if st.session_state.current_session_id:
                engine = st.session_state.dialogue_engine
                engine.end_session()
                st.session_state.current_session_id = None
                st.success("会话已结束")
                st.rerun()

    # API配置
    st.sidebar.subheader("API配置")

    # 测试API连接
    if st.sidebar.button("🔗 测试API连接"):
        engine = st.session_state.dialogue_engine
        if engine.api_client.test_connection():
            st.sidebar.success("API连接正常")
        else:
            st.sidebar.error("API连接失败")

    # 显示API配置信息
    with st.sidebar.expander("API详细信息"):
        engine = st.session_state.dialogue_engine

        # 使用缓存避免频繁API调用
        @st.cache_data(ttl=1800)  # 缓存30分钟
        def get_cached_model_info():
            return engine.api_client.get_model_info()

        try:
            model_info = get_cached_model_info()
            st.json(model_info)
        except Exception as e:
            st.error(f"获取模型信息失败: {e}")

    # 系统设置
    st.sidebar.subheader("系统设置")

    # 置信度阈值
    current_threshold = st.session_state.get("engine_config", EngineConfig()).confidence_threshold
    confidence_threshold = st.sidebar.slider(
        "意图澄清阈值", min_value=0.0, max_value=1.0, value=current_threshold, step=0.1, help="低于此置信度将触发意图澄清"
    )

    # 最大对话轮数
    current_max_turns = st.session_state.get("engine_config", EngineConfig()).max_turns
    max_turns = st.sidebar.number_input(
        "最大对话轮数", min_value=1, max_value=20, value=current_max_turns, help="单次会话的最大对话轮数"
    )

    # 新增：执行模式与轨迹开关（Phase 1）
    cfg = st.session_state.get("engine_config", EngineConfig())
    mode_value = getattr(cfg, "execution_mode", "internal_first")
    execution_mode = st.sidebar.selectbox(
        "执行模式",
        options=["internal_first", "llm_first", "parallel"],
        index=["internal_first", "llm_first", "parallel"].index(mode_value) if mode_value in ["internal_first", "llm_first", "parallel"] else 0,
        help="internal_first: 内部逻辑优先; llm_first: 大模型优先; parallel: 并行（第一阶段为顺序模拟）"
    )
    always_record_api_traces = st.sidebar.checkbox(
        "始终记录API轨迹(含确定性路径)",
        value=getattr(cfg, "always_record_api_traces", True)
    )

    # 检查配置是否有变化，如果有则更新引擎配置
    if "engine_config" in st.session_state and (
        confidence_threshold != st.session_state.engine_config.confidence_threshold
        or max_turns != st.session_state.engine_config.max_turns
        or execution_mode != getattr(st.session_state.engine_config, "execution_mode", "internal_first")
        or always_record_api_traces != getattr(st.session_state.engine_config, "always_record_api_traces", True)
    ):

        # 更新配置
        st.session_state.engine_config.update(
            confidence_threshold=confidence_threshold,
            max_turns=max_turns,
            execution_mode=execution_mode,
            always_record_api_traces=always_record_api_traces,
        )

        # 更新引擎配置
        if "dialogue_engine" in st.session_state:
            st.session_state.dialogue_engine.update_config(
                confidence_threshold=confidence_threshold,
                max_turns=max_turns,
                execution_mode=execution_mode,
                always_record_api_traces=always_record_api_traces,
            )

            # 更新子组件的配置
            st.session_state.dialogue_engine.intent_recognizer.confidence_threshold = confidence_threshold
            st.session_state.dialogue_engine.clarification_agent.confidence_threshold = confidence_threshold

        st.sidebar.success(
            f"✅ 配置已更新: 阈值={confidence_threshold}, 最大轮数={max_turns}, 模式={execution_mode}, 记录轨迹={always_record_api_traces}"
        )

    # 清理按钮
    st.sidebar.subheader("数据管理")
    if st.sidebar.button("🗑️ 清空调试日志"):
        st.session_state.debug_logs.clear()
        st.sidebar.success("调试日志已清空")


def display_chat_interface():
    """显示对话界面."""
    st.header("💬 对话界面")

    # 如果有待处理的排队输入（来自澄清候选按钮），优先处理
    if st.session_state.get("queued_input"):
        queued = st.session_state.queued_input
        st.session_state.queued_input = None
        process_user_input(queued)
        return

    # 检查是否有活跃会话
    if not st.session_state.current_session_id:
        st.warning("请先在侧边栏开始一个新会话")
        return

    # 显示对话历史
    chat_container = st.container()

    with chat_container:
        for i, turn in enumerate(st.session_state.dialogue_history):
            # 用户消息
            with st.chat_message("user"):
                st.write(turn["user_input"])
                st.caption(f"轮次 {i+1} | {turn['timestamp']}")

            # 系统回复
            with st.chat_message("assistant"):
                st.write(turn["system_response"])

                # 显示意图识别结果
                if "intent_result" in turn:
                    intent_result = turn["intent_result"]
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("识别意图", intent_result.get("intent", "unknown"))
                    with col2:
                        confidence = intent_result.get("confidence", 0.0)
                        st.metric("置信度", f"{confidence:.2f}")
                    with col3:
                        need_clarification = intent_result.get("need_clarification", False)
                        status = "需要澄清" if need_clarification else "无需澄清"
                        st.metric("澄清状态", status)

    # 用户输入框
    user_input = st.chat_input("请输入您的指令...")

    if user_input:
        process_user_input(user_input)


def process_user_input(user_input: str):
    """处理用户输入."""
    engine = st.session_state.dialogue_engine

    # 显示用户输入
    with st.chat_message("user"):
        st.write(user_input)

    # 处理输入并获取响应
    with st.spinner("正在处理..."):
        try:
            start_time = time.time()
            response, debug_info = engine.process_input(user_input)
            processing_time = time.time() - start_time
        except Exception as e:
            st.error(f"处理用户输入时发生错误: {str(e)}")
            st.error("请检查API连接和配置，或尝试重新开始会话")
            # 记录错误到日志
            import traceback
            st.code(traceback.format_exc(), language="python")
            return

    # 显示系统响应
    with st.chat_message("assistant"):
        st.write(response)

        # 显示处理信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("处理时间", f"{processing_time:.2f}s")
        with col2:
            intent = debug_info.get("intent_result", {}).get("intent", "unknown")
            st.metric("识别意图", intent)
        with col3:
            confidence = debug_info.get("intent_result", {}).get("confidence", 0.0)
            st.metric("置信度", f"{confidence:.2f}")

        # 如果需要澄清，渲染候选指令快捷按钮
        need_clarification = debug_info.get("intent_result", {}).get("need_clarification", False)
        candidates = debug_info.get("clarification_candidates", [])
        if need_clarification and candidates:
            st.markdown("**可能的表达:**")
            for cand in candidates:
                if st.button(cand, key=f"clarify_{len(st.session_state.dialogue_history)}_{cand}"):
                    st.session_state.queued_input = cand
                    st.rerun()

    # 保存到对话历史
    turn_data = {
        "user_input": user_input,
        "system_response": response,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "processing_time": processing_time,
        "intent_result": debug_info.get("intent_result", {}),
        "debug_info": debug_info,
    }

    st.session_state.dialogue_history.append(turn_data)
    st.session_state.debug_logs.append(debug_info)

    # 自动保存到数据库（每次对话后）
    if st.session_state.current_session_id:
        try:
            # 使用复用的MemoryManager实例
            memory_manager = st.session_state.get("memory_manager")
            if memory_manager:
                memory_manager.save_session(
                    st.session_state.current_session_id, st.session_state.dialogue_history, st.session_state.user_id
                )
        except Exception as e:
            st.error(f"保存对话记录时出错: {e}")

    # 刷新页面以显示新消息
    st.rerun()


def display_intent_tab(tab, debug_info):
    """显示意图识别标签内容.

    Args:
        tab: Streamlit tab
        debug_info: 调试信息
    """
    st.subheader("意图识别结果")
    intent_result = debug_info.get("intent_result", {})

    col1, col2 = st.columns(2)
    with col1:
        st.json(
            {
                "intent": intent_result.get("intent"),
                "confidence": intent_result.get("confidence"),
                "need_clarification": intent_result.get("need_clarification"),
            }
        )

    with col2:
        entities = intent_result.get("entities", [])
        if entities:
            st.subheader("识别的实体")
            try:
                cleaned_entities = [entity if isinstance(entity, dict) else {"entity": str(entity)} for entity in entities]
                if cleaned_entities:
                    entity_df = pd.DataFrame(cleaned_entities)
                    st.dataframe(entity_df)
                else:
                    st.info("实体数据格式异常")
            except Exception as e:
                st.error(f"显示实体数据时出错: {e}")
                st.json(entities)
        else:
            st.info("未识别到实体")

def display_state_tab(tab, debug_info):
    """显示状态转换标签内容.

    Args:
        tab: Streamlit tab
        debug_info: 调试信息
    """
    st.subheader("状态转换历史")
    transitions = debug_info.get("state_transitions", [])
    if transitions:
        try:
            cleaned_transitions = [transition if isinstance(transition, dict) else {"transition": str(transition)} for transition in transitions]
            if cleaned_transitions:
                transition_df = pd.DataFrame(cleaned_transitions)
                st.dataframe(transition_df)
            else:
                st.info("状态转换数据格式异常")
        except Exception as e:
            st.error(f"显示状态转换数据时出错: {e}")
            st.json(transitions)
    else:
        st.info("无状态转换记录")

def display_api_tab(tab, debug_info):
    """显示API调用标签内容.

    Args:
        tab: Streamlit tab
        debug_info: 调试信息
    """
    st.subheader("API调用记录")
    # 新增：内部/LLM计划与差异
    if debug_info.get("internal_plan"):
        st.info("内部计划(internal)")
        st.json(debug_info["internal_plan"]) 
    if debug_info.get("plan_diff"):
        st.warning("计划差异(plan_diff)")
        st.json(debug_info["plan_diff"]) 

    api_calls = debug_info.get("api_calls", [])
    if api_calls:
        # 显示API调用统计信息
        total_calls = len(api_calls)
        success_calls = sum(1 for call in api_calls if call.get("success", False))
        failed_calls = total_calls - success_calls
        avg_response_time = sum(call.get("response_time", 0) for call in api_calls) / total_calls if total_calls > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总调用次数", total_calls)
        with col2:
            st.metric("成功率", f"{(success_calls/total_calls*100):.1f}%" if total_calls > 0 else "N/A")
        with col3:
            st.metric("平均响应时间", f"{avg_response_time:.2f}s")
            
        # 如果有失败的调用，显示警告
        if failed_calls > 0:
            st.warning(f"⚠️ 有 {failed_calls} 次API调用失败，请展开查看详情")
            
        for i, call in enumerate(api_calls):
            # 根据成功状态设置不同的样式
            is_success = call.get("success", False)
            expander_label = f"API调用 {i+1} - {'✅ 成功' if is_success else '❌ 失败'} - 响应时间: {call.get('response_time', 0):.2f}s"
            
            with st.expander(expander_label):
                req_tab, resp_tab, summary_tab, debug_tab = st.tabs(["请求", "响应", "摘要", "调试信息"])

                with summary_tab:
                    st.json(
                        {
                            "success": call.get("success", False),
                            "content": call.get("content", ""),
                            "error": call.get("error_message", call.get('error', "")),
                            "response_time": call.get("response_time", 0),
                        }
                    )
                    
                with debug_tab:
                    # 显示更多调试信息
                    if not is_success:
                        st.error("错误详情")
                        error_msg = call.get("error_message", call.get('error', "未知错误"))
                        st.code(error_msg)
                        
                        # 提供可能的解决方案
                        if "timeout" in error_msg.lower():
                            st.info("💡 可能的解决方案: 增加超时时间或检查网络连接")
                        elif "rate limit" in error_msg.lower():
                            st.info("💡 可能的解决方案: 降低请求频率或等待一段时间后重试")
                        elif "connection" in error_msg.lower():
                            st.info("💡 可能的解决方案: 检查网络连接或API服务状态")
                            
                with req_tab:
                    request_data = call.get("request", {})
                    if request_data:
                        st.subheader("请求消息")
                        messages = request_data.get("messages", [])
                        for msg_idx, msg in enumerate(messages):
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            st.text_area(f"{role.capitalize()}", content, height=100, disabled=True, key=f"api_req_msg_{i}_{msg_idx}")

                        st.subheader("请求参数")
                        st.json({"model": request_data.get("model", ""), "mode": request_data.get("mode", ""), "note": request_data.get("note", "")})
                    else:
                        st.info("无请求数据")

                with resp_tab:
                    response_data = call.get("response", {})
                    if response_data:
                        st.json(response_data)
                    else:
                        st.info("无响应数据")
    else:
        st.info("无API调用记录")

def display_context_tab(tab, debug_info):
    """显示上下文更新标签内容.

    Args:
        tab: Streamlit tab
        debug_info: 调试信息
    """
    st.subheader("上下文更新")
    context_updates = debug_info.get("context_updates", {})
    if context_updates:
        st.json(context_updates)
    else:
        st.info("无上下文更新")

def display_debug_panel():
    """显示调试面板."""
    st.header("🔍 调试面板")

    if not st.session_state.debug_logs:
        st.info("暂无调试信息")
        return

    selected_turn = st.selectbox("选择对话轮次", range(len(st.session_state.debug_logs)), format_func=lambda x: f"第 {x+1} 轮")

    if selected_turn < len(st.session_state.debug_logs):
        debug_info = st.session_state.debug_logs[selected_turn]

        tab1, tab2, tab3, tab4 = st.tabs(["意图识别", "状态转换", "API调用", "上下文更新"])

        with tab1:
            display_intent_tab(tab1, debug_info)

        with tab2:
            display_state_tab(tab2, debug_info)

        with tab3:
            display_api_tab(tab3, debug_info)

        with tab4:
            display_context_tab(tab4, debug_info)


@st.cache_data
def calculate_statistics(dialogue_history_json: str):
    """计算统计数据（带缓存）."""
    dialogue_history = json.loads(dialogue_history_json)

    if not dialogue_history:
        return None

    total_turns = len(dialogue_history)
    avg_processing_time = sum(turn.get("processing_time", 0) for turn in dialogue_history) / total_turns
    clarification_count = sum(1 for turn in dialogue_history if turn.get("intent_result", {}).get("need_clarification", False))
    clarification_rate = (clarification_count / total_turns) * 100 if total_turns > 0 else 0
    avg_confidence = sum(turn.get("intent_result", {}).get("confidence", 0) for turn in dialogue_history) / total_turns

    # 意图分布
    intent_counts = {}
    for turn in dialogue_history:
        intent = turn.get("intent_result", {}).get("intent", "unknown")
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    # 处理时间序列
    processing_times = [turn.get("processing_time", 0) for turn in dialogue_history]

    return {
        "total_turns": total_turns,
        "avg_processing_time": avg_processing_time,
        "clarification_rate": clarification_rate,
        "avg_confidence": avg_confidence,
        "intent_counts": intent_counts,
        "processing_times": processing_times,
    }


def display_statistics():
    """显示统计信息."""
    st.header("📊 统计信息")

    if not st.session_state.dialogue_history:
        st.info("暂无统计数据")
        return

    # 使用缓存计算统计数据
    dialogue_history_json = json.dumps(st.session_state.dialogue_history)
    stats = calculate_statistics(dialogue_history_json)

    if not stats:
        st.info("暂无统计数据")
        return

    # 基本统计
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("总对话轮数", stats["total_turns"])

    with col2:
        st.metric("平均处理时间", f"{stats['avg_processing_time']:.2f}s")

    with col3:
        st.metric("澄清触发率", f"{stats['clarification_rate']:.1f}%")

    with col4:
        st.metric("平均置信度", f"{stats['avg_confidence']:.2f}")

    # 意图分布图
    st.subheader("意图分布")
    if stats["intent_counts"]:
        try:
            intent_df = pd.DataFrame(list(stats["intent_counts"].items()), columns=["意图", "次数"])
            st.bar_chart(intent_df.set_index("意图"))
        except Exception as e:
            st.error(f"显示意图分布图时出错: {e}")
            st.json(stats["intent_counts"])

    # 处理时间趋势
    st.subheader("处理时间趋势")
    if stats["processing_times"]:
        try:
            time_df = pd.DataFrame(
                {"轮次": range(1, len(stats["processing_times"]) + 1), "处理时间(秒)": stats["processing_times"]}
            )
            st.line_chart(time_df.set_index("轮次"))
        except Exception as e:
            st.error(f"显示处理时间趋势图时出错: {e}")
            st.json({"processing_times": stats["processing_times"]})


def display_device_panel():
    """设备概览 + 快照对比 初版"""
    st.header("🧰 设备面板")
    engine = st.session_state.dialogue_engine
    dm = getattr(engine, "device_manager", None)

    if dm is None:
        st.warning("设备管理器未启用")
        return

    # 顶部指标
    meta = dm.snapshot_with_meta()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("版本", meta.get("version", 0))
    with col2:
        st.metric("最近更新时间", meta.get("timestamp") or "—")
    with col3:
        auto_refresh = st.checkbox("自动刷新(2s)", value=False, help="用于观察实时变化")
        if auto_refresh:
            # 简化处理：提供一次性刷新按钮，避免依赖不可用的 st.autorefresh
            if st.button("立即刷新一次", key="dev_panel_refresh_once"):
                st.rerun()
            st.autorefresh(interval=2000, key="dev_panel_autorefresh")

    # 设备概览
    st.subheader("设备概览")
    data = meta.get("data", {})
    rows = []
    for key, attrs in data.items():
        room, dtype = key.split("-", 1) if "-" in key else ("", key)
        row = {"房间": room, "设备类型": dtype, "开启": attrs.get("on")}
        for k, v in attrs.items():
            if k != "on":
                row[k] = v
        rows.append(row)
    try:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
    except Exception:
        st.json(data)

    # 快照对比
    st.subheader("快照对比")
    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        if st.button("保存当前为基准(A)"):
            st.session_state.snapshot_baseline = dm.snapshot_with_meta()
            st.success("已保存当前快照为基准(A)")
    with colB:
        if st.button("清空基准"):
            st.session_state.snapshot_baseline = None
            st.info("已清空基准")
    with colC:
        st.caption("请选择左侧操作保存基准后，再点击下方按钮计算差异")

    baseline = st.session_state.get("snapshot_baseline")
    if baseline:
        st.info(
            f"基准版本: v{baseline.get('version')} @ {baseline.get('timestamp') or '—'}"
        )
        if st.button("计算与当前快照(B)的差异"):
            diff = dm.snapshot_diff(baseline, dm.snapshot_with_meta())
            # 展示差异
            added = diff.get("added", {})
            removed = diff.get("removed", {})
            changed = diff.get("changed", {})

            if not added and not removed and not changed:
                st.success("无变化 ✅")
            else:
                if added:
                    with st.expander("新增设备"):
                        st.json(added)
                if removed:
                    with st.expander("移除设备"):
                        st.json(removed)
                if changed:
                    st.subheader("变更详情")
                    for dev_key, detail in changed.items():
                        with st.expander(dev_key):
                            st.json(detail)

    # 事件日志
    st.subheader("事件日志（最近50条）")
    events = st.session_state.get("device_events", [])[-50:]
    if events:
        try:
            # 扁平化事件，适合表格展示
            flat_rows = []
            for e in events:
                dev = (e.get("device") or {})
                flat_rows.append({
                    "时间": e.get("timestamp"),
                    "版本": e.get("version"),
                    "事件": e.get("event"),
                    "房间": dev.get("room"),
                    "设备类型": dev.get("device_type"),
                    "设备名": dev.get("name"),
                    "动作": e.get("action"),
                    "属性": e.get("attribute"),
                    "消息": e.get("message"),
                })
            st.dataframe(pd.DataFrame(flat_rows), use_container_width=True)
        except Exception:
            st.json(events)
    else:
        st.info("暂无事件")


def display_log_panel():
    """显示日志面板"""
    st.header("📋 日志查看与分析")
    
    # 获取日志记录器
    dialogue_logger = get_dialogue_logger()
    
    # 日志搜索控制
    st.subheader("🔍 日志搜索")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 会话过滤
        search_session_id = st.text_input("会话ID", placeholder="输入会话ID进行过滤")
        
        # 事件类型过滤
        event_types = [e.value for e in EventType]
        selected_event_type = st.selectbox("事件类型", ["全部"] + event_types)
        if selected_event_type == "全部":
            selected_event_type = None
    
    with col2:
        # 日志级别过滤
        log_levels = [l.value for l in LogLevel]
        selected_log_level = st.selectbox("日志级别", ["全部"] + log_levels)
        if selected_log_level == "全部":
            selected_log_level = None
        
        # 关键词搜索
        keyword = st.text_input("关键词搜索", placeholder="搜索消息或错误类型")
    
    with col3:
        # 时间范围
        time_range = st.selectbox("时间范围", ["最近1小时", "最近6小时", "最近24小时", "最近7天", "自定义"])
        
        if time_range == "自定义":
            start_date = st.date_input("开始日期")
            end_date = st.date_input("结束日期")
            start_time = start_date.timestamp() if start_date else None
            end_time = end_date.timestamp() + 86400 if end_date else None  # 加一天到结束
        else:
            # 预设时间范围
            import time
            current_time = time.time()
            if time_range == "最近1小时":
                start_time = current_time - 3600
            elif time_range == "最近6小时":
                start_time = current_time - 6 * 3600
            elif time_range == "最近24小时":
                start_time = current_time - 24 * 3600
            elif time_range == "最近7天":
                start_time = current_time - 7 * 24 * 3600
            else:
                start_time = None
            end_time = current_time
    
    # 搜索按钮和结果限制
    col_search, col_limit = st.columns([2, 1])
    with col_search:
        search_clicked = st.button("🔍 搜索日志", type="primary")
    with col_limit:
        result_limit = st.number_input("结果数量", min_value=10, max_value=1000, value=100, step=10)
    
    # 执行搜索
    if search_clicked or st.session_state.get("auto_refresh_logs", False):
        try:
            logs = dialogue_logger.search_logs(
                session_id=search_session_id if search_session_id else None,
                event_type=selected_event_type,
                level=selected_log_level,
                start_time=start_time,
                end_time=end_time,
                keyword=keyword if keyword else None,
                limit=result_limit
            )
            
            if logs:
                st.success(f"找到 {len(logs)} 条日志记录")
                
                # 日志统计
                st.subheader("📊 日志统计")
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                with col_stat1:
                    total_logs = len(logs)
                    st.metric("总日志数", total_logs)
                
                with col_stat2:
                    error_logs = len([log for log in logs if log.level == LogLevel.ERROR.value])
                    st.metric("错误日志", error_logs)
                
                with col_stat3:
                    dialogue_turns = len([log for log in logs if log.event_type == EventType.DIALOGUE_TURN.value])
                    st.metric("对话轮数", dialogue_turns)
                
                with col_stat4:
                    api_calls = len([log for log in logs if log.event_type == EventType.API_CALL.value])
                    st.metric("API调用", api_calls)
                
                # 日志详情展示
                st.subheader("📝 日志详情")
                
                # 创建表格数据
                log_data = []
                for log in logs:
                    log_data.append({
                        "时间": datetime.fromtimestamp(log.timestamp).strftime("%m-%d %H:%M:%S"),
                        "级别": log.level,
                        "事件类型": log.event_type,
                        "会话ID": log.session_id[-8:] if log.session_id else "N/A",  # 显示后8位
                        "轮次": str(log.turn_id) if log.turn_id is not None else "N/A",  # 确保转换为字符串
                        "意图": log.intent if log.intent else "N/A",
                        "置信度": f"{log.confidence:.2f}" if log.confidence else "N/A",
                        "处理时间": f"{log.processing_time:.2f}s" if log.processing_time else "N/A",
                        "消息": log.message[:50] + "..." if len(log.message) > 50 else log.message
                    })
                
                # 显示表格
                df = pd.DataFrame(log_data)
                st.dataframe(df, use_container_width=True, height=400)
                
                # 详细日志查看
                st.subheader("🔍 详细日志查看")
                
                # 选择日志条目
                log_options = [f"{i+1}. {log.event_type} - {datetime.fromtimestamp(log.timestamp).strftime('%H:%M:%S')} - {log.message[:30]}..." 
                              for i, log in enumerate(logs)]
                
                if log_options:
                    selected_log_index = st.selectbox("选择日志条目查看详情", range(len(log_options)), 
                                                    format_func=lambda x: log_options[x])
                    
                    if selected_log_index < len(logs):
                        selected_log = logs[selected_log_index]
                        
                        # 显示选中日志的详细信息
                        col_detail1, col_detail2 = st.columns(2)
                        
                        with col_detail1:
                            st.info("**基本信息**")
                            st.write(f"**时间**: {datetime.fromtimestamp(selected_log.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
                            st.write(f"**级别**: {selected_log.level}")
                            st.write(f"**事件类型**: {selected_log.event_type}")
                            st.write(f"**会话ID**: {selected_log.session_id}")
                            st.write(f"**用户ID**: {selected_log.user_id or 'N/A'}")
                            st.write(f"**轮次ID**: {selected_log.turn_id or 'N/A'}")
                        
                        with col_detail2:
                            st.info("**性能信息**")
                            st.write(f"**处理时间**: {selected_log.processing_time or 'N/A'}s")
                            st.write(f"**意图**: {selected_log.intent or 'N/A'}")
                            st.write(f"**置信度**: {selected_log.confidence or 'N/A'}")
                            st.write(f"**API调用数**: {selected_log.api_calls_count or 'N/A'}")
                            if selected_log.error_type:
                                st.write(f"**错误类型**: {selected_log.error_type}")
                        
                        # 消息内容
                        st.info("**消息内容**")
                        st.code(selected_log.message, language="text")
                        
                        # 上下文数据
                        if selected_log.context_data:
                            st.info("**上下文数据**")
                            st.json(selected_log.context_data)
                        
                        # 错误追踪
                        if selected_log.error_traceback:
                            st.error("**错误追踪**")
                            st.code(selected_log.error_traceback, language="python")
            else:
                st.warning("未找到匹配的日志记录")
                
        except Exception as e:
            st.error(f"搜索日志时出错: {str(e)}")
    
    # 日志管理功能
    st.subheader("🛠️ 日志管理")
    
    col_mgmt1, col_mgmt2, col_mgmt3 = st.columns(3)
    
    with col_mgmt1:
        # 会话摘要
        if st.button("📊 会话摘要"):
            if search_session_id:
                try:
                    summary = dialogue_logger.get_session_summary(search_session_id)
                    if "error" not in summary:
                        st.json(summary)
                    else:
                        st.error(summary["error"])
                except Exception as e:
                    st.error(f"获取会话摘要时出错: {str(e)}")
            else:
                st.warning("请先输入会话ID")
    
    with col_mgmt2:
        # 导出日志
        if st.button("📥 导出日志"):
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
                    temp_file = f.name
                
                dialogue_logger.export_logs(
                    output_file=temp_file,
                    session_id=search_session_id if search_session_id else None,
                    start_time=start_time,
                    end_time=end_time
                )
                
                # 读取文件并提供下载
                with open(temp_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                
                st.download_button(
                    label="下载日志文件",
                    data=log_content,
                    file_name=f"dialogue_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
                    mime="application/json"
                )
                
                # 清理临时文件
                import os
                os.unlink(temp_file)
                
            except Exception as e:
                st.error(f"导出日志时出错: {str(e)}")
    
    with col_mgmt3:
        # 清理旧日志
        if st.button("🗑️ 清理旧日志"):
            try:
                days_to_keep = st.number_input("保留天数", min_value=1, max_value=365, value=90)
                if st.button("确认清理", type="secondary"):
                    dialogue_logger.cleanup_old_logs(days_to_keep)
                    st.success(f"已清理超过 {days_to_keep} 天的旧日志")
            except Exception as e:
                st.error(f"清理日志时出错: {str(e)}")
    
    # 自动刷新选项
    st.subheader("⚙️ 显示设置")
    auto_refresh = st.checkbox("自动刷新日志 (每30秒)", key="auto_refresh_logs")
    if auto_refresh:
        st.autorefresh(interval=30000, key="log_autorefresh")


def main():
    """主函数."""
    # 配置页面设置（必须在任何Streamlit调用之前）
    st.set_page_config(
        page_title="智能家居对话管理引擎 - 调试界面", page_icon="🏠", layout="wide", initial_sidebar_state="expanded"
    )

    init_session_state()
    display_header()
    display_sidebar()

    # 主要内容区域
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 对话", "🔍 调试", "📊 统计", "🧰 设备", "📋 日志"])

    with tab1:
        display_chat_interface()

    with tab2:
        display_debug_panel()

    with tab3:
        display_statistics()

    with tab4:
        display_device_panel()
        
    with tab5:
        display_log_panel()

    # 页脚信息
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "智能家居多轮对话管理引擎 v0.1.0 | "
        "Powered by DeepSeek & SiliconFlow"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
