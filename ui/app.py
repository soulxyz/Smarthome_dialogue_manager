"""Streamlit调试界面

提供实时的对话调试、监控和可视化功能。
"""

import streamlit as st
import json
import time
import pandas as pd
from datetime import datetime
from typing import Dict, List
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dialogue_manager import DialogueEngine, MemoryManager


def init_session_state():
    """初始化会话状态"""
    if 'dialogue_engine' not in st.session_state:
        # 从环境变量或配置文件读取API密钥
        api_key = os.getenv('SILICONFLOW_API_KEY', 'sk-fsjrtevskkmicnqdjjqdjarwqktxzwchkknnjmwgicczbubp')
        st.session_state.dialogue_engine = DialogueEngine(api_key)
    
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    
    if 'dialogue_history' not in st.session_state:
        st.session_state.dialogue_history = []
    
    if 'debug_logs' not in st.session_state:
        st.session_state.debug_logs = []
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = 'debug_user'
    
    # 用于存放从候选澄清按钮触发的待发送指令
    if 'queued_input' not in st.session_state:
        st.session_state.queued_input = None


def display_header():
    """显示页面头部"""
    st.set_page_config(
        page_title="智能家居对话管理引擎 - 调试界面",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
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
    """显示侧边栏控制面板"""
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
        model_info = engine.api_client.get_model_info()
        st.json(model_info)
    
    # 系统设置
    st.sidebar.subheader("系统设置")
    
    # 置信度阈值
    confidence_threshold = st.sidebar.slider(
        "意图澄清阈值",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="低于此置信度将触发意图澄清"
    )
    
    # 最大对话轮数
    max_turns = st.sidebar.number_input(
        "最大对话轮数",
        min_value=1,
        max_value=20,
        value=10,
        help="单次会话的最大对话轮数"
    )
    
    # 清理按钮
    st.sidebar.subheader("数据管理")
    if st.sidebar.button("🗑️ 清空调试日志"):
        st.session_state.debug_logs.clear()
        st.sidebar.success("调试日志已清空")


def display_chat_interface():
    """显示对话界面"""
    st.header("💬 对话界面")
    
    # 如果有待处理的排队输入（来自澄清候选按钮），优先处理
    if st.session_state.get('queued_input'):
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
                st.write(turn['user_input'])
                st.caption(f"轮次 {i+1} | {turn['timestamp']}")
            
            # 系统回复
            with st.chat_message("assistant"):
                st.write(turn['system_response'])
                
                # 显示意图识别结果
                if 'intent_result' in turn:
                    intent_result = turn['intent_result']
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("识别意图", intent_result.get('intent', 'unknown'))
                    with col2:
                        confidence = intent_result.get('confidence', 0.0)
                        st.metric("置信度", f"{confidence:.2f}")
                    with col3:
                        need_clarification = intent_result.get('need_clarification', False)
                        status = "需要澄清" if need_clarification else "无需澄清"
                        st.metric("澄清状态", status)
    
    # 用户输入框
    user_input = st.chat_input("请输入您的指令...")
    
    if user_input:
        process_user_input(user_input)


def process_user_input(user_input: str):
    """处理用户输入"""
    engine = st.session_state.dialogue_engine
    
    # 显示用户输入
    with st.chat_message("user"):
        st.write(user_input)
    
    # 处理输入并获取响应
    with st.spinner("正在处理..."):
        start_time = time.time()
        response, debug_info = engine.process_input(user_input)
        processing_time = time.time() - start_time
    
    # 显示系统响应
    with st.chat_message("assistant"):
        st.write(response)
        
        # 显示处理信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("处理时间", f"{processing_time:.2f}s")
        with col2:
            intent = debug_info.get('intent_result', {}).get('intent', 'unknown')
            st.metric("识别意图", intent)
        with col3:
            confidence = debug_info.get('intent_result', {}).get('confidence', 0.0)
            st.metric("置信度", f"{confidence:.2f}")
        
        # 如果需要澄清，渲染候选指令快捷按钮
        need_clarification = debug_info.get('intent_result', {}).get('need_clarification', False)
        candidates = debug_info.get('clarification_candidates', [])
        if need_clarification and candidates:
            st.markdown("**可能的表达:**")
            for cand in candidates:
                if st.button(cand, key=f"clarify_{len(st.session_state.dialogue_history)}_{cand}"):
                    st.session_state.queued_input = cand
                    st.rerun()
    
    # 保存到对话历史
    turn_data = {
        'user_input': user_input,
        'system_response': response,
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'processing_time': processing_time,
        'intent_result': debug_info.get('intent_result', {}),
        'debug_info': debug_info
    }
    
    st.session_state.dialogue_history.append(turn_data)
    st.session_state.debug_logs.append(debug_info)
    
    # 自动保存到数据库（每次对话后）
    if st.session_state.current_session_id:
        try:
            memory_manager = MemoryManager()
            memory_manager.save_session(
                st.session_state.current_session_id,
                st.session_state.dialogue_history,
                st.session_state.user_id
            )
        except Exception as e:
            st.error(f"保存对话记录时出错: {e}")
    
    # 刷新页面以显示新消息
    st.rerun()


def display_debug_panel():
    """显示调试面板"""
    st.header("🔍 调试面板")
    
    if not st.session_state.debug_logs:
        st.info("暂无调试信息")
        return
    
    # 选择要查看的调试信息
    selected_turn = st.selectbox(
        "选择对话轮次",
        range(len(st.session_state.debug_logs)),
        format_func=lambda x: f"第 {x+1} 轮"
    )
    
    if selected_turn < len(st.session_state.debug_logs):
        debug_info = st.session_state.debug_logs[selected_turn]
        
        # 创建标签页
        tab1, tab2, tab3, tab4 = st.tabs(["意图识别", "状态转换", "API调用", "上下文更新"])
        
        with tab1:
            st.subheader("意图识别结果")
            intent_result = debug_info.get('intent_result', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.json({
                    "intent": intent_result.get('intent'),
                    "confidence": intent_result.get('confidence'),
                    "need_clarification": intent_result.get('need_clarification')
                })
            
            with col2:
                entities = intent_result.get('entities', [])
                if entities:
                    st.subheader("识别的实体")
                    try:
                        # 确保实体数据格式一致
                        cleaned_entities = []
                        for entity in entities:
                            if isinstance(entity, dict):
                                cleaned_entities.append(entity)
                            else:
                                # 如果不是字典，尝试转换
                                cleaned_entities.append({"entity": str(entity)})
                        
                        if cleaned_entities:
                            entity_df = pd.DataFrame(cleaned_entities)
                            st.dataframe(entity_df)
                        else:
                            st.info("实体数据格式异常")
                    except Exception as e:
                        st.error(f"显示实体数据时出错: {e}")
                        st.json(entities)  # 显示原始数据
                else:
                    st.info("未识别到实体")
        
        with tab2:
            st.subheader("状态转换历史")
            transitions = debug_info.get('state_transitions', [])
            if transitions:
                try:
                    # 确保转换数据格式一致
                    cleaned_transitions = []
                    for transition in transitions:
                        if isinstance(transition, dict):
                            cleaned_transitions.append(transition)
                        else:
                            cleaned_transitions.append({"transition": str(transition)})
                    
                    if cleaned_transitions:
                        transition_df = pd.DataFrame(cleaned_transitions)
                        st.dataframe(transition_df)
                    else:
                        st.info("状态转换数据格式异常")
                except Exception as e:
                    st.error(f"显示状态转换数据时出错: {e}")
                    st.json(transitions)  # 显示原始数据
            else:
                st.info("无状态转换记录")
        
        with tab3:
            st.subheader("API调用记录")
            api_calls = debug_info.get('api_calls', [])
            if api_calls:
                for i, call in enumerate(api_calls):
                    with st.expander(f"API调用 {i+1} - 响应时间: {call.get('response_time', 0):.2f}s"):
                        # 创建请求和响应的标签页
                        req_tab, resp_tab, summary_tab = st.tabs(["请求", "响应", "摘要"])
                        
                        with summary_tab:
                            st.json({
                                "success": call.get('success', False),
                                "content": call.get('content', ""),
                                "error": call.get('error_message', ""),
                                "response_time": call.get('response_time', 0)
                            })
                        
                        with req_tab:
                            request_data = call.get('request', {})
                            if request_data:
                                st.subheader("请求消息")
                                messages = request_data.get('messages', [])
                                for msg in messages:
                                    role = msg.get('role', '')
                                    content = msg.get('content', '')
                                    st.text_area(f"{role.capitalize()}", content, height=100, disabled=True)
                                
                                st.subheader("请求参数")
                                st.json({
                                    "model": request_data.get('model', ''),
                                })
                            else:
                                st.info("无请求数据")
                        
                        with resp_tab:
                            response_data = call.get('response', {})
                            if response_data:
                                st.json(response_data)
                            else:
                                st.info("无响应数据")
            else:
                st.info("无API调用记录")
        
        with tab4:
            st.subheader("上下文更新")
            context_updates = debug_info.get('context_updates', {})
            if context_updates:
                st.json(context_updates)
            else:
                st.info("无上下文更新")


def display_statistics():
    """显示统计信息"""
    st.header("📊 统计信息")
    
    if not st.session_state.dialogue_history:
        st.info("暂无统计数据")
        return
    
    # 基本统计
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_turns = len(st.session_state.dialogue_history)
        st.metric("总对话轮数", total_turns)
    
    with col2:
        avg_processing_time = sum(turn.get('processing_time', 0) for turn in st.session_state.dialogue_history) / total_turns
        st.metric("平均处理时间", f"{avg_processing_time:.2f}s")
    
    with col3:
        clarification_count = sum(1 for turn in st.session_state.dialogue_history 
                                if turn.get('intent_result', {}).get('need_clarification', False))
        clarification_rate = (clarification_count / total_turns) * 100 if total_turns > 0 else 0
        st.metric("澄清触发率", f"{clarification_rate:.1f}%")
    
    with col4:
        avg_confidence = sum(turn.get('intent_result', {}).get('confidence', 0) for turn in st.session_state.dialogue_history) / total_turns
        st.metric("平均置信度", f"{avg_confidence:.2f}")
    
    # 意图分布图
    st.subheader("意图分布")
    intent_counts = {}
    for turn in st.session_state.dialogue_history:
        intent = turn.get('intent_result', {}).get('intent', 'unknown')
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    if intent_counts:
        try:
            intent_df = pd.DataFrame(list(intent_counts.items()), columns=['意图', '次数'])
            st.bar_chart(intent_df.set_index('意图'))
        except Exception as e:
            st.error(f"显示意图分布图时出错: {e}")
            st.json(intent_counts)
    
    # 处理时间趋势
    st.subheader("处理时间趋势")
    processing_times = [turn.get('processing_time', 0) for turn in st.session_state.dialogue_history]
    if processing_times:
        try:
            time_df = pd.DataFrame({
                '轮次': range(1, len(processing_times) + 1),
                '处理时间(秒)': processing_times
            })
            st.line_chart(time_df.set_index('轮次'))
        except Exception as e:
            st.error(f"显示处理时间趋势图时出错: {e}")
            st.json({"processing_times": processing_times})


def main():
    """主函数"""
    init_session_state()
    display_header()
    display_sidebar()
    
    # 主要内容区域
    tab1, tab2, tab3 = st.tabs(["💬 对话", "🔍 调试", "📊 统计"])
    
    with tab1:
        display_chat_interface()
    
    with tab2:
        display_debug_panel()
    
    with tab3:
        display_statistics()
    
    # 页脚信息
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "智能家居多轮对话管理引擎 v0.1.0 | "
        "Powered by DeepSeek & SiliconFlow"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()