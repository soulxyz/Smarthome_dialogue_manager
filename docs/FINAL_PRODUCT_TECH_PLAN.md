我将创建一份面向产品经理和技术人员的最终方案文档，并生成对应的执行TODO清单，便于团队落地。随后我会把文档保存到 docs/FINAL_PRODUCT_TECH_PLAN.md。

[1 tool called]

### 最终方案（面向产品与技术的统一执行文档）

本方案整合“对话系统优化方案（plan）”“改进指南（guide）”与当前代码现状，目标是：以最小风险、快速见效的方式，建立一个可持续进化、可产品化运营的对话系统。

---

## 1. 目标与衡量

- 产品目标
  - 提升多轮对话可用性：能处理省略、指代、语序变化、口语化表达
  - 降低交互阻碍：低置信度时优先澄清，极低置信度时使用LLM回退
  - 支持持续演进：低成本接入新设备/场景/表达，不破坏既有系统

- 技术目标
  - 去除硬编码：设备/动作/场景词典与模式经由 device_manager/配置层统一输出
  - 策略可插拔：省略消解/焦点切换/结果融合抽象为策略接口
  - 观测与护栏：结构化日志、P95延迟、LLM触发率、预算控制

- 指标与阈值
  - P0：省略主语处理成功率 > 85%，焦点切换准确性 > 90%，稳定性 > 99.5%
  - P1：LLM回退触发率 < 20%，有效纠正率 > 70%，回退场景 P95 ≤ 2.5s
  - CI：语义覆盖报警不阻塞构建；错误率（非用户输入导致）≤ 1%

---

## 2. 分阶段路线（混合战术+架构演进）

### M1（1-2周）：核心可用性 + 架构净化
- 面向产品
  - 立即提升体验：支持“关掉/调亮一点/它/那个”等省略与指代表达
  - 保持响应速度：无网络/LLM时系统仍强可用
- 面向技术
  - P0功能落地（规则策略）
    - 省略消解：基于 current_focus 填补缺失设备实体
    - 智能焦点切换：保守策略（query不切换、相同实体不变）
    - 已有实现需梳理：`_extract_entities_with_context`、pronoun处理
  - 去硬编码（并行进行，关键）
    - 词典与模式来源统一：`device_manager` 输出
      - `get_device_synonyms() / get_action_synonyms() / get_scene_definitions()`
      - `get_available_device_types() / get_available_rooms() / get_device_patterns()`
    - `intent/clarification` 只消费数据，不维护设备/场景硬编码
  - 语义覆盖度测试（报警不阻塞）
    - 保持 `tests/test_semantic_coverage_cases.py`；完善样本
    - 将覆盖率、类别错误作为告警指标汇报

### M2（2-4周）：智能回退 + 策略预埋 + 场景编排
- 面向产品
  - 降低用户疑惑：中置信度先澄清，低置信度才用LLM
  - 场景行为更一致：场景触发是“能力编排”，不是“关键字堆砌”
- 面向技术
  - 双阈值决策
    - clarify_threshold（如0.7）与 llm_threshold（如0.4）分离
    - 不同意图可配置权重（未来→A/B学习）
  - LLM回退（灰度）
    - 超时（如3秒）、预算阈值、触发率监控、结构化日志
    - 回退成功采样入库，为后续规则/ML升级提供样本
  - 场景编排迁移到 device_manager
    - `get_scene_definitions()`：场景→设备动作编排、同义触发词
    - `execute_scene(scene_id, params)`：由 device_manager 执行，intent 只做路由
  - 策略接口预埋
    - `OmissionResolutionStrategy` / `FocusSwitchStrategy` / `FusionStrategy`
    - 当前实现基于规则，后续可替换 ML 或混合策略

### M3（4-8周）：A/B与学习 + 可观测性完善
- 面向产品
  - 策略升级不打扰用户，实验性上线统一管理
- 面向技术
  - A/B与策略权重学习（离线→灰度）
  - 完整可观测性
    - 结构化日志：意图、置信度、焦点、澄清/回退命中、API耗时、token
    - 监控指标：P95延迟、回退命中率、有效纠正率、错误率
    - 护栏：超预算自动降级、异常自动禁用回退

---

## 3. 架构原则（提交给工程团队）

- 单一事实来源（Single Source of Truth）
  - 设备/动作/场景/房间/同义词/模式全部由 `device_manager` 与配置层提供
  - `intent.py` 和 `clarification.py` 不再书写硬编码词典
- 策略可插拔（Strategy Interfaces）
  - Omission、Focus、Fusion 等策略抽象；当前规则实现，未来ML实现可替换
- 能力边界清晰
  - 场景是“设备编排能力”，意图层只做“识别与路由”，不做“行为编排”
- 观测与护栏内置
  - 结构化日志、预算、超时、灰度开关，是功能上线的必要条件

---

## 4. 关键接口（供实现参考）

- device_manager（新增/强化）
  - `get_device_synonyms() -> Dict[str, List[str]]`
  - `get_action_synonyms() -> Dict[str, List[str]]`
  - `get_scene_definitions() -> Dict[str, Any]  # 场景-设备编排与触发表达`
  - `get_available_device_types() -> List[str]`
  - `get_available_rooms() -> List[str]`
  - `get_device_patterns() -> Dict[str, str]`
  - `execute_scene(scene_id: str, params: Dict) -> Dict`

- intent
  - `update_dynamic_patterns()`：仅组合 device_manager 数据→生成动态模式
  - `OmissionResolutionStrategy.resolve(user_input, context) -> List[Entity]`
  - `FocusSwitchStrategy.should_switch(new_entity, current_focus, intent) -> bool`
  - `FusionStrategy.fuse(results, context) -> Dict`

- clarification
  - Prompt 注入：设备/房间清单、焦点设备、最近操作摘要
  - 候选合成：基于 device_manager 返回的词典/场景定义构造规范化候选

- engine
  - 双阈值：clarify_threshold 与 llm_threshold
  - 灰度、超时、预算、结构化日志、触发率监控

---

## 5. 验收标准（每阶段）

- M1（P0）
  - 省略主语识别成功率 > 85%，焦点切换准确性 > 90%
  - 语义覆盖告警可见，无阻塞
  - intent/clarification 无设备/场景硬编码残留
- M2（P1）
  - LLM 触发率 < 20%，有效纠正率 > 70%
  - 回退场景 P95 ≤ 2.5s；超时与预算命中率日志可查
  - 场景编排从 device_manager 成功执行
- M3
  - AB试验框架可路由10%-50%流量
  - 结构化日志字段齐全；监控面板可视化
  - 异常自动降级可验证

---

## 6. 执行TODO清单（对齐里程碑）

- M1
  - 落地P0：省略消解与智能焦点切换（规则策略）
  - 去硬编码：将设备/动作/场景词典迁移到 device_manager/配置层
  - 完善语义覆盖度测试并接入CI为报警不阻塞

- M2
  - 实现双阈值决策：clarify 与 LLM 回退阈值分离
  - 灰度上线 LLM 回退，加入超时、预算与结构化日志
  - 将场景控制迁移为 device_manager 编排能力
  - 预埋策略接口：Omission/Focus/Fusion 策略可插拔

- M3
  - 建立 A/B 测试与策略权重学习（离线+灰度）
  - 完善可观测性：关键字段结构化日志与性能监控阈值

---

## 7. 风险与对策

- 规则继续膨胀
  - 统一用“词典+模板”自动生成模式，限制手写正则
- LLM成本/时延
  - 双阈值+灰度+预算/超时；高频澄清缓存
- 场景误触发
  - 场景独立为编排能力，识别层路由+用户确认
- 回归风险
  - 黄金样本集+CI基线+语义覆盖报警

---

## 8. 附：结构化日志字段建议

- session_id, turn_id, user_input, intent, confidence
- entities, current_focus, need_clarification, llm_used
- api_calls_count, tokens, latency_ms, budget_hit
- errors: type, message, stage

---

这份方案可以直接落地执行。建议把本文保存到 `docs/FINAL_PRODUCT_TECH_PLAN.md` 并在周会中对齐里程碑与验收标准，工程侧按“执行TODO清单”推进；产品侧同步观察“语义覆盖报警面板”和“用户满意度反馈”。