"""
SmartEmailAgent — 多 Agent 架构入口

支持两种使用模式：
1. 直接使用专业 Agent（轻量、单步调用）
2. 使用 OrchestratorAgent 编排（多 Agent 协作、自动路由）

使用示例:
    # 模式1: 单独使用 Agent
    python main.py --mode single --email "test@example.com"

    # 模式2: Orchestrator 编排
    python main.py --mode orchestrate --goal "处理今天的收件箱"
"""

import argparse
import json
import logging
import os
import sys

# 清理代理设置（内网环境兼容）
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def demo_standalone_agents():
    """演示：单独使用各专业 Agent"""
    from rag_modules.agents import (
        SummarizerAgent,
        ClassifierAgent,
        TaskExtractorAgent,
        ReplyAgent,
    )

    # 模拟一封邮件
    sample_email = {
        "from": "张三 <zhangsan@company.com>",
        "sender": "张三 <zhangsan@company.com>",
        "date": "2026-06-04 14:30",
        "subject": "【紧急】Q3 项目架构评审会议通知",
        "body": (
            "您好，\n\n"
            "定于本周五（6月6日）下午2点在大会议室召开 Q3 项目架构评审会，"
            "请您准备以下材料：\n"
            "1. 当前系统架构文档\n"
            "2. 技术选型对比分析\n"
            "3. 性能压测报告\n\n"
            "会议由王总主持，预计时长2小时。请务必准时参加。\n\n"
            "会议链接：https://meeting.company.com/room-301\n\n"
            "祝好，\n张三"
        ),
        "attachments": [],
    }

    print("=" * 60)
    print("📧 原始邮件")
    print("=" * 60)
    print(f"发件人: {sample_email['from']}")
    print(f"主题:   {sample_email['subject']}")
    print(f"正文:   {sample_email['body'][:100]}...")
    print()

    # Agent 1: 摘要
    print("-" * 40)
    print("🤖 SummarizerAgent 输出:")
    summarizer = SummarizerAgent()
    result = summarizer.run(sample_email)
    print(f"  状态: {result.success}")
    print(f"  摘要: {result.data}")
    print()

    # Agent 2: 分类
    print("-" * 40)
    print("🤖 ClassifierAgent 输出:")
    classifier = ClassifierAgent()
    result = classifier.run(sample_email)
    print(f"  状态: {result.success}")
    print(f"  优先级: {result.data.get('priority')}")
    print(f"  原因:   {result.data.get('reason')}")
    print()

    # Agent 3: 任务提取
    print("-" * 40)
    print("🤖 TaskExtractorAgent 输出:")
    extractor = TaskExtractorAgent()
    result = extractor.run(sample_email)
    print(f"  状态: {result.success}")
    print(f"  任务数: {len(result.data.get('tasks', []))}")
    print(f"  会议数: {len(result.data.get('meetings', []))}")
    if result.data.get("tasks"):
        for task in result.data["tasks"]:
            print(f"    - {task}")
    print()


def demo_orchestrator():
    """演示：使用 OrchestratorAgent 编排多 Agent"""
    from rag_modules.agents import OrchestratorAgent

    sample_email = {
        "from": "李四 <lisi@partner.com>",
        "sender": "李四 <lisi@partner.com>",
        "date": "2026-06-04 09:15",
        "subject": "合作提案：AI 邮件助手集成方案",
        "body": (
            "您好，\n\n"
            "我们是 XYZ 科技，对贵公司的 AI 邮件助手产品非常感兴趣。"
            "希望能与贵公司合作，将我们的 NLP 能力集成到产品中。\n\n"
            "请在下周三（6月11日）前回复合作意向。\n"
            "附件是我们的产品介绍和技术白皮书。\n\n"
            "期待您的回复！\n李四\nXYZ 科技 CEO"
        ),
        "attachments": [],
    }

    print("=" * 60)
    print("🎯 OrchestratorAgent 编排演示")
    print("=" * 60)
    print(f"目标邮件: {sample_email['subject']}")
    print()

    orchestrator = OrchestratorAgent()
    result = orchestrator.run({
        "goal": "处理这封合作提案邮件：总结、分类、提取待办、如果需要则起草回复",
        "from": sample_email["from"],
        "sender": sample_email["sender"],
        "date": sample_email["date"],
        "subject": sample_email["subject"],
        "body": sample_email["body"],
        "attachments": sample_email["attachments"],
    })

    print(f"编排结果: {'✅ 成功' if result.success else '❌ 失败'}")
    if result.success:
        data = result.data
        if isinstance(data, dict):
            for agent_name, agent_output in data.items():
                print(f"\n  [{agent_name}]:")
                if isinstance(agent_output, dict):
                    for k, v in agent_output.items():
                        v_str = str(v)
                        if len(v_str) > 120:
                            v_str = v_str[:120] + "..."
                        print(f"    {k}: {v_str}")
                else:
                    out_str = str(agent_output)
                    if len(out_str) > 200:
                        out_str = out_str[:200] + "..."
                    print(f"    {out_str}")
    print()


def demo_message_bus():
    """演示：MessageBus Agent 间通信"""
    from rag_modules.bus import MessageBus, MessageRole

    print("=" * 60)
    print("📨 MessageBus 通信演示")
    print("=" * 60)

    bus = MessageBus()
    session = bus.create_session("demo_session")

    # 模拟 Agent 间消息传递
    bus.send("OrchestratorAgent", "SummarizerAgent",
             "请总结邮件 #001", MessageRole.ORCHESTRATOR, session)
    bus.send("SummarizerAgent", "OrchestratorAgent",
             "邮件总结：Q3架构评审会议通知...", MessageRole.AGENT, session)
    bus.send("OrchestratorAgent", "ClassifierAgent",
             "请分类邮件 #001", MessageRole.ORCHESTRATOR, session)
    bus.send("ClassifierAgent", "OrchestratorAgent",
             '{"priority": "High", "reason": "紧急会议通知"}',
             MessageRole.AGENT, session)

    print(f"会话 ID: {session.session_id}")
    print(f"消息数: {len(session.history)}")
    for msg in session.history:
        print(f"  [{msg.role.value}] {msg.sender} → {msg.receiver}: "
              f"{str(msg.content)[:80]}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="SmartEmailAgent — 多 Agent 邮件助手"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "orchestrate", "bus", "all"],
        default="all",
        help="运行模式: single=独立Agent, orchestrate=编排器, bus=消息总线, all=全部演示",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="",
        help="编排器目标（仅 orchestrate 模式）",
    )
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║       SmartEmailAgent — 多 Agent 架构             ║")
    print("║       Agent数: 5 专业 + 1 编排器                   ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    if args.mode in ("single", "all"):
        try:
            demo_standalone_agents()
        except Exception as e:
            logger.warning(f"独立 Agent 演示需配置 LLM API: {e}")

    if args.mode in ("orchestrate", "all"):
        try:
            demo_orchestrator()
        except Exception as e:
            logger.warning(f"编排器演示需配置 LLM API: {e}")

    if args.mode in ("bus", "all"):
        demo_message_bus()

    print("=" * 60)
    print("✅ 演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
