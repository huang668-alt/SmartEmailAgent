"""
SmartEmailAgent — AI 邮件助手入口

用法:
    # 交互式问答（默认）
    python main.py

    # 单次提问
    python main.py --question "Q3 项目进展如何？"

    # 同步收件箱
    python main.py --mode sync --max 20
"""

import argparse
import logging
import os
import sys

# Windows 终端 UTF-8 编码支持
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 清理代理设置（内网环境兼容）
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── 单次问答 ────────────────────────────────────────────────

def rag_query(question: str, top_k: int = 5):
    """单次 RAG 问答"""
    from rag_modules import Pipeline

    pipeline = Pipeline()

    print(f"\n🔍 问题: {question}")
    print(f"   检索范围: top_k={top_k}")
    print()

    result = pipeline.chat(user_input=question, top_k=top_k)

    if not result.success:
        print(f"❌ 查询失败: {result.error}")
        return

    data = result.data
    print(f"置信度: {data.get('confidence', 'N/A')}  |  引用来源: {data.get('cited_count', 0)} 条")
    print()
    print(data.get("answer", ""))
    print()

    sources = result.metadata.get("sources", {})
    emails = sources.get("emails", [])
    if emails:
        print("-" * 40)
        print(f"📧 相关邮件 ({len(emails)} 封):")
        for e in emails:
            print(f"  [{e['priority']}] {e['subject']}")
            print(f"       发件人: {e['from_address']} | {e['date']}")
            print(f"       相似度: {e['distance']}")
    print()


# ── 交互式问答 ──────────────────────────────────────────────

def interactive_query(top_k: int = 5):
    """交互式 RAG 问答 REPL"""
    from rag_modules import Pipeline

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║     SmartEmailAgent — 交互式问答                 ║")
    print("║     /sync <N>  同步最近 N 封邮件                 ║")
    print("║     /exit      退出                              ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    pipeline = Pipeline()
    history: list = []

    print("📥 正在同步收件箱...")
    try:
        pipeline.process_inbox(max_emails=top_k)
        print(f"✅ 已同步 {top_k} 封邮件\n")
    except Exception as e:
        print(f"⚠️ 同步失败（请检查 Gmail 认证和网络）: {e}\n")

    while True:
        try:
            question = input("💬 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not question:
            continue

        if question.lower() in ("/exit", "/quit", "/q"):
            print("👋 再见！")
            break

        if question.lower().startswith("/sync"):
            parts = question.split()
            n = int(parts[1]) if len(parts) > 1 else 10
            print(f"📥 正在同步 {n} 封邮件...")
            try:
                pipeline.process_inbox(max_emails=n)
                print("✅ 同步完成\n")
            except Exception as e:
                print(f"⚠️ 同步失败: {e}\n")
            continue

        print("🔍 正在检索...", end="", flush=True)
        result = pipeline.chat(user_input=question, top_k=top_k, history=history)

        if result.success:
            data = result.data
            print(f"\r{' ' * 20}\r", end="")  # 清除"正在检索..."
            print(data.get("answer", ""))
            print(f"\n(置信度: {data.get('confidence', 'N/A')}  |  来源: {data.get('cited_count', 0)} 条)")
            print("-" * 40)
            history.append({"question": question, "answer": data.get("answer", "")})
            history = history[-6:]
        else:
            print(f"\r❌ 查询失败: {result.error}")
            print("-" * 40)


# ── 同步收件箱 ──────────────────────────────────────────────

def sync_inbox(max_emails: int = 20):
    """同步收件箱邮件到向量库"""
    from rag_modules import Pipeline

    print(f"📥 正在同步最近 {max_emails} 封邮件...")
    pipeline = Pipeline()
    try:
        pipeline.process_inbox(max_emails=max_emails)
        print("✅ 同步完成")
    except Exception as e:
        print(f"❌ 同步失败: {e}")


# ── CLI ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SmartEmailAgent — AI 邮件助手"
    )
    parser.add_argument(
        "--mode",
        choices=["query", "sync"],
        default="query",
        help="运行模式: query=交互问答, sync=同步收件箱",
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default="",
        help="单次提问（留空进入交互模式）",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="向量搜索返回数量（默认 5）",
    )
    parser.add_argument(
        "--max", "-n",
        type=int,
        default=20,
        help="同步邮件数量（默认 20）",
    )
    args = parser.parse_args()

    if args.mode == "sync":
        sync_inbox(max_emails=args.max)
    elif args.question:
        rag_query(question=args.question, top_k=args.top_k)
    else:
        interactive_query(top_k=args.top_k)


if __name__ == "__main__":
    main()
