"""Gmail 邮箱访问模块测试（使用拆分后的 email 模块）"""

import logging

from rag_modules.email import GmailAuth, EmailFetcher, EmailParser

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    print("=== 开始测试 Gmail 邮箱访问模块 ===")

    # 1. 认证
    print("\n[1/3] 正在进行 OAuth 授权认证...")
    try:
        service = GmailAuth().authenticate()
        print("✅ 授权认证成功！")
    except Exception as e:
        print(f"❌ 授权失败: {e}")
        print("💡 请检查当前目录下是否包含 'credentials.json' 文件。")
        exit(1)

    fetcher = EmailFetcher(service)
    parser = EmailParser(service)

    # 2. 拉取收件箱
    print("\n[2/3] 正在拉取最新邮件...")
    max_test_emails = 3
    messages = fetcher.fetch_inbox(max_results=max_test_emails)

    if not messages:
        print("📭 未找到任何邮件或拉取失败。")
        exit(0)

    print(f"✅ 成功拉取到 {len(messages)} 封邮件，开始解析...")

    # 3. 解析邮件
    print("\n[3/3] 邮件详情解析：")
    for index, msg in enumerate(messages, start=1):
        msg_id = msg["id"]
        print(f"\n--- 邮件 {index}/{len(messages)} (ID: {msg_id}) ---")

        parsed = parser.parse(msg_id)
        if parsed:
            print(f"📌 主题: {parsed['subject']}")
            print(f"👤 发件人: {parsed['from']}")
            print(f"📅 时间: {parsed['date']}")
            body = parsed["body"][:100].replace("\n", " ") + "..." if len(parsed["body"]) > 100 else parsed["body"]
            print(f"📝 正文预览: {body}")

            if parsed["attachments"]:
                print(f"📎 附件: {len(parsed['attachments'])} 个")
                for att in parsed["attachments"]:
                    print(f"    - {att['filename']} (→ {att['filepath']})")
            else:
                print("📎 附件: 无")
        else:
            print("❌ 解析失败")

    print("\n=== 测试结束 ===")
