# (这里上方是您刚刚提供的 EmailAccessAndSynchronizationModule 类的代码)
import logging

from rag_modules import EmailAccessAndSynchronizationModule

if __name__ == "__main__":
    # 配置日志输出格式，方便在控制台看到报错或提示信息
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("=== 开始测试 Gmail 邮箱访问模块 ===")

    # 1. 实例化模块
    email_module = EmailAccessAndSynchronizationModule()

    # 2. 授权认证
    print("\n[1/3] 正在进行授权认证...")
    try:
        email_module.authenticate()
        print("✅ 授权认证成功！")
    except Exception as e:
        print(f"❌ 授权失败: {e}")
        print("💡 请检查当前目录下是否包含从 Google Cloud 下载的 'credentials.json' 文件。")
        exit(1)

    # 3. 拉取最新邮件列表 (这里设置 max_results=3 获取最新的3封邮件作为测试)
    print("\n[2/3] 正在拉取最新邮件...")
    max_test_emails = 3
    messages = email_module.fetch_new_emails(max_results=max_test_emails)

    if not messages:
        print("📭 未找到任何邮件或拉取失败。")
        exit(0)

    print(f"✅ 成功拉取到 {len(messages)} 封邮件，开始解析...")

    # 4. 遍历并解析每封邮件
    print("\n[3/3] 邮件详情解析：")
    for index, msg in enumerate(messages, start=1):
        msg_id = msg['id']
        print(f"\n--- 邮件 {index}/{len(messages)} (ID: {msg_id}) ---")

        parsed_data = email_module.parse_email(msg_id)
        if parsed_data:
            print(f"📌 主题 (Subject): {parsed_data['subject']}")
            print(f"👤 发件人 (From)  : {parsed_data['from']}")
            print(f"📅 时间 (Date)    : {parsed_data['date']}")

            # 正文可能很长，只打印前 100 个字符
            body_preview = parsed_data['body'][:100].replace('\n', ' ') + "..." if len(parsed_data['body']) > 100 else \
            parsed_data['body']
            print(f"📝 正文预览      : {body_preview}")

            # 打印附件信息
            if parsed_data['attachments']:
                print(f"📎 附件数量      : {len(parsed_data['attachments'])} 个")
                for att in parsed_data['attachments']:
                    print(f"    - {att['filename']} (已存至 {att['filepath']})")
            else:
                print("📎 附件数量      : 无")
        else:
            print("❌ 解析该封邮件失败")

    print("\n=== 测试结束 ===")