# 设置日志
import logging
import traceback

from rag_modules import EmailAccessAndSynchronizationModule

import os
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    try:
        connector = EmailAccessAndSynchronizationModule()
        connector.authenticate()
        messages = connector.fetch_new_emails(max_results=15)
        for msg in messages:
            email = connector.parse_email(msg['id'])
            if email:
                print(f"【{email['subject']}】")
                print(f"发件人: {email['from']}")
                print(f"时间: {email.get('date', '未知')}")
                print("-" * 80)
    except Exception as e:
        logger.error(f"系统运行失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()