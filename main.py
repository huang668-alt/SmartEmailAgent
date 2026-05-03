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
        print()
    except Exception as e:
        logger.error(f"系统运行失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()