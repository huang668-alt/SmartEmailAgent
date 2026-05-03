import logging

from pymilvus import MilvusClient

class MilvusConnectionModule:

    def __init__(self):
        self.service = None

    def connection(self):
        """创建milvus的连接"""
        try:
            self.service = MilvusClient(
                uri="http://localhost:19530",
                # token="root:Milvus"
            )
        except Exception as e:
            logging.error(e)

    def close(self):
        """关闭milvus的连接"""
        try:
            self.service.close()
        except Exception as e:
            logging.error(e)