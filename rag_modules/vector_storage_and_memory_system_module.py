import logging

from rag_modules import EmailAccessAndSynchronizationModule
from rag_modules import MilvusConnectionModule


def _create_collection(milvus_connection, email_collection):

    pass


class VectorStorageAndMemorySystemModule:

    def __init__(self):
        pass

    def store_historical_emails(self):
        milvus_connection_module = MilvusConnectionModule()
        milvus_connection = milvus_connection_module.connection()
        email_collection = "email_collection"
        if milvus_connection.has_collection(email_collection):
            pass
        else:
            _create_collection(milvus_connection, email_collection)

        email_access_and_synchronization_module = EmailAccessAndSynchronizationModule()
        try:
            email_access_and_synchronization_module.authenticate()
        except Exception as e:
            logging.error(f"❌ 授权失败: {e}")
        max_emails = 5
        messages = email_access_and_synchronization_module.fetch_new_emails(max_results=max_emails)
        for index, msg in enumerate(messages, start=1):
            msg_id = msg['id']
            parsed_data = email_access_and_synchronization_module.parse_email(msg_id)
