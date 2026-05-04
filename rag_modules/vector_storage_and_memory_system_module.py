import logging

from pymilvus import FieldSchema, DataType, CollectionSchema

from rag_modules import EmailAccessAndSynchronizationModule
from rag_modules import MilvusConnectionModule
from rag_modules import AiAnalysisCoreModule


def _create_collection(milvus_connection, email_collection):
    """创建milvus的邮箱数据集合"""

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
        FieldSchema(name="threadId", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="snippet", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="from_address", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="body", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="attachments", dtype=DataType.JSON),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
    ]
    schema = CollectionSchema(
        fields=fields,
        description="存储邮件向量的集合",
        enable_dynamic_field=True
    )
    collection = milvus_connection(name=email_collection, schema=schema)
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {
            "M": 8,
            "efConstruction": 64
        }
    }
    collection.create_index(field_name="embedding", index_params=index_params)

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
            logging.error(f"授权失败: {e}")
        max_emails = 5
        messages = email_access_and_synchronization_module.fetch_new_emails(max_results=max_emails)
        for index, msg in enumerate(messages, start=1):
            msg_id = msg['id']
            parsed_data = email_access_and_synchronization_module.parse_email(msg_id)
            email_data = {
                'id': msg_id,
                'threadId': msg.get('threadId'),
                'snippet': msg.get('snippet'),

                'subject': parsed_data.get('subject', ''),
                'from': parsed_data.get('from', ''),
                'date': parsed_data.get('date', ''),
                'body': parsed_data.get('body', ''),
                'attachments': parsed_data.get('attachments', []),

                'embedding': None,
            }
            email_data.update(
                parsed_data.get('embedding', AiAnalysisCoreModule.summarizer_agent())
            )
            milvus_connection.insert_one(email_data)