import logging

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from pymilvus import FieldSchema, DataType, CollectionSchema

from config import SmartEmailAgentConfig
from rag_modules import EmailAccessAndSynchronizationModule
from rag_modules import MilvusConnectionModule
from rag_modules import AiAnalysisCoreModule

def _embedding_original(text_data: str) -> list[float]:
    try:
        embeddings_model = OpenAIEmbeddings(
            model=SmartEmailAgentConfig.embeddings_model_name,
            dimensions=SmartEmailAgentConfig.embeddings_dimension,
            base_url=SmartEmailAgentConfig.embeddings_url,
            api_key=SecretStr(SmartEmailAgentConfig.embeddings_api_key)
        )
        vector = embeddings_model.embed_query(text_data)
        return vector

    except Exception as e:
        logging.error(f"数据向量化(Embedding)失败: {e}")
        return []

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
        FieldSchema(name="priority", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="reason", dtype=DataType.VARCHAR, max_length=256),
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
        self.ai_analyzer = None
        self.milvus_connection = MilvusConnectionModule().connection()

    def store_historical_emails(self):
        self.ai_analyzer = AiAnalysisCoreModule()
        email_collection = "email_collection"
        if self.milvus_connection.has_collection(email_collection):
            pass
        else:
            _create_collection(self.milvus_connection, email_collection)
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
                'reason' : None,
                'priority' : None,
                'embedding' : None,
            }
            embedding_original = AiAnalysisCoreModule.summarizer_agent(self.ai_analyzer, email_data)
            embedding_final = _embedding_original(embedding_original)
            email_data['embedding'] = embedding_final
            class_and_reason = AiAnalysisCoreModule.classifier_agent(self.ai_analyzer, email_data)
            email_data['priority'] = class_and_reason.get("priority", "Low")
            email_data['reason'] = class_and_reason.get("reason", "未提供理由")
            self.milvus_connection.insert_one(email_data)