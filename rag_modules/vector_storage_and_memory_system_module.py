import json
import logging
import re
import time
from typing import Counter

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from pymilvus import MilvusClient

from pymilvus import MilvusClient, DataType

import config
from config import SmartEmailAgentConfig
from rag_modules import EmailAccessAndSynchronizationModule
from rag_modules import MilvusConnectionModule
from rag_modules import AiAnalysisCoreModule

def _create_collection_context_historical(client: MilvusClient, collection_name: str):
    """
    创建历史上下文集合并建立向量索引
    """
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
        description="存储用户长期记忆与项目上下文"
    )

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=512)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="timestamp", datatype=DataType.DOUBLE)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        metric_type="COSINE",
        index_type="HNSW",
        index_name="vector_index",
        params={"M": 16, "efConstruction": 256}
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

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

def _create_collection(client: MilvusClient, email_collection: str):
    """创建milvus的邮箱数据集合"""

    # 1. 创建 schema
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
        description="存储邮件向量的集合"
    )

    # 2. 依次添加字段
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=256, is_primary=True)
    schema.add_field(field_name="threadId", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="snippet", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="subject", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="from_address", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="date", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="body", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="attachments", datatype=DataType.JSON)
    schema.add_field(field_name="priority", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="reason", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=512)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        metric_type="COSINE",
        index_type="HNSW",
        index_name="vector_index",
        params={"M": 8, "efConstruction": 64}
    )
    client.create_collection(
        collection_name=email_collection,
        schema=schema,
        index_params=index_params
    )



def _create_collection_sent(client: MilvusClient, email_collection: str):
    """创建milvus的发送邮件数据集合"""

    # 1. 创建 schema
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
        description="存储发送的邮件向量的集合"
    )

    # 2. 依次添加字段
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=256, is_primary=True)
    schema.add_field(field_name="threadId", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="snippet", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="subject", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="to_address", datatype=DataType.VARCHAR, max_length=256)  # 发件箱主要存收件人
    schema.add_field(field_name="date", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="body", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="attachments", datatype=DataType.JSON)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=512)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        metric_type="COSINE",
        index_type="HNSW",
        index_name="vector_index",
        params={"M": 8, "efConstruction": 64}
    )

    client.create_collection(
        collection_name=email_collection,
        schema=schema,
        index_params=index_params
    )
    print(f"发送邮件集合 {email_collection} 创建成功！")


class VectorStorageAndMemorySystemModule:

    def __init__(self):
        self.ai_analyzer = None
        self.milvus_connection = MilvusConnectionModule().connection()

    def store_historical_emails(self):
        self.ai_analyzer = AiAnalysisCoreModule()
        email_collection = "accept_email_collection"
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
            response_str = AiAnalysisCoreModule.classifier_agent(self.ai_analyzer, email_data)
            try:
                class_and_reason = json.loads(response_str)
                email_data['priority'] = class_and_reason.get("priority", "Low")
                email_data['reason'] = class_and_reason.get("reason", "未提供理由")
            except Exception as e:
                email_data['priority'] = "Low"
                email_data['reason'] = "解析失败"
                logging.error(e)
            self.milvus_connection.insert(
                collection_name=email_collection,
                data=email_data
            )

    def email_reply_tone(self):
        email_collection = "send_email_collection"
        if self.milvus_connection.has_collection(email_collection):
            pass
        else:
            _create_collection_sent(self.milvus_connection, email_collection)
        email_access_and_synchronization_module = EmailAccessAndSynchronizationModule()
        try:
            email_access_and_synchronization_module.authenticate()
        except Exception as e:
            logging.error(f"授权失败: {e}")
        max_emails = 5
        messages = email_access_and_synchronization_module.fetch_new_emails_sent(max_results=max_emails)
        for index, msg in enumerate(messages, start=1):
            parsed_data = email_access_and_synchronization_module.parse_email(msg['id'])
            email_data = {
                'id': msg['id'],
                'threadId': msg.get('threadId'),
                'snippet': msg.get('snippet'),
                'subject': parsed_data.get('subject', ''),
                'to': parsed_data.get('to', ''),
                'date': parsed_data.get('date', ''),
                'body': parsed_data.get('body', ''),
                'attachments': parsed_data.get('attachments', []),
                'embedding': None,
            }
            embedding_original = AiAnalysisCoreModule.summarizer_agent(self.ai_analyzer, email_data)
            embedding_final = _embedding_original(embedding_original)
            email_data['embedding'] = embedding_final
            self.milvus_connection.insert(
                collection_name=email_collection,
                data=email_data
            )
    def historical_chat(self):
        email_access_and_synchronization_module = EmailAccessAndSynchronizationModule()
        try:
            email_access_and_synchronization_module.authenticate()
        except Exception as e:
            logging.error(f"授权失败: {e}")
        max_emails = 5
        messages_from = email_access_and_synchronization_module.fetch_new_emails(max_results=max_emails)
        messages_to = email_access_and_synchronization_module.fetch_new_emails_sent(max_results=max_emails)
        sender_list = []
        receiver_list = []
        for msg in messages_from:
            parsed_data = email_access_and_synchronization_module.parse_email(msg['id'])
            email_match = re.search(r'<([^>]+)>', parsed_data.get('from'))
            if parsed_data.get('from'):
                clean_email = email_match.group(1).lower()
            else:
                clean_email = parsed_data.get('from').strip().lower()
            sender_list.append(clean_email)
        sender_counts = Counter(sender_list)
        top_10_senders = sender_counts.most_common(config.SmartEmailAgentConfig.number_of_common_contacts)

        for msg in messages_to:
            parsed_data = email_access_and_synchronization_module.parse_email(msg['id'])
            email_match = re.search(r'<([^>]+)>', parsed_data.get('to'))
            if parsed_data.get('to'):
                clean_email = email_match.group(1).lower()
            else:
                clean_email = parsed_data.get('to').strip().lower()
            receiver_list.append(clean_email)
        receiver_list = Counter(sender_list)
        top_10_receiver = receiver_list.most_common(config.SmartEmailAgentConfig.number_of_common_contacts)

    def context_historical(self, context):
        email_collection = "context_historical"
        if self.milvus_connection.has_collection(email_collection):
            pass
        else:
            _create_collection_context_historical(self.milvus_connection, email_collection)
        embedding_context = _embedding_original(context)
        context_summary = AiAnalysisCoreModule.context_summary_agent(self.ai_analyzer, context)
        data_to_insert = [
            {
                "embedding": embedding_context,
                "content": context_summary,
                "timestamp": time.time()
            }
        ]
        self.milvus_connection.insert(
            collection_name=email_collection,
            data=data_to_insert
        )