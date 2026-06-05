"""Milvus Collection Schema 定义"""

import logging

from pymilvus import MilvusClient, DataType

logger = logging.getLogger(__name__)

VECTOR_DIM = 512
HNSW_PARAMS_LIGHT = {"M": 8, "efConstruction": 64}
HNSW_PARAMS_HEAVY = {"M": 16, "efConstruction": 256}


def _add_vector_index(client: MilvusClient, params: dict):
    """为 collection 添加 HNSW + COSINE 向量索引"""
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding", metric_type="COSINE",
        index_type="HNSW", index_name="vector_index", params=params,
    )
    return index_params


def create_inbox_collection(client: MilvusClient, name: str = "accept_email_collection"):
    """收件箱邮件集合"""
    if client.has_collection(name):
        return
    schema = MilvusClient.create_schema(
        auto_id=False, enable_dynamic_field=True, description="收件邮件向量集合",
    )
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
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

    client.create_collection(collection_name=name, schema=schema,
                             index_params=_add_vector_index(client, HNSW_PARAMS_LIGHT))
    logger.info(f"集合 {name} 创建成功")


def create_sent_collection(client: MilvusClient, name: str = "send_email_collection"):
    """发件箱邮件集合"""
    if client.has_collection(name):
        return
    schema = MilvusClient.create_schema(
        auto_id=False, enable_dynamic_field=True, description="发送邮件向量集合",
    )
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=256, is_primary=True)
    schema.add_field(field_name="threadId", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="snippet", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="subject", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="to_address", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="date", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="body", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="attachments", datatype=DataType.JSON)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

    client.create_collection(collection_name=name, schema=schema,
                             index_params=_add_vector_index(client, HNSW_PARAMS_LIGHT))
    logger.info(f"集合 {name} 创建成功")


def create_context_collection(client: MilvusClient, name: str = "context_historical"):
    """长期记忆集合"""
    if client.has_collection(name):
        return
    schema = MilvusClient.create_schema(
        auto_id=True, enable_dynamic_field=True, description="长期记忆与项目上下文",
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="timestamp", datatype=DataType.DOUBLE)

    client.create_collection(collection_name=name, schema=schema,
                             index_params=_add_vector_index(client, HNSW_PARAMS_HEAVY))
    logger.info(f"集合 {name} 创建成功")
