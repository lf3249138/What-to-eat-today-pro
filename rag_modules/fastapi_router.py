"""
Web服务处理模块
负责处理Web API和静态文件服务
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from .web_fastapi_handler import get_handler
from datetime import datetime


class QueryData(BaseModel):
    message: str = Field(..., min_length=1, description="问题")
    session_id: str = Field(..., min_length=1, description="会话Id")

router = APIRouter(tags=["api"])


@router.get('/')
def serve_index():
    """提供主页"""
    return get_handler().serve_static_file('index.html')

@router.get('/{filename}')
def serve_static(filename: str):
    """提供静态文件服务"""
    return get_handler().serve_static_file(filename)

@router.get('/health')
def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": str(datetime.now()),
        "service": "RAG System"
    }

@router.post('/api/chat')
def chat(data: QueryData):
    """聊天API - 普通响应"""
    return get_handler().handle_chat_request(data.message, data.session_id)

@router.post('/api/chat/stream')
def chat_stream(data: QueryData):
    """聊天API - 流式响应"""
    return get_handler().handle_stream_request(data.message, data.session_id)

@router.post('/api/recipes/recommendations')
def get_recommendations():
    """获取菜谱推荐"""
    return get_handler().handle_recommendations_request()

@router.get('/api/recipes/{recipe_id}')
def get_recipe_detail(recipe_id: str):
    """获取菜谱详情"""
    return get_handler().handle_recipe_detail_request(recipe_id)

@router.get('/api/stats')
def get_stats():
    """获取系统统计信息"""
    return get_handler().handle_stats_request()