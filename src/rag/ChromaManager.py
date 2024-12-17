# AutoDL官方学术资源加速
import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

# 开始做ChromaDB的向量存储
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv # type: ignore
from langchain_openai import OpenAIEmbeddings # type: ignore
from langchain_chroma import Chroma # type: ignore
from langchain.document_loaders import PyPDFLoader # type: ignore 
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain.docstore.document import Document # type: ignore
from uuid import uuid4
from chromadb.utils import embedding_functions # type: ignore
import chromadb # type: ignore

# 这个类是一个适配器，将 LangChain 的 embedding 函数转换为 ChromaDB 可用的格式：
class LangChainEmbeddingFunction:
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings
    
    def __call__(self, input: List[str]) -> List[List[float]]: # __call__: 实现向量化功能，接收文本列表，返回向量列表
        """将输入文本转换为向量表示
        
        Args:
            input (List[str]): 输入文本列表
            
        Returns:
            List[List[float]]: 文本向量表示列表
        """
        return self.langchain_embeddings.embed_documents(input)

class ChromaVectorStoreManager:
    def __init__(self, collection_name: str = "default", persist_directory: str = None) -> None:

        # 加载环境变量
        load_dotenv()
        api_key = os.getenv('ZETATECHS_API_KEY')
        base_url = os.getenv('ZETATECHS_API_BASE')

        # 初始化embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key, base_url=base_url)

        # 如果没有提供persist_directory，则使用默认路径
        if persist_directory is None:
            current_directory = os.getcwd()
            persist_directory = os.path.join(current_directory, '..', 'ChromaVDB')

        # 初始化Chroma向量存储
        self.vector_store = Chroma( # 这里的Chroma是LangChain的Chroma，不是chromadb的Collection
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

        # 创建 ChromaDB embedding 函数
        self.chromadb_ef = LangChainEmbeddingFunction(self.embeddings)
        
        # 初始化 ChromaDB client 和 collection
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.chromadb_ef
        )

    def upload_pdf_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        """上传单个PDF文件并处理
        
        Args:
            file_path (str): PDF文件路径
            chunk_size (int): 文本分块大小
            chunk_overlap (int): 分块重叠大小
        """
        # 获取文件名作为metadata
        file_name = os.path.basename(file_path)
        file_type = "pdf"
        
        # 加载和分割文档
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # 添加文件相关metadata
        for doc in documents:
            doc.metadata.update({
                "file_name": file_name,
                "file_type": file_type,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "page_number": doc.metadata.get("page"),  # PDF页码
                "total_pages": len(documents)  # 总页数
            })
        
        # 分割文档
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents_chunks = splitter.split_documents(documents)
        
        # 生成唯一ID
        uuids = [str(uuid4()) for _ in range(len(documents_chunks))]
        
        # 准备数据
        texts = [doc.page_content for doc in documents_chunks]
        metadatas = [doc.metadata for doc in documents_chunks]
        
        # 同时添加到两个存储
        self.vector_store.add_documents(documents=documents_chunks, ids=uuids)
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=uuids
        )

    def upload_text_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        """上传单个文本文件（txt, markdown等）并处理"""
        file_name = os.path.basename(file_path)
        file_type = file_path.split('.')[-1].lower()
        
        # 读取文本文件
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 创建Document对象
        doc = Document(
            page_content=text,
            metadata={
                "file_name": file_name,
                "file_type": file_type,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
        )
        
        # 分割文档
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents_chunks = splitter.split_documents([doc])
        
        # 生成唯一ID
        uuids = [str(uuid4()) for _ in range(len(documents_chunks))]
        
        # 准备数据
        texts = [doc.page_content for doc in documents_chunks]
        metadatas = [doc.metadata for doc in documents_chunks]
        
        # 同时添加到两个存储
        self.vector_store.add_documents(documents=documents_chunks, ids=uuids)
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=uuids
        )

    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        results = self.vector_store.similarity_search(query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in results
        ]

    def similarity_search_with_score(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]
    
    def hybrid_search(self, 
                     query: str, 
                     content_keyword: str = None,
                     file_filter: Dict = None,
                     k: int = 3) -> List[Dict[str, Any]]:
        """执行混合搜索
        
        Args:
            query (str): 向量搜索的查询文本
            content_keyword (str, optional): 文档内容中要搜索的关键词
            file_filter (Dict, optional): 文件相关的过滤条件，如:
                {"file_name": {"$eq": "某文件.pdf"}}
                {"file_type": {"$eq": "pdf"}}
                {"page_number": {"$gte": 5}}
            k (int): 返回结果数量
            
        Example:
            results = chroma_store.hybrid_search(
                query="深度强化学习方法",
                content_keyword="策略梯度",
                file_filter={"file_type": {"$eq": "pdf"}},
                k=3
            )
        """
        where_document = {"$contains": content_keyword} if content_keyword else None
        
        results = self.collection.query(
            query_texts=[query],
            where_document=where_document,
            where=file_filter,
            n_results=k
        )
        
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                "distance": results['distances'][0][i] if results['distances'] else None
            })
        
        return formatted_results

    def add_documents(self, documents: List[Document], ids: List[str] = None) -> None:
        """添加文档到向量存储"""
        # 添加到 LangChain Chroma
        self.vector_store.add_documents(documents=documents, ids=ids)
        
        # 同时添加到 ChromaDB collection
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(documents))]
            
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

    def get_vector_store(self):
        """返回vector_store以便在chain中使用"""
        return self.vector_store