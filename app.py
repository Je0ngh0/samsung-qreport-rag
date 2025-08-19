import json
import os
# import sys
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

# 필요한 경우 아래 주석 해제
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openai_model import (
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamingResponse,
    ListModelsResponse,
    ModelData
)

# ==== ⬇️ LangChain 추가 (최소) =================================================
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
# =============================================================================


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 특정 origin만 허용: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # 모든 메서드 허용 (GET, POST, OPTIONS 등)
    allow_headers=["*"],  # 모든 헤더 허용
)

# ==== ⬇️ LangChain 체인 정의 (원하시면 retriever 연결만 하세요) ===============
PROMPT = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Question: 
{question} 
#Context: 
{context} 

#Answer:"""
)

from openai import OpenAI
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
models = client.models.list()
model = models.data[0].id

# 내부 LLM: Gemini (Google GenAI). GOOGLE_API_KEY 필요
LLM = ChatOpenAI(
    model=model,
    temperature=0.1,
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    )

# 출력만 문자열로 만들어주는 간단 체인 (프롬프트는 아래에서 문자열로 미리 만들어 넣습니다)
CHAIN = LLM | StrOutputParser()

# https://python.langchain.com/docs/how_to/custom_embeddings/

from typing import List, Iterable, Union

from langchain_core.embeddings import Embeddings

import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

def _last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class MyEmbeddings(Embeddings):
    def __init__(
        self,
        model: str = "Qwen/Qwen3-Embedding-8B",
        *,
        max_length: int = 8192,
        device = None
    ):
        self.model_name = model
        self.max_length = max_length

        self._tokenizer = tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')

        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        self._model = AutoModel.from_pretrained(self.model_name,quantization_config=bnb_config,).eval()

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = torch.device(device)
        self._model.to(self._device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_texts([text])[0]

    @torch.inference_mode()
    def _embed_texts(self, texts: Union[List[str], Iterable[str]]) -> List[List[float]]:
        batch_dict = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_dict.to(self._device)
        outputs = self._model(**batch_dict)
        

        embeddings = _last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        out = embeddings.tolist()
        return out
from langchain_chroma import Chroma

embeddings = MyEmbeddings()

DB_PATH = "./data/embedded/chroma_db"

persist_db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings,
    collection_name="samsung_quarterly_report",
)
retriever = persist_db.as_retriever()
CONTEXT_CHAR_BUDGET = int(os.getenv("CONTEXT_CHAR_BUDGET", "8000"))
def _format_docs(docs) -> str:
    """
    검색된 Document 리스트를 하나의 문자열 컨텍스트로 변환.
    - 소스/페이지 같은 메타데이터가 있으면 헤더에 붙임
    - 너무 길면 잘라냄(문자 기준)
    """
    parts = []
    for i, d in enumerate(docs, 1):
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source") or meta.get("file_path") or meta.get("id") or "source"
        page = meta.get("page")
        header = f"[{i}] {src}" + (f" p.{page}" if page is not None else "")
        parts.append(f"{header}\n{d.page_content}")

    text = "\n\n---\n\n".join(parts)
    # 너무 길면 잘라서 토큰 초과 방지(간단히 문자 수 기준)
    if len(text) > CONTEXT_CHAR_BUDGET:
        text = text[:CONTEXT_CHAR_BUDGET]
    return text
def get_context(question: str) -> str:
    """
    질문을 받아 retriever로 관련 문서 검색 후 문자열 컨텍스트로 반환
    (동기 버전: server의 event_stream과 동일하게 sync로 동작)
    """
    try:
        docs = retriever.invoke(question) # sync
        return _format_docs(docs)
    except Exception as e:
        # 필요 시 로깅 추가 가능
        return ""
# =============================================================================


def event_stream(prompt: str):
    """
    ✅ 원본 구조 유지.
    - 기존엔 google genai stream → 이제는 LangChain 체인 stream
    - 응답 포맷/엔드포인트/미디어타입 그대로
    """
    try:
        # 프롬프트 구성 (질문/컨텍스트 채워서 최종 문자열로)
        final_prompt = PROMPT.format(question=prompt, context=get_context(prompt))

        # LangChain 동기 스트리밍 (원본과 동일한 generator 스타일 유지)
        for token in CHAIN.stream(final_prompt):
            if token:
                sample = CreateChatCompletionStreamingResponse.sample(
                    content=token, finish_reason=None
                )
                yield f"data: {sample.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"


def send_response(request_body: CreateChatCompletionRequest):
    """
    ✅ 원본 구조 유지.
    - stream=True면 위의 event_stream 사용
    - 아니면 LangChain 체인 단발 호출
    """
    prompt = request_body.messages[-1].content
    if request_body.stream:
        return StreamingResponse(
            event_stream(prompt),
            media_type="application/x-ndjson"  # 원본 그대로 유지
        )
    else:
        final_prompt = PROMPT.format(question=prompt, context=get_context(prompt))
        text = CHAIN.invoke(final_prompt)   # 문자열 반환
        return CreateChatCompletionResponse.sample(text)


@app.post("/v1/chat/completions")
def create_chat_completion(request_body: CreateChatCompletionRequest):
    return send_response(request_body)


@app.get("/v1/models")
def list_models():
    created_time = int(time.time())
    models = [
        ModelData(id=model+"-rag", created=created_time, owned_by="organization-owner"),
    ]
    return ListModelsResponse(data=models)