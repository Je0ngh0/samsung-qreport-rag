import time
from typing import List, Union, Optional

from pydantic import BaseModel


class ChatCompletionRequestMessage(BaseModel):
    content: str
    role: str
    name: str = None


class CreateChatCompletionRequest(BaseModel):
    messages: List[ChatCompletionRequestMessage]
    model: Union[str, List[str]] = "gpt-4o"
    temperature: Optional[float] = 1
    stream: Optional[bool] = False


class MessageChoice(BaseModel):
    message: dict
    finish_reason: str = "stop"
    index: int = 0


class DeltaChoice(BaseModel):
    delta: dict
    finish_reason: Optional[str] = None
    index: int = 0


class CompletionUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class CreateChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[MessageChoice]
    usage: Optional[CompletionUsage]

    @staticmethod
    def sample(content: str):
        return CreateChatCompletionResponse(
            id="chatcmpl-abc123",
            object="chat.completion",
            created=int(time.time()),
            model="gpt-4o",
            choices=[
                MessageChoice(
                    message={"role": "assistant", "content": content},
                    finish_reason="stop",
                    index=0,
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=10, total_tokens=20
            ),
        )


class CreateChatCompletionStreamingResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[DeltaChoice]

    @staticmethod
    def sample(content: str, finish_reason: Optional[str] = None):
        return CreateChatCompletionStreamingResponse(
            id="chatcmpl-abc123",
            object="chat.completion",
            created=int(time.time()),
            model="gpt-4o",
            choices=[
                DeltaChoice(
                    delta={"content": content},
                    finish_reason=finish_reason,
                    index=0,
                )
            ],
        )


class ModelData(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str

class ListModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelData]