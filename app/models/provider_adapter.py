from abc import ABC, abstractmethod

from app.schemas import LLMRequest, LLMResponse


class ProviderAdapter(ABC):
    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError
