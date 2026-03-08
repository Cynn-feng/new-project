from app.models import ModelRegistry


class FallbackPolicy:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def build_chain(self, primary_alias: str) -> list[str]:
        # fallback 链按注册表逐级展开，同时避免环形配置导致死循环。
        chain: list[str] = []
        visited = {primary_alias}
        current_alias = primary_alias

        while self.registry.has(current_alias):
            profile = self.registry.get(current_alias)
            next_alias = profile.fallback_model_alias
            if not next_alias or next_alias in visited:
                break
            chain.append(next_alias)
            visited.add(next_alias)
            current_alias = next_alias

        # 最后兜底追加本地 mock fallback，保证平台在离线环境也能返回结果。
        if self.registry.has("local_fallback") and "local_fallback" not in visited:
            chain.append("local_fallback")

        return chain
