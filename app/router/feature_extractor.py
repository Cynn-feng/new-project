import re

from app.schemas import TaskFeatures


class FeatureExtractor:
    _CODE_KEYWORDS = (
        "debug",
        "bug",
        "traceback",
        "exception",
        "stack trace",
        "python",
        "java",
        "javascript",
        "typescript",
        "sql",
        "function",
        "代码",
        "函数",
        "报错",
        "异常",
    )
    _RETRIEVAL_KEYWORDS = (
        "search",
        "find",
        "look up",
        "retrieve",
        "knowledge base",
        "document",
        "pdf",
        "file",
        "reference",
        "source",
        "检索",
        "搜索",
        "文档",
        "资料",
        "查一下",
    )
    _REASONING_KEYWORDS = (
        "analyze",
        "analysis",
        "compare",
        "why",
        "prove",
        "derive",
        "tradeoff",
        "reason",
        "step by step",
        "分析",
        "比较",
        "推理",
        "证明",
        "原因",
        "权衡",
    )
    _SUMMARIZATION_KEYWORDS = (
        "summarize",
        "summary",
        "tl;dr",
        "recap",
        "digest",
        "总结",
        "概括",
        "提炼",
        "摘要",
        "要点",
    )
    _JSON_KEYWORDS = (
        "json",
        "schema",
        "structured output",
        "structure output",
        "结构化",
        "字段",
    )
    _COMPARE_PATTERNS = (
        r"\bcompare\b",
        r"\bvs\b",
        r"\bversus\b",
        r"\bpros?\s+and\s+cons?\b",
        r"\btrade[- ]offs?\b",
        r"对比",
        r"比较",
    )
    _CODE_PATTERNS = (
        r"```",
        r"\b(def|class|import|from|return|async|await|try|except)\b",
        r"\b(function|const|let|var|console\.log|=>)\b",
        r"\b(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|JOIN)\b",
        r"Traceback \(most recent call last\):",
    )
    _CODE_LINE_PATTERNS = (
        r"^\s{0,8}(def |class |import |from |return |if |for |while |try:|except |with )",
        r"^\s{0,8}(function |const |let |var )",
        r"^\s{0,8}(SELECT |INSERT |UPDATE |DELETE )",
    )
    _REFERENCE_PATTERNS = (
        r"https?://",
        r"\b[\w\-/]+\.(pdf|docx?|pptx?|xlsx?|csv|md|txt|json)\b",
    )

    def extract(self, text: str) -> TaskFeatures:
        if not text or not text.strip():
            return TaskFeatures(input_length=len(text or ""))

        lowered = text.casefold()
        input_length = len(text)
        line_count = max(text.count("\n") + 1, 1)
        question_count = text.count("?") + text.count("？")

        code_keyword_hits = self._keyword_hits(lowered, self._CODE_KEYWORDS)
        retrieval_keyword_hits = self._keyword_hits(lowered, self._RETRIEVAL_KEYWORDS)
        reasoning_keyword_hits = self._keyword_hits(lowered, self._REASONING_KEYWORDS)
        summary_keyword_hits = self._keyword_hits(lowered, self._SUMMARIZATION_KEYWORDS)

        code_pattern_hits = self._pattern_hits(text, self._CODE_PATTERNS)
        code_line_hits = self._pattern_hits(text, self._CODE_LINE_PATTERNS, multiline=True)
        compare_hits = self._pattern_hits(lowered, self._COMPARE_PATTERNS)
        reference_hits = self._pattern_hits(text, self._REFERENCE_PATTERNS)

        needs_json = any(token in lowered for token in self._JSON_KEYWORDS) or bool(
            re.search(r"[{\[]\s*\"?[A-Za-z_][\w-]*\"?\s*:", text)
        )

        code_marker_count = code_keyword_hits + code_pattern_hits + code_line_hits
        retrieval_marker_count = retrieval_keyword_hits + reference_hits
        reasoning_marker_count = reasoning_keyword_hits + compare_hits
        summarization_marker_count = summary_keyword_hits

        code_signal = min(
            self._normalized(code_pattern_hits + code_line_hits, 3) * 0.45
            + self._normalized(code_keyword_hits, 3) * 0.3
            + (0.25 if code_keyword_hits >= 2 else 0.0)
            + (0.1 if line_count >= 4 and code_line_hits > 0 else 0.0)
            + (0.1 if "```" in text else 0.0),
            1.0,
        )
        contains_code = code_signal >= 0.48 or code_line_hits >= 2 or "```" in text

        retrieval_signal = min(
            self._normalized(retrieval_marker_count, 3) * 0.7
            + (0.2 if reference_hits > 0 else 0.0)
            + (0.1 if input_length >= 1200 else 0.0),
            1.0,
        )
        summarization_signal = min(
            self._normalized(summarization_marker_count, 2) * 0.75
            + (0.15 if input_length >= 280 else 0.0)
            + (0.1 if line_count >= 4 and not contains_code else 0.0),
            1.0,
        )
        reasoning_signal = min(
            self._normalized(reasoning_marker_count, 3) * 0.68
            + (0.15 if question_count > 0 else 0.0)
            + (0.1 if input_length >= 180 and not contains_code else 0.0)
            + (0.07 if needs_json else 0.0),
            1.0,
        )

        needs_long_context = input_length > 4000 or line_count > 80
        needs_tools = retrieval_signal >= 0.55 or (reference_hits > 0 and retrieval_keyword_hits > 0)
        high_reasoning = reasoning_signal >= 0.6 or (needs_json and reasoning_signal >= 0.48)

        return TaskFeatures(
            input_length=input_length,
            line_count=line_count,
            question_count=question_count,
            code_marker_count=code_marker_count,
            retrieval_marker_count=retrieval_marker_count,
            reasoning_marker_count=reasoning_marker_count,
            summarization_marker_count=summarization_marker_count,
            contains_code=contains_code,
            needs_long_context=needs_long_context,
            needs_tools=needs_tools,
            needs_json=needs_json,
            high_reasoning=high_reasoning,
            code_signal=round(code_signal, 3),
            retrieval_signal=round(retrieval_signal, 3),
            reasoning_signal=round(reasoning_signal, 3),
            summarization_signal=round(summarization_signal, 3),
        )

    def _keyword_hits(self, text: str, keywords: tuple[str, ...]) -> int:
        return sum(1 for keyword in keywords if keyword in text)

    def _pattern_hits(
        self,
        text: str,
        patterns: tuple[str, ...],
        *,
        multiline: bool = False,
    ) -> int:
        flags = re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE
        return sum(len(re.findall(pattern, text, flags)) for pattern in patterns)

    def _normalized(self, value: int, full_score_at: int) -> float:
        if full_score_at <= 0:
            return 0.0
        return min(value / full_score_at, 1.0)
