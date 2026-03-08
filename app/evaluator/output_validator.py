from app.evaluator.quality_checks import has_code_block, has_content, is_valid_json
from app.schemas import TaskFeatures, TaskType, ValidationResult


class OutputValidator:
    def validate(
        self,
        task_type: TaskType,
        features: TaskFeatures,
        output: str,
    ) -> ValidationResult:
        # 校验层只做廉价且稳定的基础检查，不在这里引入第二次模型调用。
        issues: list[str] = []
        json_valid: bool | None = None
        code_block: bool | None = None

        if not has_content(output):
            issues.append("empty_output")

        if features.needs_json:
            json_valid = is_valid_json(output)
            if not json_valid:
                issues.append("invalid_json")

        if task_type is TaskType.CODING or features.contains_code:
            code_block = has_code_block(output)
            if not code_block:
                issues.append("missing_code_block")

        if task_type is TaskType.RAG and "source" not in output.casefold():
            issues.append("missing_source_reference")

        if len(output.strip()) < 12:
            issues.append("suspiciously_short")

        # 只有致命问题才会阻止本次结果返回，其余问题先进入观测数据。
        fatal_issues = {"empty_output", "invalid_json"}
        valid = not any(issue in fatal_issues for issue in issues)
        return ValidationResult(
            valid=valid,
            issues=issues,
            json_valid=json_valid,
            contains_code_block=code_block,
            passed_basic_checks=len(issues) == 0,
        )
