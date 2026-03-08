from app.schemas import TaskClassification, TaskFeatures, TaskType


class TaskClassifier:
    def classify(self, features: TaskFeatures) -> TaskClassification:
        scores = self._task_scores(features)
        ranked_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_task, best_score = ranked_scores[0]
        second_score = ranked_scores[1][1]

        if best_task is not TaskType.GENERAL_QA and (best_score < 0.45 or best_score - second_score < 0.08):
            best_task = TaskType.GENERAL_QA
            best_score = scores[TaskType.GENERAL_QA]

        confidence = min(0.42 + best_score * 0.38 + max(best_score - second_score, 0.0) * 0.35, 0.95)

        if best_task is TaskType.GENERAL_QA:
            reason = (
                "Signals stayed mixed, keeping general_qa. "
                f"code={features.code_signal:.2f}, reasoning={features.reasoning_signal:.2f}, "
                f"rag={features.retrieval_signal:.2f}, summary={features.summarization_signal:.2f}."
            )
        else:
            reason = (
                f"Feature scores favored {best_task.value} ({best_score:.2f}). "
                f"code={features.code_signal:.2f}, reasoning={features.reasoning_signal:.2f}, "
                f"rag={features.retrieval_signal:.2f}, summary={features.summarization_signal:.2f}."
            )

        return TaskClassification(
            task_type=best_task,
            confidence=round(confidence, 3),
            reason=reason,
        )

    def _task_scores(self, features: TaskFeatures) -> dict[TaskType, float]:
        specialized_scores = {
            TaskType.CODING: min(
                features.code_signal * 0.88
                + (0.12 if features.code_marker_count >= 2 else 0.0)
                + (0.12 if features.contains_code else 0.0),
                1.0,
            ),
            TaskType.SUMMARIZATION: min(
                features.summarization_signal * 0.82
                + (0.08 if features.summarization_marker_count > 0 else 0.0)
                + (0.08 if not features.contains_code else 0.0)
                + (0.1 if features.input_length >= 240 else 0.0),
                1.0,
            ),
            TaskType.REASONING: min(
                features.reasoning_signal * 0.82
                + (0.06 if features.reasoning_marker_count >= 2 else 0.0)
                + (0.08 if features.question_count > 0 else 0.0)
                + (0.1 if features.needs_json else 0.0),
                1.0,
            ),
            TaskType.RAG: min(
                features.retrieval_signal * 0.8
                + (0.08 if features.retrieval_marker_count >= 2 else 0.0)
                + (0.1 if features.needs_tools else 0.0)
                + (0.1 if features.needs_long_context else 0.0),
                1.0,
            ),
        }
        max_specialized = max(specialized_scores.values())
        general_qa_score = min(
            max(
                0.32,
                0.72 - max_specialized * 0.55 + (0.05 if features.question_count > 0 else 0.0),
            ),
            0.78,
        )
        return specialized_scores | {TaskType.GENERAL_QA: general_qa_score}
