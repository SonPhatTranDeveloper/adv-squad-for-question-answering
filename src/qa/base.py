class QABase:
    def get_answers(
        self, questions_and_contexts: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """
        Get the answer to a question from a context.

        Args:
            questions_and_contexts: A list of dictionaries containing the question and
            context, has format
            {
                "question": str,
                "context": str,
                "id": str,
            }

        Returns:
            A list of answers to the questions, has format
            {
                "prediction_text": str,
                "id": str,
            }
        """
        raise NotImplementedError("Subclasses must implement this method")
