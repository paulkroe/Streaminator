import re

class GSM8KAnswerChecker:
    @staticmethod
    def _remove_prompt(text):
        """
        Removes everything before and including the delimiter
        'Now solve the following problem:\n'.
        """
        delimiter = "Now solve the following problem:\n"
        if delimiter in text:
            return text.split(delimiter, 1)[-1].strip()
        return text.strip()

    @staticmethod
    def _remove_think_tags(text):
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    @staticmethod
    def _extract_answer(text):
        def clean_number(num_str):
            # Remove any characters except digits, period, and minus sign.
            cleaned = re.sub(r'[^\d.-]', '', num_str)
            try:
                return float(cleaned)
            except ValueError:
                return None

        # Remove the prompt parts.
        cleaned_text = GSM8KAnswerChecker._remove_prompt(text)

        # 1) Attempt to extract from a \boxed{} tag.
        boxed_matches = re.findall(r'\\boxed{([^}]*)}', cleaned_text)
        if boxed_matches:
            return clean_number(boxed_matches[-1])

        # 2) Attempt to extract from a line starting with three or more # signs.
        hash_match = re.search(r'#{3,}\s*(.*)', cleaned_text)
        if hash_match:
            return clean_number(hash_match.group(1).strip())

        # 3) Otherwise, look for the last number from the last non-empty line.
        lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
        for line in reversed(lines):
            number_matches = re.findall(r'\d+(?:\.\d+)?', line)
            if number_matches:
                return clean_number(number_matches[-1])
        return None

    @staticmethod
    def check_answer(answer, ground_truth):
        # Remove <think> tags from the generated text.
        answer = GSM8KAnswerChecker._remove_think_tags(answer)
        extracted_answer = GSM8KAnswerChecker._extract_answer(answer)
        extracted_ground_truth = GSM8KAnswerChecker._extract_answer(ground_truth)
        if extracted_answer is not None and extracted_ground_truth is not None:
            if abs(extracted_answer - extracted_ground_truth) < 1e-6:
                return {
                    "correct": True,
                    "mode": "match",
                    "extracted_answer": extracted_answer,
                    "extracted_ground_truth": extracted_ground_truth
                }
            else:
                return {
                    "correct": False,
                    "mode": "match",
                    "extracted_answer": extracted_answer,
                    "extracted_ground_truth": extracted_ground_truth
                }
        else:
            return {
                "correct": False,
                "mode": "no_match",
                "extracted_answer": extracted_answer,
                "extracted_ground_truth": extracted_ground_truth
            }

    @staticmethod
    def eval(output_dict):
        """
        output_dict should map:
          prompt_text -> {
            "generations": [list of strings],
            "ground_truth": string
          }
        Returns a list of dicts, each containing evaluation data for that prompt.
        """
        evaluated_outputs = []
        for prompt, entry in output_dict.items():
            generations = entry["generations"]
            ground_truth = entry["ground_truth"]

            evaluated_answers = []
            for text in generations:
                answer_eval = GSM8KAnswerChecker.check_answer(text, ground_truth)
                evaluated_answers.append({
                    "text": text,
                    "answer_eval": answer_eval
                })

            # "accuracy" = fraction of completions that are correct
            acc = sum(a["answer_eval"]["correct"] for a in evaluated_answers) / len(evaluated_answers)
            evaluated_outputs.append({
                "prompt": prompt,
                "answers": evaluated_answers,
                "ground_truth": ground_truth,
                "evaluation": {
                    "accuracy": acc,
                    "pass@n": acc > 0.0,    # pass@n => at least one correct completion
                    "match@n": acc >= 0.5  # match@n => at least half are correct
                }
            })
        return evaluated_outputs
