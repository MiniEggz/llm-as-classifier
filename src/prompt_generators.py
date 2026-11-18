import ast
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from jinja2.exceptions import TemplateError
from transformers import PreTrainedTokenizerBase


def truncate_input_text(
    input_text: str,
    sys_template: str,
    user_template: str,
    max_model_len: int,
    tokenizer: PreTrainedTokenizerBase,
    buffer_len: int = 50,
) -> str:
    sys_len = len(tokenizer.tokenize(sys_template))
    user_template_len = len(tokenizer.tokenize(user_template))
    budget = max_model_len - (sys_len + user_template_len + buffer_len)

    if budget < 0:
        print("WARNING: input for this data point is empty!")
        return ""

    token_ids = tokenizer.encode(input_text, add_special_tokens=False)
    if len(token_ids) < budget:
        return input_text

    truncated = tokenizer.decode(token_ids[:budget])
    assert isinstance(truncated, str)
    return truncated


class BasePromptGenerator(ABC):
    """Abstract class for specific dataset prompt generators
    to build from.
    """

    TEMPLATE_VARS = []
    BUFFER_LEN = 150  # default buffer_len

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        template: str,
        buffer_len: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.template = template
        if not self.template_ok():
            template_vars = ", ".join(self.TEMPLATE_VARS)
            raise ValueError(f"Template must contain all of: {template_vars}")

        self.buffer_len = int(buffer_len) if buffer_len is not None else self.BUFFER_LEN

    def template_ok(self) -> bool:
        """Check whether self.template is a valid
        template for the specific dataset.

        Returns:
            bool: whether the template is valid.
        """
        return all(var in self.template for var in self.TEMPLATE_VARS)

    @abstractmethod
    def generate_base_prompt(self, row, max_model_len=None) -> str:
        """Generate user prompt from the class template
        and dataset row.
        """
        pass

    def map_prompt(self, row, max_model_len=None) -> dict:
        return {
            "input_prompt": self.generate_base_prompt(row, max_model_len=max_model_len)
        }


class Banking77BasePromptGenerator(BasePromptGenerator):
    TEMPLATE_VARS = ["{INPUT_TEXT}"]
    BUFFER_LEN = 50

    # TODO: maybe make this a standard setup
    def generate_base_prompt(self, row, max_model_len=None) -> str:
        input_text = row["input"]

        if max_model_len is not None:
            input_text = truncate_input_text(
                input_text,
                "",
                self.template,
                max_model_len,
                self.tokenizer,
                buffer_len=self.buffer_len,
            )

        return self.template.format(INPUT_TEXT=input_text)


class TwentyNGBasePromptGenerator(BasePromptGenerator):
    TEMPLATE_VARS = ["{INPUT_TEXT}"]
    BUFFER_LEN = 50

    # TODO: maybe make this a standard setup
    def generate_base_prompt(self, row, max_model_len=None) -> str:
        input_text = row["input"]

        if max_model_len is not None:
            input_text = truncate_input_text(
                input_text,
                "",
                self.template,
                max_model_len,
                self.tokenizer,
                buffer_len=self.buffer_len,
            )

        return self.template.format(INPUT_TEXT=input_text)


class SST5BasePromptGenerator(BasePromptGenerator):
    TEMPLATE_VARS = ["{INPUT_TEXT}"]
    BUFFER_LEN = 50

    # TODO: maybe make this a standard setup
    def generate_base_prompt(self, row, max_model_len=None) -> str:
        input_text = row["input"]

        if max_model_len is not None:
            input_text = truncate_input_text(
                input_text,
                "",
                self.template,
                max_model_len,
                self.tokenizer,
                buffer_len=self.buffer_len,
            )

        return self.template.format(INPUT_TEXT=input_text)


class CLINC150BasePromptGenerator(BasePromptGenerator):
    TEMPLATE_VARS = ["{INPUT_TEXT}"]
    BUFFER_LEN = 50

    # TODO: maybe make this a standard setup
    def generate_base_prompt(self, row, max_model_len=None) -> str:
        input_text = row["input"]

        if max_model_len is not None:
            input_text = truncate_input_text(
                input_text,
                "",
                self.template,
                max_model_len,
                self.tokenizer,
                buffer_len=self.buffer_len,
            )

        return self.template.format(INPUT_TEXT=input_text)


class FeverBasePromptGenerator(BasePromptGenerator):
    TEMPLATE_VARS = ["{CONTEXT}", "{INPUT_TEXT}"]
    BUFFER_LEN = 50

    def _format_context(self, row_context: list) -> str:
        """Format context from FEVER into XML-like tags for input
        into the prompt.

        Args:
            row_context (list): 3d list of elements from FEVER dataset context.

        Returns:
            str: XML-like formatted context.
        """
        context = ""
        if row_context == [[]]:
            context = "No context provided."
        else:
            # add evidence set mapping here
            for i, evidence_set in enumerate(row_context):
                context += "<evidence_set>" + "\n" + "  "
                for evidence_piece in evidence_set:
                    context += (
                        "  "
                        + f"<evidence_piece title={evidence_piece[0]}>"
                        + "\n"
                        + "  " * 3
                    )
                    context += evidence_piece[1] + "\n" + "  " * 2
                    context += "</evidence_piece>" + "\n" + "  "
                context += "</evidence_set>"
                if i < len(row_context) - 1:
                    context += "\n" + "  "
        return context

    def generate_base_prompt(self, row, max_model_len=None) -> str:
        claim = row["input"]
        context = self._format_context(row["context"])

        if max_model_len is not None:
            context = truncate_input_text(
                context,
                claim,
                self.template,
                max_model_len,
                self.tokenizer,
                buffer_len=self.buffer_len,
            )

        return self.template.format(CONTEXT=context, INPUT_TEXT=claim)


class PromptGenerator(ABC):
    """Abstract class for specific dataset prompt generators
    to build from.
    """

    TEMPLATE_VARS = []
    BUFFER_LEN = 150  # default buffer

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        sys_prompt: str,
        template: str,
        buffer_len: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.sys_prompt = sys_prompt
        self.template = template
        if not self.template_ok():
            template_vars = ", ".join(self.TEMPLATE_VARS)
            raise ValueError(f"Template must contain all of: {template_vars}")

        self.buffer_len = int(buffer_len) if buffer_len is not None else self.BUFFER_LEN

    def template_ok(self) -> bool:
        """Check whether self.template is a valid
        template for the specific dataset.

        Returns:
            bool: whether the template is valid.
        """
        return all(var in self.template for var in self.TEMPLATE_VARS)

    @abstractmethod
    def _generate_user_prompt(self, row, max_model_len=None) -> str:
        """Generate user prompt from the class template
        and dataset row.
        """
        pass

    def messages_from_row(self, row, max_model_len=None) -> List[dict]:
        """Create prompt messages from provided dataset row.

        Args:
            row (dict): dataset row.

        Returns:
            List[dict]: list of formatted messages.
        """
        user_prompt = self._generate_user_prompt(row, max_model_len=max_model_len)

        # create messages list
        sys_message = {"role": "system", "content": self.sys_prompt}
        try:
            self.tokenizer.apply_chat_template([sys_message])
            user_message = {"role": "user", "content": user_prompt}
            messages = [sys_message, user_message]
        except (TemplateError, ValueError):
            combined_message = f"**System instructions**\n{self.sys_prompt}\n**End of system instructions**\n\n{user_prompt}"
            messages = [{"role": "user", "content": combined_message}]

        return messages

    def map_messages(self, row, max_model_len=None) -> dict:
        return {"messages": self.messages_from_row(row, max_model_len=max_model_len)}


# TODO: make sure abc prompting works here...
class MMLUPromptGenerator(PromptGenerator):
    TEMPLATE_VARS = ["{INPUT_TEXT}", "{OPTIONS}"]
    BUFFER_LEN = 150

    def _generate_user_prompt(self, row, max_model_len=None) -> str:
        question = row["input"]
        options = ", ".join(row["prompt_options"])

        if max_model_len is not None:
            question = truncate_input_text(
                question,
                self.sys_prompt,
                self.template,
                max_model_len,
                self.tokenizer,
                buffer_len=self.buffer_len,
            )

        return self.template.format(INPUT_TEXT=question, OPTIONS=options)


class Banking77PromptGenerator(PromptGenerator):
    TEMPLATE_VARS = ["{INPUT_TEXT}", "{OPTIONS}"]
    BUFFER_LEN = 300

    # TODO: maybe make this a standard setup
    def _generate_user_prompt(self, row, max_model_len=None) -> str:
        input_text = row["input"]
        options = ", ".join(row["prompt_options"])

        if max_model_len is not None:
            input_text = truncate_input_text(
                input_text,
                self.sys_prompt,
                self.template,
                max_model_len,
                self.tokenizer,
                buffer_len=self.buffer_len,
            )

        return self.template.format(INPUT_TEXT=input_text, OPTIONS=options)


class TwentyNGPromptGenerator(PromptGenerator):
    TEMPLATE_VARS = ["{INPUT_TEXT}", "{OPTIONS}"]
    BUFFER_LEN = 300

    # TODO: maybe make this a standard setup
    def _generate_user_prompt(self, row, max_model_len=None) -> str:
        input_text = row["input"]
        options = ", ".join(row["prompt_options"])

        if max_model_len is not None:
            input_text = truncate_input_text(
                input_text,
                self.sys_prompt,
                self.template,
                max_model_len,
                self.tokenizer,
                buffer_len=self.buffer_len,
            )

        return self.template.format(INPUT_TEXT=input_text, OPTIONS=options)


class SST5PromptGenerator(PromptGenerator):
    TEMPLATE_VARS = ["{INPUT_TEXT}", "{OPTIONS}"]
    BUFFER_LEN = 200

    # TODO: maybe make this a standard setup
    def _generate_user_prompt(self, row, max_model_len=None) -> str:
        input_text = row["input"]
        options = ", ".join(row["prompt_options"])

        if max_model_len is not None:
            input_text = truncate_input_text(
                input_text,
                self.sys_prompt,
                self.template,
                max_model_len,
                self.tokenizer,
                buffer_len=self.buffer_len,
            )

        return self.template.format(INPUT_TEXT=input_text, OPTIONS=options)


class CLINC150PromptGenerator(PromptGenerator):
    TEMPLATE_VARS = ["{INPUT_TEXT}", "{OPTIONS}"]
    BUFFER_LEN = 500

    # TODO: maybe make this a standard setup
    def _generate_user_prompt(self, row, max_model_len=None) -> str:
        input_text = row["input"]
        options = ", ".join(row["prompt_options"])

        if max_model_len is not None:
            input_text = truncate_input_text(
                input_text,
                self.sys_prompt,
                self.template,
                max_model_len,
                self.tokenizer,
                buffer_len=self.buffer_len,
            )

        return self.template.format(INPUT_TEXT=input_text, OPTIONS=options)


class FeverPromptGenerator(PromptGenerator):
    TEMPLATE_VARS = ["{CONTEXT}", "{INPUT_TEXT}", "{OPTIONS}"]

    def _format_context(self, row_context: list) -> str:
        """Format context from FEVER into XML-like tags for input
        into the prompt.

        Args:
            row_context (list): 3d list of elements from FEVER dataset context.

        Returns:
            str: XML-like formatted context.
        """
        context = ""
        if row_context == [[]]:
            context = "No context provided."
        else:
            # add evidence set mapping here
            for i, evidence_set in enumerate(row_context):
                context += "<evidence_set>" + "\n" + "  "
                for evidence_piece in evidence_set:
                    context += (
                        "  "
                        + f"<evidence_piece title={evidence_piece[0]}>"
                        + "\n"
                        + "  " * 3
                    )
                    context += evidence_piece[1] + "\n" + "  " * 2
                    context += "</evidence_piece>" + "\n" + "  "
                context += "</evidence_set>"
                if i < len(row_context) - 1:
                    context += "\n" + "  "
        return context

    def _generate_user_prompt(self, row, max_model_len=None) -> str:
        claim = row["input"]
        options = ", ".join(row["prompt_options"])
        context = self._format_context(row["context"])

        if max_model_len is not None:
            context = truncate_input_text(
                context,
                self.sys_prompt,
                self.template,
                max_model_len,
                self.tokenizer,
                buffer_len=250,
            )

        return self.template.format(CONTEXT=context, INPUT_TEXT=claim, OPTIONS=options)


class IndependentJudge(PromptGenerator):

    TEMPLATE_VARS = ["SYS_PROMPT", "USER_PROMPT", "RESPONSE"]

    def _generate_user_prompt(self, row, max_model_len=None) -> str:
        messages = ast.literal_eval(row["messages"])
        response = row["pred"]

        if len(messages) == 2:  # most cases
            sys_prompt = messages[0]["content"]
            user_prompt = messages[1]["content"]
        elif len(messages) == 1:  # qwen 3.5
            sys_prompt = ""
            user_prompt = messages[0]["content"]
        else:
            raise ValueError("More than 2 messages detected, please check the input.")

        return self.template.format(
            SYS_PROMPT=sys_prompt, USER_PROMPT=user_prompt, RESPONSE=response
        )


class JevPromptGenerator(ABC):
    """Abstract class for specific dataset prompt generators
    to build from -- specifically for the JEV model that requires
    an API body.
    """

    def __init__(self, instructions: str, criteria: dict, model: str = "jev-1.13.0"):
        self.instructions = instructions
        self.criteria = criteria
        self.model = model

        self.questions_dict = self._generate_questions_dict()

    @abstractmethod
    def _generate_state(self, row) -> str | dict:
        """Generate state dict for JEV. Expects the row to already
        not contain a high number of tokens.
        """
        pass

    def _generate_questions_dict(self):
        return {
            "label": {
                "type": "choice",
                "instructions": self.instructions,
                "criteria": self.criteria,
            }
        }

    def body_from_row(self, row) -> dict:
        """Create API body from provided dataset row.

        Args:
            row (dict): dataset row.

        Returns:
            dict: body to submit to JEV API.
        """
        state_dict = self._generate_state(row)

        return {
            "model": self.model,
            "state": state_dict,
            "questions": self.questions_dict,
        }

        # create messages list
        sys_message = {"role": "system", "content": self.sys_prompt}
        try:
            self.tokenizer.apply_chat_template([sys_message])
            user_message = {"role": "user", "content": user_prompt}
            messages = [sys_message, user_message]
        except (TemplateError, ValueError):
            combined_message = f"**System instructions**\n{self.sys_prompt}\n**End of system instructions**\n\n{user_prompt}"
            messages = [{"role": "user", "content": combined_message}]

        return messages

    def map_body(self, row, max_model_len=None) -> dict:
        return {"body": self.body_from_row(row)}


class Banking77JevPromptGenerator(JevPromptGenerator):

    def _generate_state(self, row) -> str | dict:
        return row["input"]


class TwentyNGJevPromptGenerator(JevPromptGenerator):

    def _generate_state(self, row) -> str | dict:
        return row["input"]


class SST5JevPromptGenerator(JevPromptGenerator):

    def _generate_state(self, row) -> str | dict:
        return row["input"]


class CLINC150JevPromptGenerator(JevPromptGenerator):

    def _generate_state(self, row) -> str | dict:
        return row["input"]


class FeverJevPromptGenerator(JevPromptGenerator):

    def _get_evidence_sets(self, row_context: list) -> list:
        """Format context from FEVER into evidence sets array.

        Args:
            row_context (list): 3d list of elements from FEVER dataset context.

        Returns:
            dict: JEV state evidence_sets array.
        """
        evidence_sets = []
        # if not empty...
        if row_context != [[]]:
            for i, evidence_set in enumerate(row_context):
                evidence_sets.append(
                    [
                        {"title": evidence_piece[0], "text": evidence_piece[1]}
                        for evidence_piece in evidence_set
                    ]
                )
        return evidence_sets

    def _generate_state(self, row) -> str | dict:
        claim = row["input"]
        evidence_sets = self._get_evidence_sets(row["context"])
        return {"claim": claim, "evidence_sets": evidence_sets}
