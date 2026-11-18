from abc import abstractmethod
from typing import Optional, Union


class BaseClassifier:

    @abstractmethod
    def classify(
        self,
        ipt: Union[str, list[str], list[dict]],
        input_messages: bool = False,
        **kwargs
    ) -> tuple[list, list, Optional[list]]:
        """Classify a set of inputs (or single input) that will be either
        of type string (prompt), List of strings (prompts), or a list of
        messages (dicts) (this can only be a single list).

        Args:
            ipt (str | List[str] | List[dict]): input to the classifier
                which is either:
                - str: single prompt;
                - List[str]: multiple prompts (for batch mode);
                - List[dict]: messages to tokenise and input.
            input_messages (bool): whether the input will be messages --
                can therefore just check for a whether the input is a list.

        Returns:
            Tuple[list, list]: predictions and confidence scores as two lists
            of size 1 (if single input or messages) or len(ipt).
        """
        pass
