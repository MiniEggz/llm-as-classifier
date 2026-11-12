import gc
import json
import re
import time
from typing import List, Optional, Union

import torch
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList)

from utils import set_all_seeds


class JSONCompletionStopping(StoppingCriteria):
    """Stopping criteria for JSON objects -- stop as soon as valid JSON object has
    been produced.
    """

    def __init__(self, tokenizer, prompt_len, **kwargs):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self._decoder = json.JSONDecoder()

    def _has_complete_json_object(self, text: str) -> bool:
        start = text.find("{")
        if start == -1:
            return False

        try:
            obj, _ = self._decoder.raw_decode(text, idx=start)
            return isinstance(obj, dict)
        except json.JSONDecodeError:
            return False

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ):
        batch_size = input_ids.shape[0]
        stop_matrix = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        num_generated = input_ids.shape[1] - self.prompt_len

        for i in range(batch_size):
            new_tokens = input_ids[i, self.prompt_len :]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            if self._has_complete_json_object(generated_text):
                stop_matrix[i] = True

        return stop_matrix


class LabelClassificationStopping(StoppingCriteria):
    """Stopping criteria to prevent base models from generating unnecessarily long
    outputs.
    """

    def __init__(
        self,
        tokenizer,
        target_labels,
        max_label_len,
        is_prefix_of_another,
        prompt_len,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.target_labels = target_labels
        self.prompt_len = prompt_len
        self.max_label_len = max_label_len
        self.is_prefix_of_another = is_prefix_of_another

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ):
        batch_size = input_ids.shape[0]
        stop_matrix = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        for i in range(batch_size):
            new_tokens = input_ids[i, self.prompt_len :]
            generated_text = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()

            for label in self.target_labels:
                if label in generated_text:
                    # if this label is a prefix of a longer one, hold off unless
                    # we've already generated more than the longest label's length
                    # (meaning it truly stopped short and won't extend further)
                    if (
                        self.is_prefix_of_another[label]
                        and len(generated_text) < self.max_label_len
                    ):
                        continue
                    stop_matrix[i] = True
                    break

        return stop_matrix


class GreedyInferenceBase:
    def __init__(
        self,
        model_name: str,
        class_labels: List[str],
        device: str = "cuda",
        num_gpus: int = 1,
        max_model_len: Optional[int] = None,
        dtype=torch.float16,
    ):
        self.model_name = model_name
        self.class_labels = class_labels
        self.device = device
        self.max_model_len = max_model_len if max_model_len is not None else 2048

        # load tokeniser
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # add padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # padding to left to not affect generation
        self.tokenizer.padding_side = "left"

        # load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
            trust_remote_code=True,
            dtype=dtype,
        )

        # for stopping callback
        self.max_label_len = max(len(l) for l in self.class_labels)
        self.is_prefix_of_another = {
            label: any(
                other != label and other.startswith(label)
                for other in self.class_labels
            )
            for label in self.class_labels
        }

        # generation parameters (to match with vllm + greedy decoding)
        self.gen_config = {
            "max_new_tokens": 64,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

    def classify(
        self,
        ipt: Union[str, List[str], List[dict]],
        stop_criteria_class=LabelClassificationStopping,
        **kwargs,
    ) -> tuple[list, list, Optional[list]]:

        set_all_seeds()

        # get prompts in list for batch inference
        is_single_input = isinstance(ipt, str)
        prompts = [ipt] if is_single_input else ipt

        start_time = time.time()

        # set mini-batch size
        outputs = []

        start_time = time.time()

        # process by prompt
        for prompt in tqdm(
            prompts,
            desc=f"Running inference:",
        ):

            stop_criteria = stop_criteria_class(
                tokenizer=self.tokenizer,
                target_labels=self.class_labels,
                max_label_len=self.max_label_len,
                is_prefix_of_another=self.is_prefix_of_another,
                prompt_len=len(prompt),
            )
            try:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_model_len,
                    stopping_criteria=StoppingCriteriaList([stop_criteria]),
                ).to(self.model.device)

                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, **self.gen_config)

                # only extract generated tokens
                prompt_len = inputs.input_ids.shape[1]
                decoded = self.tokenizer.decode(
                    generated_ids[0, prompt_len:], skip_special_tokens=True
                )
                outputs.append(decoded)

                # clear memory after each mini-batch
                del inputs
                del generated_ids
                torch.cuda.empty_cache()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                outputs.append("OOM ERROR")
                gc.collect()
                torch.cuda.empty_cache()

        time_taken = time.time() - start_time
        num_samples = len(outputs)
        print(f"Time taken for inference on {num_samples}: {time_taken}.")
        print(f"Time taken per data point: {time_taken / num_samples}")

        confs = [1.0 for _ in range(num_samples)]

        return outputs, confs, outputs


class GreedyInferenceBaseConf(GreedyInferenceBase):

    def _extract_from_gen(self, output: str):
        """Extract individual results for label and confidence from
        JSON output (hopefully) from LLM.

        Args:
            output (str): string output of the LLM - hopefully in JSON format.

        Returns:
            tuple[str, float]: label and confidence for the given output.
        """
        try:
            match = re.search(r"\{.*\}", output, re.DOTALL)
            if not match:
                return output, -1

            data = json.loads(match.group())

            label = data.get("label", output)
            conf = data.get("confidence", -1)

            return label, conf

        except (json.JSONDecodeError, ValueError, TypeError):
            return output, -1

    def classify(
        self,
        ipt: Union[str, List[str], List[dict]],
        stop_criteria_class=JSONCompletionStopping,
        **kwargs,
    ) -> tuple[list, list, Optional[list]]:
        outputs, _, _ = super().classify(
            ipt=ipt, stop_criteria_class=stop_criteria_class
        )

        # extract labels and confs from outputs
        results = [self._extract_from_gen(output) for output in outputs]
        labels, confs = map(list, zip(*results))

        return labels, confs, outputs


class GreedyInferenceBaseJson(GreedyInferenceBaseConf):

    def classify(
        self,
        ipt: Union[str, List[str], List[dict]],
        stop_criteria_class=JSONCompletionStopping,
        **kwargs,
    ) -> tuple[list, list, Optional[list]]:
        labels, _, outputs = super().classify(
            ipt=ipt, stop_criteria_class=stop_criteria_class
        )
        confs = [1.0 for _ in labels]
        return labels, confs, outputs
