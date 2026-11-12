import gc
import json
import re
import time
from typing import List, Optional, Union

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import set_all_seeds


class GreedyInference:
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

        # generation parameters (to match with vllm + greedy decoding)
        self.gen_config = {
            "max_new_tokens": 64,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

    def classify(
        self,
        ipt: Union[str, List[str], List[dict]],
        input_messages: bool = False,
        **kwargs,
    ) -> tuple[list, list, Optional[list]]:

        set_all_seeds()

        if input_messages and isinstance(ipt, list):
            ipt = self.tokenizer.apply_chat_template(
                ipt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

        # get prompts in list for batch inference
        is_single_input = isinstance(ipt, str)
        prompts = [ipt] if is_single_input else ipt

        start_time = time.time()

        start_time = time.time()

        # process by prompt
        outputs = []
        for prompt in tqdm(
            prompts,
            desc=f"Running inference:",
        ):

            try:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_model_len,
                ).to(self.model.device)

                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, **self.gen_config)

                # only extract generated tokens
                prompt_len = inputs.input_ids.shape[1]
                decoded = self.tokenizer.decode(
                    generated_ids[0, prompt_len:], skip_special_tokens=True
                )
                outputs.append(decoded)

                # clear memory after each prompt
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


class GreedyInferenceConf(GreedyInference):

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
        input_messages: bool = False,
        **kwargs,
    ) -> tuple[list, list, Optional[list]]:
        outputs, _, _ = super().classify(ipt=ipt, input_messages=input_messages)

        # extract labels and confs from outputs
        results = [self._extract_from_gen(output) for output in outputs]
        labels, confs = map(list, zip(*results))

        return labels, confs, outputs


class GreedyInferenceJson(GreedyInferenceConf):

    def classify(
        self,
        ipt: Union[str, List[str], List[dict]],
        input_messages: bool = False,
        **kwargs,
    ) -> tuple[list, list, Optional[list]]:
        labels, _, outputs = super().classify(ipt=ipt, input_messages=input_messages)
        confs = [1.0 for _ in labels]
        return labels, confs, outputs
