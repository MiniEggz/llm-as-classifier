import gc
import json
import re
import time
from typing import List, Optional, Union

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_clean_pred, set_all_seeds


class SamplingClassifier:
    def __init__(
        self,
        model_name: str,
        class_labels: List[str],
        device: str = "cuda",
        num_gpus: int = 1,
        max_model_len: Optional[int] = None,
        num_gens: int = 10,
        dtype=torch.float16,
    ):
        self.model_name = model_name
        self.class_labels = class_labels
        self.device = device
        self.max_model_len = max_model_len if max_model_len is not None else 2048
        self.num_gens = num_gens

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
            dtype=dtype,
            device_map="auto" if num_gpus > 1 else self.device,  # auto for multi-gpu
            trust_remote_code=True,
        )

        # generation parameters (to match with vllm + greedy decoding)
        self.gen_config = {
            "max_new_tokens": 64,
            "do_sample": True,
            "temperature": 0.5,
            "num_return_sequences": self.num_gens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

    def classify(
        self,
        ipt: Union[str, List[str], List[dict]],
        input_messages: bool = False,
        **kwargs,
    ) -> tuple[list, list, Optional[list]]:

        set_all_seeds()
        print("set the seeds")

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

        # set mini-batch size
        mini_batch_size = kwargs.get("batch_size", 1)
        outputs = []

        start_time = time.time()

        # process in chunks
        for i in tqdm(
            range(0, len(prompts), mini_batch_size),
            desc=f"Running inference (batch_size={mini_batch_size}",
        ):
            batch_prompts = prompts[i : i + mini_batch_size]

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_model_len,
            ).to(self.model.device)

            oom = False
            failed = False
            with torch.no_grad():
                try:
                    generated_ids = self.model.generate(**inputs, **self.gen_config)
                # add fallback for OOMs
                except torch.cuda.OutOfMemoryError:
                    oom = True

                if oom:
                    print("OOM with parallel generation, falling back to sequential...")
                    gc.collect()
                    torch.cuda.empty_cache()
                    try:
                        all_ids = []
                        single_gen_config = {
                            **self.gen_config,
                            "num_return_sequences": 1,
                        }
                        for j in range(len(batch_prompts)):
                            single_input = {k: v[j : j + 1] for k, v in inputs.items()}
                            for _ in range(self.num_gens):
                                out = self.model.generate(
                                    **single_input, **single_gen_config
                                )
                                all_ids.append(out)
                        # stack so shape matches the parallel case (batch*num_gens, seq_len)
                        max_len = max(x.shape[1] for x in all_ids)
                        all_ids = [
                            torch.nn.functional.pad(
                                x,
                                (max_len - x.shape[1], 0),
                                value=self.tokenizer.pad_token_id,
                            )
                            for x in all_ids
                        ]
                        generated_ids = torch.cat(all_ids, dim=0)
                    except torch.cuda.OutOfMemoryError:
                        print("Failed again, marking as OOM.")
                        failed = True

            if failed:
                decoded_batch = [["OOM"] for _ in batch_prompts]
            else:
                # only extract generated tokens
                prompt_len = inputs.input_ids.shape[1]
                decoded_batch = self.tokenizer.batch_decode(
                    generated_ids[:, prompt_len:], skip_special_tokens=True
                )
                # make the decoded batch 2d based on how many generations
                # per prompt
                decoded_batch = [
                    decoded_batch[j * self.num_gens : (j + 1) * self.num_gens]
                    for j in range(len(batch_prompts))
                ]
                del generated_ids

            # outputs contains a list of all outputs for each prompt
            outputs.extend(decoded_batch)

            # clear memory after each mini-batch
            del inputs
            torch.cuda.empty_cache()

        time_taken = time.time() - start_time
        num_samples = len(outputs)
        print(f"Time taken for inference on {num_samples}: {time_taken}.")
        print(f"Time taken per data point: {time_taken / num_samples}")

        # handle generation of soft label and label output
        soft_labels = [
            [
                sum(
                    1
                    for gen in output
                    if get_clean_pred(gen, self.class_labels) == label
                )
                for label in self.class_labels
            ]
            for output in outputs
        ]

        # cheaper softmax (everything should be +ve)
        soft_labels = [
            (
                [v / sum(soft_label) for v in soft_label]
                if sum(soft_label) != 0
                else soft_label
            )
            for soft_label in soft_labels
        ]

        # TODO: might need to think about case where none of the generations are correct
        # will always just pick SUPPORTS
        labels = [
            (
                self.class_labels[soft_label.index(max(soft_label))]
                if sum(soft_label) != 0
                else "None"
            )
            for soft_label in soft_labels
        ]
        confs = [float(max(soft_label)) for soft_label in soft_labels]

        return labels, confs, outputs
