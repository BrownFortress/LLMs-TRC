import time
from typing import Any
from transformers import AutoTokenizer
from openai import OpenAI

SEC_PER_REQUEST = 1

class ChatGPTModel:
    def __init__(self, api_key: str, tokenizer: AutoTokenizer, model: str):
        self.client = OpenAI(api_key=api_key)
        self.tokenizer = tokenizer
        self.model = model
        self.device = "cpu"

    def generate(self, input_ids, max_new_tokens=5, q_a_list=[], *args: Any, **kwargs: Any) -> Any:
        if self.model == "davinci-002":
            assert len(input_ids) == 1
            input_ids = input_ids[0]
            if len(q_a_list) == 0:
                print("class")
                input_str = "".join([input_ids["examples"], input_ids["context"]])
            else:
                print("sequential questions")
                input_str = "".join([input_ids["examples"], input_ids["context"]] + q_a_list)
            print(input_str)
            compl = self.client.completions.create(
                model=self.model,
                prompt=input_str,
                max_tokens=max_new_tokens,
                temperature=0,
                n=1 # num completions per prompt
            )
            outputs = [compl.choices[0].text]
        else:
            assert len(input_ids) == 1

            input_ids = input_ids[0]
            sys_prompt = input_ids["examples"]
            messages = [{"role": "system", "content": sys_prompt}]

            if len(q_a_list) == 0:
                print("class")
                usr_prompt = input_ids["context"]
                messages.append({"role": "user", "content": usr_prompt})
            else:
                print("sequential questions")
                usr_prompt = "".join([input_ids["context"], q_a_list[0]])
                messages.append({"role": "user", "content": usr_prompt})
                # [1:] -> first question in usr_prompt
                for i, q_a in enumerate(q_a_list[1:]):
                    if i % 2 == 0:
                        answ = q_a
                        messages.append({"role": "system", "content": answ})
                    else:
                        question = q_a
                        messages.append({"role": "user", "content": question})
            print(messages)
            start = time.time_ns()
            compl = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0,
                n=1 # num completions per prompt
            )
            end = time.time_ns()
            passed = (end - start)/10**9
            to_wait = SEC_PER_REQUEST - passed
            print("Passed: ", passed)
            if to_wait > 0:
                print("wait: ", to_wait)
                time.sleep(to_wait)
            outputs = [compl.choices[0].message.content]
        print(compl)
        return outputs


    def to(self, device):
        self.device = device

    def eval(self):
        pass

    def train(self):
        pass
