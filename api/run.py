# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import atexit
import json
import os
import time

import requests
from msal import PublicClientApplication, SerializableTokenCache

TOKEN = ""

class LLMClient:
    _ENDPOINT = ''
    _SCOPES = ['']

    def __init__(self):
        self._cache = SerializableTokenCache()
        atexit.register(lambda:
                        open('.llmapi.bin', 'w').write(self._cache.serialize())
                        if self._cache.has_state_changed else None)

        self._app = PublicClientApplication('',
                                            authority='',
                                            token_cache=self._cache)
        if os.path.exists('.llmapi.bin'):
            self._cache.deserialize(open('.llmapi.bin', 'r').read())

    def send_request(self, model_name, request):
        # get the token
        if TOKEN != "":
            token = TOKEN
        else:
            token = self._get_token()

        # populate the headers
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + token,
            'X-ModelType': model_name}

        body = str.encode(json.dumps(request))
        response = requests.post(LLMClient._ENDPOINT, data=body, headers=headers)
        return response.json()

    def send_stream_request(self, model_name, request):
        # get the token
        if TOKEN != "":
            token = TOKEN
        else:
            token = self._get_token()

        # populate the headers
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + token,
            'X-ModelType': model_name}

        body = str.encode(json.dumps(request))
        response = requests.post(LLMClient._ENDPOINT, data=body, headers=headers, stream=True)
        for line in response.iter_lines():
            text = line.decode('utf-8')
            if text.startswith('data: '):
                text = text[6:]
                if text == '[DONE]':
                    break
                else:
                    yield json.loads(text)

    def _get_token(self):
        accounts = self._app.get_accounts()
        result = None

        if accounts:
            # Assuming the end user chose this one
            chosen = accounts[0]

            # Now let's try to find a token in cache for this account
            result = self._app.acquire_token_silent(LLMClient._SCOPES, account=chosen)

        if not result:
            # So no suitable token exists in cache. Let's get a new one from AAD.
            flow = self._app.initiate_device_flow(scopes=LLMClient._SCOPES)

            if "user_code" not in flow:
                raise ValueError(
                    "Fail to create device flow. Err: %s" % json.dumps(flow, indent=4))

            print(flow["message"])

            result = self._app.acquire_token_by_device_flow(flow)

        return result["access_token"]

    def run_llm(self, prompt):
        request_data = {
            "prompt": prompt,
            "max_tokens": 3200,
            "temperature": 0.0,
            "top_p": 1,
            "n": 1,
            "stream": False,
            "logprobs": None,
            "stop": None
        }
        cnt = 0
        while True:
            try:
                response = self.send_request('dev-moonshot', request_data)
                if 'error' in response and "This model's maximum context length is" in response['error']['message']:
                    print(response['error']['message'])
                    return None
                result = response['choices'][0]['text']
                # time.sleep(10)
                return result
            except:
                print(f"retrying with {cnt} times...")
                cnt += 1
                time.sleep(1)


def eval(data_path='../data/structureWeb_infer_v1.jsonl'):
    ans_path = data_path.replace(".jsonl", "_ans.jsonl")
    cnt = 0
    llm_client = LLMClient()
    writer = open(ans_path, 'a')
    with open(data_path, 'r', encoding="utf-8-sig") as f:
        json_list = list(f)
        json_list = [json.loads(json_str) for json_str in json_list]
        for prompt_data in json_list:
            request_data = {
                "prompt": prompt_data["prompt"] + "\r\n",
                "max_tokens": 95,
                "temperature": 0.0,
                "top_p": 1,
                "n": 1,
                "stream": False,
                "logprobs": None,
                "stop": "\n"
            }
            print(f"processing file {cnt}...", end="")
            cnt += 1
            response = llm_client.send_request('dev-moonshot', request_data)
            ans = response['choices'][0]['text']
            data = {"id": prompt_data["id"], "completion": ans}
            json.dump(data, writer)
            writer.flush()
            print("sleeping...")
            time.sleep(15)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', type=str, default='1_shot_metadata_sample200_2.jsonl')
    # args = parser.parse_args()
    # eval(args.data_path)
    prompt = "test"
    llm_client = LLMClient()
    response = llm_client.run_llm(prompt)
    print(response)
