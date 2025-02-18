import asyncio
import dataclasses
from swiftllm.perfpredictor import PerfPredictor, ZeroPerfPredictor
from swiftllm.model_config import LlamaModelConfig

@dataclasses.dataclass
class StepOutput:
    """
    The output of one decoding step
    """
    token_id: int
    request: "Request"


class RawRequest:
    """
    A request issued by user
    """
    prompt: str | list[int]
    max_output_len: int

    def __init__(self, prompt: str | list[int], max_output_len: int):
        self.prompt = prompt
        self.max_output_len = max_output_len


class Request:
    """
    A (queuing, processing, or finished) request in the system
    """   

    prompt_token_ids: list[int]     # Prompt token ids, generated by the tokenizer upon request arrival
    prompt_len: int     # len(prompt_token_ids)
    output_len: int     # Current output length
    max_output_len: int     # Final output length

    output_q: asyncio.Queue[StepOutput] # Queue initialized when the raw request enters the
                                        # engine, and to be set upon a new token being generated
                                        # Mainly for streaming the output back to the user
    finished_event: asyncio.Event       # Event to be set when the request is finished
                                        # Mainly for the non-streaming case

    request_id: int     # Request ID, within range [0, max_seqs_in_block_table).
                        # Generated before being prefilled, and used as the index
                        # into the block table
    output_token_ids: list[int]     # Output token ids'

    @property
    def seq_len(self) -> int:
        return self.prompt_len + self.output_len

    def __init__(self, raw_request: RawRequest):
        # A request is __init__-ed when entering `untokenized_raw_requests`, and
        # its `prompt_token_ids` and `prompt_len` will be set upon tokenization
        self.prompt_token_ids = []
        self.prompt_len = 0
        self.max_output_len = raw_request.max_output_len
        self.output_len = 0
        self.output_q = asyncio.Queue()
        self.finished_event = asyncio.Event()
        self.request_id = -1
        self.output_token_ids = []
    
    def is_finished(self) -> bool:
        return self.output_len == self.max_output_len

    @staticmethod
    def get_ids(reqs: list["Request"]) -> list[int]:
        """
        Get the request IDs of a list of requests
        """
        return [req.request_id for req in reqs]

    @staticmethod
    def get_lens(reqs: list["Request"]) -> list[int]:
        """
        Get the sequence lengths of a list of requests
        """
        return [req.seq_len for req in reqs]


    @staticmethod
    def get_input_tokens(reqs: list["Request"]) -> list[list[int]]:
        """
        Get the concatenated input tokens for model forward pass
        """
        return sum([req.prompt_token_ids if req.output_len == 0 else req.output_token_ids[-1:] for req in reqs], [])


    @staticmethod
    def update_output(reqs: list["Request"], output_toks: list[int]) -> list["Request"]:
        """
        Update the output of a list of requests, requires the reqs are in the order that the output tokens are generated

        Returns the list of requests that are finished
        """
        assert len(reqs) == len(output_toks), f"Number of requests {len(reqs)} and output tokens {len(output_toks)} do not match"
        finished_reqs = []
        for req, tok in zip(reqs, output_toks):
            req.output_len += 1
            req.output_token_ids.append(tok)
            req.output_q.put_nowait(StepOutput(tok, req))
            if req.is_finished():
                req.finished_event.set()
                finished_reqs.append(req)
        return finished_reqs
            

    def __getstate__(self):
        """
        Get the state of the request for serialization, we only pass useful information
        """
        return {
            "prompt_token_ids": self.prompt_token_ids if self.output_len == 0 else [],
            "output_token_ids": self.output_token_ids[-1:] if self.output_len > 0 else [], 
            "prompt_len": self.prompt_len,
            "output_len": self.output_len,
            "request_id": self.request_id
        }
    
    
    def __setstate__(self, state):
        """
        Set the state of the request from the serialized state
        """
        self.prompt_token_ids = state["prompt_token_ids"]
        self.output_token_ids = state["output_token_ids"]
        self.prompt_len = state["prompt_len"]
        self.output_len = state["output_len"]
        self.request_id = state["request_id"]


def create_request(
    prompt_token_ids: list[int], 
    req_id: int, 
    output_token_ids: list[int] | None = None,
    quick_stop: bool = False
) -> Request:
    ret = Request(RawRequest("", 0))
    ret.prompt_token_ids = prompt_token_ids
    ret.output_token_ids = output_token_ids or []
    ret.prompt_len = len(ret.prompt_token_ids)
    ret.output_len = len(ret.output_token_ids)
    ret.max_output_len = ret.output_len + 1 if quick_stop else 10 ** 9
    ret.request_id = req_id
    return ret

class BatchPerfData:
    """
    Performance data for a batch
    """
    # pylint: disable=too-many-instance-attributes, missing-function-docstring
    def __init__(self, predictor: PerfPredictor):
        self.x = 0
        self.s = 0
        self.n_g = 0
        self.x_c = 0
        self.n_c = 0

        self.predictor = predictor
        self.pref_T = 0
        self.gdec_T = 0
        self.lnch_T = predictor.get_lnch_T()

    def add_pref(self, prompt_len):
        self.x += 1
        self.s += prompt_len
        self.pref_T += self.predictor.get_pref_T(prompt_len)
    
    def pop_pref(self, prompt_len):
        self.x -= 1
        self.s -= prompt_len
        self.pref_T -= self.predictor.get_pref_T(prompt_len)

    def add_gdec(self, seq_len):
        self.x += 1
        self.s += 1
        self.n_g += seq_len
        self.gdec_T = self.predictor.get_gdec_T(self.n_g)

    def add_cdec(self, seq_len):
        self.x += 1
        self.s += 1
        self.x_c += 1
        self.n_c += seq_len

    def pop_cdec(self, seq_len):
        self.x -= 1
        self.s -= 1
        self.x_c -= 1
        self.n_c -= seq_len

    @property
    def linr_T(self) -> float:
        return self.predictor.get_linr_T(self.s)
    
    @property
    def cdec_T(self) -> float:
        return self.predictor.get_cdec_T(self.x_c, self.n_c)
    
    @property
    def gpu_time(self) -> float:
        return self.linr_T + self.pref_T + self.gdec_T
    
    @property
    def cpu_time(self) -> float:
        return self.cdec_T + self.lnch_T

        

class SubBatch:
    """
    A sub-batch of requests
    """
    # pylint: disable=too-many-instance-attributes, missing-function-docstring
    def __init__(self, predictor: PerfPredictor=ZeroPerfPredictor()):
        self.gprf_reqs = []
        self.cprf_reqs = []
        self.gdec_reqs = []
        self.cdec_reqs = []
        self.perfdata = BatchPerfData(predictor)  
   
    def __len__(self):
        return self.perfdata.x

    def add_pref(self, req: Request, is_gpu: bool):
        if is_gpu:
            self.gprf_reqs.append(req)
        else:
            self.cprf_reqs.append(req)
        self.perfdata.add_pref(req.prompt_len)

    def pop_pref(self) -> Request:
        is_gpu = not self.cprf_reqs
        req = self.gprf_reqs.pop() if is_gpu else self.cprf_reqs.pop()
        self.perfdata.pop_pref(req.prompt_len)
        return req, is_gpu
        
    def add_gdec(self, req: Request):
        self.gdec_reqs.append(req)
        self.perfdata.add_gdec(req.seq_len)

    def add_cdec(self, req: Request):
        self.cdec_reqs.append(req)
        self.perfdata.add_cdec(req.seq_len)

    def pop_cdec(self):
        req = self.cdec_reqs.pop()
        self.perfdata.pop_cdec(req.seq_len)

    def get_num_prefs(self) -> int:
        return len(self.gprf_reqs) + len(self.cprf_reqs)

    def set_model_forward_args(self, model_config: LlamaModelConfig):
        """
        Set useful attributes for the model forward pass

        The comments indicate each attribute's usage in the model forward pass
        """
        # pylint: disable=attribute-defined-outside-init
        self.batch_size = self.perfdata.x # post-layer
        self.iter_width = self.perfdata.s # post-layer
        del self.perfdata

        self.num_cprfs = len(self.cprf_reqs)
        self.num_gprfs = len(self.gprf_reqs)
        self.num_gdecs = len(self.gdec_reqs)
        self.num_cdecs = len(self.cdec_reqs)
        self.num_prefs = self.num_cprfs + self.num_gprfs
        self.num_prgds = self.num_prefs + self.num_gdecs

        self.all_reqs = self.cprf_reqs + self.gprf_reqs + self.gdec_reqs + self.cdec_reqs
        assert all(req.request_id >= 0 for req in self.all_reqs), "Request ID not set"
        del self.cprf_reqs, self.gprf_reqs, self.gdec_reqs, self.cdec_reqs

        self.seq_ids_list = Request.get_ids(self.all_reqs)
        self.seq_lens_list = Request.get_lens(self.all_reqs)

        # Useful for attn kernels
        self.sum_pref_toks = sum(self.seq_lens_list[:self.num_prefs]) # store-pref-KV, pref, gdec
        self.sum_prgd_toks = self.sum_pref_toks + self.num_gdecs # gdec
        self.max_pref_toks = max(self.seq_lens_list[:self.num_prefs], default=0) # store-pref-KV, pref

        # Useful for paged attention
        sum_gdec_toks = sum(self.seq_lens_list[self.num_prefs:self.num_prgds])
        max_gdec_toks = max(self.seq_lens_list[self.num_prefs:self.num_prgds], default=0)
        seq_block_size = 2048
        num_kv_heads = model_config.num_kv_heads
        while num_kv_heads*(sum_gdec_toks/seq_block_size) < 1024 and seq_block_size//2 >= 64 and \
            max_gdec_toks / (seq_block_size//2) <= 128:
            seq_block_size //= 2
        self.seq_block_size = seq_block_size
        self.num_seq_blocks = (max_gdec_toks + seq_block_size - 1) // seq_block_size


    def print_profile(self):
        print(f"cprf lens: {[req.prompt_len for req in self.cprf_reqs]}, gprf lens: {[req.prompt_len for req in self.gprf_reqs]}, "
              f"gdec lens: {[req.seq_len for req in self.gdec_reqs]}, cdec lens: {[req.seq_len for req in self.cdec_reqs]}")
