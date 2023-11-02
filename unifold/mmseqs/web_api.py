import requests
from tqdm import tqdm
import time
import logging
import os
import urllib3

TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'
DEFAULT_API_SERVER = "https://api.colabfold.com"

class MMSeqs2ServerError(Exception):
    pass

class FileExistError(Exception):
    pass

class MMSeqs2API:
    def __init__(self, *, host_url=DEFAULT_API_SERVER, logger=None, use_tqdm=False, max_retries=3, sleep_secs=1) -> None:
        self.host_url = host_url
        self.logger = logger
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.WARNING)
        self.use_tqdm = use_tqdm
        self.max_retries = max_retries
        self.sleep_secs = sleep_secs
    
    def _request(self, query, endpoint, mode):
        res = requests.post(f'{self.host_url}/{endpoint}', data={'q':query,'mode': mode})
        try:
            out = res.json()
        except ValueError:
            self.logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status":"ERROR"}
        except urllib3.exceptions.NewConnectionError as ex:
            self.logger.error(f"Cannot establish connection to the server. Message:")
            self.logger.error(str(ex))
            out = {"status": "ERROR"}
        return out

    def _query_status(self, mmseqs_job_id):
        try:
            res = requests.get(f'{self.host_url}/ticket/{mmseqs_job_id}')
            out = res.json()
        except ValueError:
            self.logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status":"ERROR"}
        except urllib3.exceptions.NewConnectionError as ex:
            self.logger.error(f"Cannot establish connection to the server. Message:")
            self.logger.error(str(ex))
            out = {"status": "ERROR"}
        return out

    def _get_result(self, mmseqs_job_id, output_path):
        try:
            res = requests.get(f'{self.host_url}/result/download/{mmseqs_job_id}')
        except urllib3.exceptions.NewConnectionError as ex:
            self.logger.error(f"Cannot establish connection to the server. Message:")
            self.logger.error(str(ex))
            return {"status": "ERROR"}
        with open(output_path, "wb") as f:
            f.write(res.content)
            return {"status": "SUCCEED"}

    def _sleep(self, pbar=None):
        time.sleep(self.sleep_secs)
        if pbar is not None:
            pbar.update(self.sleep_secs)

    def query(
        self,
        fasta_string: str,              # must be cleaned
        output_filename: str,           # outputs as tar.gz
        use_pairing: bool = False,
        max_retries: int = None,
        allow_rewrite: bool = False,
    ):
        if os.path.exists(output_filename) and not allow_rewrite:
            raise FileExistError(f"{output_filename} exists. If you want to rewrite it, please set allow_rewrite=True.")

        os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)
        endpoint = "ticket/pair" if use_pairing else "ticket/msa"
        mode = "" if use_pairing else "env"
        time_est = len(fasta_string) // 5    # roughly linear to the total length of the query.
        with tqdm(total=time_est, bar_format=TQDM_BAR_FORMAT, disable=(not self.use_tqdm)) as pbar:
            n_retry = 0
            while n_retry < max_retries:
                n_retry += 1
                pbar.set_description("SUBMIT")
                out = self._request(fasta_string, endpoint, mode)
                # resubmit circle
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    self._sleep(pbar)
                    print(f"Waiting for resubmit. Reason: {out['status']}")
                    out = self._request(fasta_string, endpoint, mode)
                
                if out["status"] == "ERROR":
                    self.logger.error(f'MMseqs2 API is giving errors. (retry {n_retry})')
                    self._sleep(pbar)
                    continue

                if out["status"] == "MAINTENANCE":
                    self.logger.error(f'MMseqs2 API is undergoing maintenance. (retry {n_retry})')
                    self._sleep(pbar)
                    continue

                mmseqs_job_id, status = out["id"], out["status"]
                while status in ["UNKNOWN","RUNNING","PENDING"]:
                    pbar.set_description(status)
                    self._sleep(pbar)
                    out = self._query_status(mmseqs_job_id)
                    status = out["status"]

                if status == "ERROR":
                    self.logger.error(f'MMseqs2 API is giving errors. (retry {n_retry})')
                    self._sleep(pbar)
                    continue
                
                if status == "COMPLETE":
                    pbar.set_description("DOWNLOAD")
                    out = self._get_result(mmseqs_job_id, output_filename)
                    if out["status"] == "ERROR":
                        self.logger.error(f"File downloading failed. (retry {n_retry})")
                        self._sleep(pbar)
                        continue
                    elif out["status"] == "SUCCEED":
                        self.logger.info("Results from MMSeqs2 API successfully downloaded.")
                        return
        
        raise MMSeqs2ServerError(f"Cannot correctly obtain results from MMSeq2 server after {n_retry} retries.")

