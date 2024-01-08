from unifold.launching.mmseqs_api import MMSeqs2API
import argparse
import logging
import os

logger = logging.getLogger("run_mmseqs2_api")
logger.setLevel(logging.INFO)

def main(args):
    api = MMSeqs2API(logger=logger, use_tqdm=True)
    query = open(args.fasta).read()
    api.query(
        query,
        args.output_tgz,
        args.use_pair,
        args.max_retries,
        allow_rewrite=True,
    )
    os.system("mkdir -p test_out")
    os.system(f"tar zxvf {args.output_tgz} -C test_out")
    api.get_templates(
        "test_out/pdb70.m8",
        "test_out/templs"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fasta", type=str, required=True, help="input fasta path")
    parser.add_argument("-o", "--output-tgz", type=str, required=True, help="output tar.gz path")
    parser.add_argument("-p", "--use-pair", action="store_true", help="use msa pairing mode if set")
    parser.add_argument("--max-retries", type=int, default=3, help="number of max retry times")

    args = parser.parse_args()
    main(args)
