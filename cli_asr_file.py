import asr
import argparse
import json


def parse_arguments():
    """
    Command line argument parser
    Returns:
        parser: Argument Parser
    """
    asr_cli = argparse.ArgumentParser(description="Google ASR from file CLI")
    asr_cli.add_argument("-i", "--input_file", help="Input audio file")
    return asr_cli.parse_args()


def load_conf_file(path):
    with open(path) as f:
        conf = json.load(f)
    return conf


def main():
    conf = load_conf_file('config.json')
    args = parse_arguments()
    
    asr_results, data = asr.audio_to_asr_text(args.input_file,
                                              conf['google_credentials'])
    print(asr_results)


if __name__ == "__main__":
    main()
