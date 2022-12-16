import whisper
import datetime
import argparse

parser = argparse.ArgumentParser(description='Transcription using Whisper')
parser.add_argument('name', metavar='n', type=str,
                    help='file name')
parser.add_argument('output', metavar='o', type=str,
                    help='output file')

args = parser.parse_args()

def transcribe(name):
    model = whisper.load_model("base")

    result = model.transcribe(name)
    return result 

if __name__ == "__main__":
    output = transcribe(args.name)
    with open("./log/"+args.output) as f:
        f.write(output)

