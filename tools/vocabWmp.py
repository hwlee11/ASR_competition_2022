import sentencepiece as spm


def load_dataset(transcripts_path):
    """
    Provides dictionary of filename and labels

    Args:
        transcripts_path (str): path of transcripts

    Returns: target_dict
        - **target_dict** (dict): dictionary of filename and labels
    """
    audio_paths = list()
    transcripts = list()

    with open(transcripts_path) as f:
        for idx, line in enumerate(f.readlines()):
            try:
                audio_path, korean_transcript, transcript = line.split('\t')
            except:
                print(line)
            transcript = transcript.replace('\n', '')

            audio_paths.append(audio_path)
            transcripts.append(transcript)

    return audio_paths, transcripts


def wmp(transcripts_path):
    #audio_paths,transcripts = load_dataset(transcripts_path)

    transcripts = []
    f = open(transcripts_path,'r')
    while True:
        line = f.readline()
        if line == "":
            break
        transcripts.append(line.replace('\n',''))
    f.close()


    f = open('spm_input.txt','w',encoding='utf-8')
    for sent in transcripts:
        f.write('{}\n'.format(sent))
    f.close()
    #train

    spm.SentencePieceTrainer.train(input='spm_input.txt',model_prefix='asr',vocab_size=110)

    #sp = spm.SentencePieceProcessor()
    #sp.Load('{}.model'.format('asr'))
    #sp.EncodeAsPieces()


if __name__ == '__main__':
    wmp('test.txt')
