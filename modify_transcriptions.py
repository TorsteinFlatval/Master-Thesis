def modify_transcriptions(transcriptions):
    modified_transcriptions = []
    for transcription in transcriptions:
        words = transcription.split()
        for i, word in enumerate(words):
            if word == "mm":
                words[i] = word.replace('mm', 'mmm')
            elif word == "ee":
                words[i] = word.replace('ee', 'eee')
            elif word == "q":
                words[i] = word.replace('q', 'qqq')
            elif word == "qq":
                words[i] = word.replace('qq', 'qqq')
            elif word == "qqqq":
                words[i] = word.replace('qqqq', 'qqq')
        modified_transcription = ' '.join(words)
        modified_transcriptions.append(modified_transcription)
    return modified_transcriptions
