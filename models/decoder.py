# beam search

import torch
import time

def beamSearchDecoder(data,k=5):

    sequences = [[list(),0.0]]
    for row in data:
        allCandidates = list()
        # expand each currrent candidate
        for i in range(len(sequences)):
            seq,score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score - torch.log(row[j])]
                allCandidates.append(candidate)
            ordred = sorted(allCandidates, key=lambda tup:tup[1])
            sequences = ordred[:k]
    return sequences

def main():
    data = torch.rand(90,50)
    s = time.time()
    for i in range(10):
        sequences = beamSearchDecoder(data,8)
    e = time.time()
    print('time: ',(e-s)/10)

if __name__ == '__main__':
    print('test run')
    main()
