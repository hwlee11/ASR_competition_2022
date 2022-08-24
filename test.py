import torch
import queue
import os
import math
import random
import warnings
import time
import json
import argparse
from glob import glob

from tools.preprocess import preprocessing
from tools.trainer import trainer
from tools.utils import (
    get_optimizer,
    get_criterion,
    get_lr_scheduler,
)
from tools.audio import (
    FilterBankConfig,
    MelSpectrogramConfig,
    MfccConfig,
    SpectrogramConfig,
)
#from tools.model import build_model
from tools.model_builder import build_conformer
from tools.vocab import KoreanSpeechVocabulary
from tools.data import split_dataset, collate_fn
from tools.utils import Optimizer
from tools.metrics import get_metric
from tools.inference import single_infer


from torch.utils.data import DataLoader


def bind_model(model, optimizer=None):
    def save(path, *args, **kwargs):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(path, 'model.pt'))
        print('Model saved')

    def load(path, *args, **kwargs):
        state = torch.load(os.path.join(path, 'model.pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Model loaded')

    # 추론
    def infer(path, **kwargs):
        return inference(path, model)


def inference(path, model, **kwargs):
    model.eval()

    results = []
    for i in glob(os.path.join(path, '*')):
        results.append(
            {
                'filename': i.split('/')[-1],
                'text': single_infer(model, i)[0]
            }
        )
    return sorted(results, key=lambda x: x['filename'])



if __name__ == '__main__':

    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    args.add_argument('--use_cuda', type=bool, default=True)
    args.add_argument('--seed', type=int, default=777)
    args.add_argument('--num_epochs', type=int, default=20)
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--save_result_every', type=int, default=10)
    args.add_argument('--checkpoint_every', type=int, default=1)
    args.add_argument('--print_every', type=int, default=50)
    args.add_argument('--dataset', type=str, default='kspon')
    args.add_argument('--output_unit', type=str, default='character')
    args.add_argument('--num_workers', type=int, default=8)
    args.add_argument('--num_threads', type=int, default=16)
    args.add_argument('--init_lr', type=float, default=1e-06)
    args.add_argument('--final_lr', type=float, default=1e-07)
    #args.add_argument('--peak_lr', type=float, default=1e-04)
    args.add_argument('--peak_lr', type=float, default= 0.025/math.sqrt(512))
    args.add_argument('--init_lr_scale', type=float, default=1e-02)
    args.add_argument('--final_lr_scale', type=float, default=0.001)
    args.add_argument('--max_grad_norm', type=int, default=400)
    args.add_argument('--warmup_steps', type=int, default=10000)
    args.add_argument('--decay_steps', type=int, default=80000)
    args.add_argument('--weight_decay', type=float, default=1e-06)
    args.add_argument('--reduction', type=str, default='mean')
    args.add_argument('--optimizer', type=str, default='adam')
    args.add_argument('--optimizer_betas', type=tuple, default=(0.9,0.98))
    args.add_argument('--optimizer_eps', type=float, default=1e-9)
    args.add_argument('--lr_scheduler', type=str, default='transformer_lr_scheduler')
    #args.add_argument('--lr_scheduler', type=str, default='tri_stage_lr_scheduler')
    args.add_argument('--total_steps', type=int, default=200000)

    args.add_argument('--architecture', type=str, default='conformer')
    args.add_argument('--use_bidirectional', type=bool, default=True)
    args.add_argument('--dropout', type=float, default=3e-01)
    #args.add_argument('--num_encoder_layers', type=int, default=3)
    args.add_argument('--hidden_dim', type=int, default=512)
    args.add_argument('--rnn_type', type=str, default='gru')
    args.add_argument('--max_len', type=int, default=400)
    args.add_argument('--activation', type=str, default='hardtanh')
    args.add_argument('--teacher_forcing_ratio', type=float, default=1.0)
    args.add_argument('--teacher_forcing_step', type=float, default=0.0)
    args.add_argument('--min_teacher_forcing_ratio', type=float, default=1.0)
    args.add_argument('--joint_ctc_attention', type=bool, default=False)

    #args.add_argument('--',default=)
    args.add_argument('--encoder_dim',type=int,default=512)
    args.add_argument('--decoder_dim',type=int,default=640)
    args.add_argument('--num_encoder_layers',type=int,default=17)
    args.add_argument('--num_decoder_layers',type=int,default=1)
    args.add_argument('--decoder_rnn_type',type=str,default="lstm")
    args.add_argument('--num_attention_heads',type=int,default=8)
    args.add_argument('--feed_forward_expansion_factor',type=int,default=4)
    args.add_argument('--conv_expansion_factor',type=int,default=2)
    args.add_argument('--input_dropout_p',type=float,default=0.1)
    args.add_argument('--attention_dropout_p',type=float,default=0.1)
    args.add_argument('--conv_dropout_p',type=float,default=0.1)
    args.add_argument('--decoder_dropout_p',type=float,default=0.1)
    args.add_argument('--feed_forward_dropout_p',type=float,default=0.1)
    args.add_argument('--conv_kernel_size',type=int,default=31)
    args.add_argument('--half_step_residual',type=str,default=True)
    args.add_argument('--decoder',type=str,default="rnnt")
    #args.add_argument('--decoder',type=str,default="None")

    args.add_argument('--audio_extension', type=str, default='pcm')
    args.add_argument('--transform_method', type=str, default='fbank')
    args.add_argument('--feature_extract_by', type=str, default='kaldi')
    args.add_argument('--sample_rate', type=int, default=16000)
    args.add_argument('--frame_length', type=int, default=25)
    args.add_argument('--frame_shift', type=int, default=10)
    args.add_argument('--n_mels', type=int, default=80)
    args.add_argument('--freq_mask_para', type=int, default=18)
    args.add_argument('--time_mask_num', type=int, default=4)
    args.add_argument('--freq_mask_num', type=int, default=2)
    args.add_argument('--normalize', type=bool, default=True)
    args.add_argument('--del_silence', type=bool, default=True)
    args.add_argument('--spec_augment', type=bool, default=True)
    args.add_argument('--input_reverse', type=bool, default=False)

    config = args.parse_args()
    warnings.filterwarnings('ignore')

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = 'cuda' if config.use_cuda == True else 'cpu'
    if hasattr(config, "num_threads") and int(config.num_threads) > 0:
        torch.set_num_threads(config.num_threads)

    #wmpVocabulary(os.path.join(os.getcwd(),'transcripts.txt'))
    vocab = KoreanSpeechVocabulary(os.path.join(os.getcwd(), 'labels.csv'), output_unit='character')

    if config.architecture == 'deepspeech2':
        model = build_model(config, vocab, device)
    elif config.architecture == 'conformer':
        model = build_conformer(config,vocab,device)

    #optimizer = get_optimizer(model, config)
    #bind_model(model, optimizer=optimizer)

    #metric = get_metric(metric_name='CER', vocab=vocab)

    dummy_input = torch.rand(2,230,80).to('cuda:0')
    dummy_input_lengths = torch.randint(100,230,(2,)).to('cuda:0')
    dummy_labels = torch.randint(32,(2,64)).to('cuda:0')
    dummy_label_lengths = torch.randint(16,32,(2,)).to('cuda:0')
    print(dummy_input,dummy_input_lengths)
    print(dummy_labels,dummy_label_lengths)

    criterion = get_criterion(config, vocab)

    outputs = model(dummy_input,dummy_input_lengths,dummy_labels,dummy_label_lengths)
    print(outputs.size())
    print(dummy_labels[:,1:].size())
    print(dummy_input_lengths.size())
    print(dummy_label_lengths.size())

    loss = criterion(
                outputs,
                #targets[:, 1:].contiguous().int(),
                dummy_labels[:, 1:].contiguous().int(),
                #targets[:, 1:].transpose(0, 1).contiguous().int(),
                dummy_input_lengths.int(),
                dummy_label_lengths.int()
    )
    loss.backward()


    '''
            model, train_loss, train_cer = trainer(
                'train',
                config,
                train_loader,
                optimizer,
                model,
                criterion,
                metric,
                train_begin_time,
                device
            )
    '''
