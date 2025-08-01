from model import  WaveRNN
from audio import *


def gen_testset(model: WaveRNN, test_set, samples, batched, target, overlap, save_path):
    k = model.get_step() // 1000

    for i, (m, x) in enumerate(test_set, 1):
        if i > samples: 
            break

        print('\n| Generating: %i/%i' % (i, samples))

        x = x[0].numpy()

        bits = 9

        
        x = decode_mu_law(x, 2**bits, from_labels=True)
        

        save_wav(x, save_path.joinpath("%dk_steps_%d_target.wav" % (k, i)))
        
        batch_str = "gen_batched_target%d_overlap%d" % (target, overlap) if batched else \
            "gen_not_batched"
        save_str = save_path.joinpath("%dk_steps_%d_%s.wav" % (k, i, batch_str))

        wav = model.generate(m, batched, target, overlap, True)
        save_wav(wav, save_str)

