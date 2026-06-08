import sys, json, torch, numpy as np, soundfile as sf
torch.set_num_threads(6)
wav, out_json = sys.argv[1], sys.argv[2]
model, utils = torch.hub.load('snakers4/silero-vad','silero_vad',trust_repo=True)
get_speech_timestamps = utils[0]
d, sr = sf.read(wav, dtype='int16'); 
if d.ndim>1: d=d[:,0]
d = d.astype(np.float32)/32768.0
ts = get_speech_timestamps(torch.from_numpy(d), model, sampling_rate=16000)
# segments in samples
segs = [[int(t['start']), int(t['end'])] for t in ts]
speech_s = sum(e-s for s,e in segs)/16000.0
json.dump({"sr":16000,"n_samples":len(d),"segments":segs}, open(out_json,"w"))
print(f"{wav}: {len(segs)} speech segs, speech {speech_s:.1f}s / {len(d)/16000:.1f}s total ({100*speech_s*16000/len(d):.0f}% speech)")
