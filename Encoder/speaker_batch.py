import numpy as np
from typing import List
from speaker import Speaker


class SpeakerBatch:
    def __init__(self, speakers: List[Speaker], utterances_per_speaker: int, n_frames: int):
        self.speakers = speakers
        self.partials = {s: s.random_partial(utterances_per_speaker, n_frames) for s in speakers}
        
        self.data = np.array([frames for s in speakers for _, frames, _ in self.partials[s]])
