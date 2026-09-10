import os
import pandas as pd
import torchaudio
import torch
import soundfile as sf

from .utils import (
    compute_span_masking,
    sample_negative_indices,
    compute_sub_attention_mask,
    Wav2Vec2Config
)

from .utils import get_audio_duration

from transformers import Wav2Vec2CTCTokenizer
from torch.utils.data import Dataset, DataLoader

class LibriSpeechDataset(Dataset):
    def __init__(
        self,
        path_to_data_root='./dataset',
        include_splits=['dev'],
        max_audio_duration=20.0,
        min_audio_duration=2.0,
        sampling_rate=160000,
        num_audio_channels=1,
        truncate_audio=True,
        return_transcript=True,
        hf_model_name="facebook/wav2vec2-base"
    ):

        if isinstance(include_splits, str):
            include_splits = [include_splits]

        self.sampling_rate = sampling_rate
        self.return_transcript = return_transcript
        self.truncate_audio = truncate_audio
        self.num_audio_channels = num_audio_channels
        self.min_audio_samples = int(min_audio_duration * sampling_rate)        
        self.max_audio_samples = int(max_audio_duration * sampling_rate)
        self.librespeechdata = []
        for split in include_splits:
            path_to_split = os.path.join(path_to_data_root, split)
            
            for speaker in os.listdir(path_to_split):
                if ".DS_Store" in speaker:
                    continue
                path_to_speaker = os.path.join(path_to_split, speaker)
                
                for section in os.listdir(path_to_speaker):
                    path_to_section = os.path.join(path_to_speaker, section)
                    
                    if ".DS_Store" in section:
                        continue
                    
                    files = os.listdir(path_to_section)
                    # there will be a single file which have all the transcript of files
                    transcript_file = [path for path in files if ".txt" in path][0]
                    transcript_file = os.path.join(path_to_section, transcript_file)
                    
                    
                    with open(transcript_file, "r") as f:
                        transcripts = f.readlines()
                    
                    for i in transcripts:
                        i = i.split(' ', maxsplit=1)
                        audio_root = i[0]
                        audio_file = audio_root + ".flac"
                        audio_transcript = i[1].strip()    
                        path_to_audio_file = os.path.join(path_to_section, audio_file)
                        duration = get_audio_duration(path_to_audio_file)
                        
                        if (duration >= min_audio_duration and duration <= max_audio_duration or self.truncate_audio):
                            self.librespeechdata.append((path_to_audio_file, audio_transcript))


        if return_transcript:
            self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(hf_model_name)
            

    def __len__(self):
        return len(self.librespeechdata)

    def __getitem__(self, idx):
        path_to_audio, transcript = self.librespeechdata[idx]
        audio, sr = sf.read(path_to_audio)
        audio = torch.from_numpy(audio).unsqueeze(0).float()
        
        if self.truncate_audio:
            audio = audio[:, :self.max_audio_samples]

        # sanity check for sampling rate
        if sr != self.sampling_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sampling_rate)
        
        # audio channels should be 1
        if self.num_audio_channels != 2:
            audio = audio.squeeze()
        
        
        
        # normalize 
        audio = (audio - audio.mean()) / (audio.std() + 1e-7)
        
        if self.return_transcript:
            tokenized_transcript = torch.tensor(self.tokenizer.encode(transcript))
            
            batch = {
                "input_values" : audio,
                "labels" : tokenized_transcript
            }
        
        else:
            batch = {
                "input_values" : audio
            }
            
        return batch
            



def Wav2Vec2CollateFunctionForPretraining(config: Wav2Vec2Config):
    
    def collate_fun(batch):
        batch_audios = [sample['input_values'] for sample in batch]
        attention_mask = [torch.ones(len(audio)) for audio in batch_audios]
        batch_audios = torch.nn.utils.rnn.pad_sequence(batch_audios, batch_first=True, padding_value=0.0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0.0)
        
        sub_attention_mask = compute_sub_attention_mask(config, attention_mask)
        
        span_mask = compute_span_masking(
            shape=sub_attention_mask.shape,
            mask_prob=config.masking_probability,
            mask_length=config.masking_span_length,
            min_masks=config.minimum_spans,
            sub_attention_mask=sub_attention_mask
        )
        
        sampled_negatives = sample_negative_indices(
            feature_shape=sub_attention_mask.shape,
            num_negatives=config.num_negatives,
            mask_time_indices=span_mask
        )
        
        batch = {
            "input_values": batch_audios,
            "attention_mask": attention_mask,
            "sub_attention_mask": sub_attention_mask,
            "mask_time_indices": span_mask,
            "sampled_negatives": sampled_negatives
        }
        
        return batch
        
    return collate_fun
    
    



if __name__ == "__main__":
    config = Wav2Vec2Config()
    dataset = LibriSpeechDataset(include_splits="dev")
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=Wav2Vec2CollateFunctionForPretraining(config))
    
    sample = next(iter(dataloader))
    print(sample) 