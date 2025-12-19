# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re

from lhotse.serialization import get_aistore_client
import torch
import torch.utils.data
import torchaudio

from lhotse import CutSet, MonoCut, Recording, Seconds, SupervisionSegment, compute_num_frames
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_audio, collate_vectors
from lhotse.utils import ifnone

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.utils import logging
from nemo.collections.common.data.lhotse.text_adapters import Formattable

COT_RE = re.compile(r'(?s)<cot>(.*?)</cot>')
CL_RE  = re.compile(r'(?s)<cl_?cot>(.*?)</cl_?cot>')

def parse_three_segments(text: str):
    """
    返回 (cot, cl_cot, resp)
    - 若某段缺失则返回空串
    - 不要求 <cot> 与 <cl_cot> 的先后顺序
    - 允许 </cl_cot> 或 </clcot> 这类小写变体（可按需放宽）
    """
    cot_match = COT_RE.search(text)
    cl_match  = CL_RE.search(text)

    cot = cot_match.group(1).strip() if cot_match else ""
    cl  = cl_match.group(1).strip() if cl_match else ""

    # 从原文中移除已匹配片段，剩余即 response
    resp = text
    if cot_match:
        resp = resp[:cot_match.start()] + resp[cot_match.end():]
    if cl_match:
        # 注意重新在当前 resp 上定位并移除 cl 片段
        tmp = CL_RE.search(resp)
        if tmp:
            resp = resp[:tmp.start()] + resp[tmp.end():]
    resp = resp.strip()
    return cot, cl, resp


class DuplexS2SDataset(torch.utils.data.Dataset):
    """
    A dataset for duplex speech-to-speech models that handles bidirectional conversations.

    This dataset processes Lhotse CutSet objects containing recordings with supervision segments
    from different speakers (roles). It creates aligned representations of audio and text for
    both source (input) and target (output) channels, preserving temporal alignment between
    audio frames and text tokens.

    Args:
        tokenizer (TokenizerSpec):
            Tokenizer for converting text to token IDs and vice versa. Must support BOS and EOS tokens.
            It's expected to support PAD token as well, otherwise we will use 0 as the pad token
            and emit a warning.

        frame_length (Seconds):
            Duration of a single frame in seconds. Used to calculate frame positions for token alignment.

        source_sample_rate (int):
            Sample rate for source audio (e.g., 16000 Hz).

        target_sample_rate (int):
            Sample rate for target audio (e.g., 22050 Hz).

        input_roles (list[str], optional):
            List of speaker roles (cut.supervisions[:].speaker) to consider as inputs. Defaults to ["user"].

        output_roles (list[str], optional):
            List of speaker roles (cut.supervisions[:].speaker) to consider as outputs. Defaults to ["agent"].

    Returns:
        A dictionary with the following keys:
            - source_audio: Tensor of source waveform samples [B, T]
            - source_audio_lens: Tensor of source audio lengths [B]
            - target_audio: Tensor of target waveform samples [B, T]
            - target_audio_lens: Tensor of target audio lengths [B]
            - target_tokens: Tensor of target text tokens [B, T], with special tokens (BOS/EOS/PAD)
                at positions aligned with audio frames
            - target_token_lens: Tensor of target token sequence lengths [B]
            - source_tokens: Tensor of source text tokens [B, T], with special tokens (BOS/EOS/PAD)
                at positions aligned with audio frames
            - source_token_lens: Tensor of source token sequence lengths [B]
            - target_texts: List of full target texts joined from output_roles supervisions [B]

    Notes:
        - The dataset ensures frame-level alignment between audio and text by inserting tokens at
          specific frame positions based on the timing of supervision segments.
        - PAD tokens (typically 0) are used to fill gaps where there's no text.
        - BOS tokens mark the beginning of each speech segment.
        - EOS tokens mark the end of each speech segment.
        - Text tokens from each speaker are placed at frame positions corresponding to their
          timestamp in the original recording, preserving the temporal relationship.
          This is a segment-level alignment only, not word-level alignment.
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        frame_length: Seconds,
        source_sample_rate: int,
        target_sample_rate: int,
        input_roles: list[str] = None,
        output_roles: list[str] = None,
        aug_by_swap_role: bool = False,
        loss_mask_forcot: float = 1,
        loss_mask_forres: float = 1,
    ):
        self.tokenizer = tokenizer
        self.frame_length = frame_length
        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.input_roles = set(ifnone(input_roles, ["user"]))
        self.output_roles = set(ifnone(output_roles, ["agent"]))
        self.aug_by_swap_role = aug_by_swap_role
        self.loss_mask_forcot = loss_mask_forcot
        self.loss_mask_forres = loss_mask_forres
        print('loss_mask_forcot', self.loss_mask_forcot)
        print('loss_mask_forres', self.loss_mask_forres)
        
        assert tokenizer.bos is not None, "BOS support in the tokenizer is required for S2S models."
        assert tokenizer.eos is not None, "EOS support in the tokenizer is required for S2S models."

        # print('bos', tokenizer.eos)
        # print('pad_id', tokenizer.pad)

    def __getitem__(self, all_cuts: CutSet) -> dict:
        # audio mini-batch
        cuts = all_cuts.filter(lambda c: isinstance(c, Cut))
        audio_data = None

        if cuts:
            cuts = cuts.transform_text(_strip_timestamps)

            swapped_cuts = []

            if self.aug_by_swap_role:
                for cut in cuts:
                    total_turns = cut.custom.get('total_turns', len(cut.supervisions))

                    if total_turns > 4 and total_turns % 2 == 0:
                        swapped_cut = self._create_role_swapped_cut(cut)
                        if swapped_cut:
                            swapped_cuts.append(swapped_cut)

            if swapped_cuts:
                all_cuts_combined = CutSet.from_cuts(list(cuts) + swapped_cuts)
            else:
                all_cuts_combined = cuts

            source_audio, source_audio_lens = collate_audio(all_cuts_combined.resample(self.source_sample_rate))
            target_audio, target_audio_lens = collate_audio(
                all_cuts_combined.resample(self.target_sample_rate), recording_field="target_audio"
            )
            target_tokens, target_token_lens, loss_mask = collate_target_token_channel(
                all_cuts_combined, self.tokenizer, self.frame_length, roles=self.output_roles, 
                loss_mask_forcot = self.loss_mask_forcot,
                loss_mask_forres = self.loss_mask_forres,
            )
            source_tokens, source_token_lens = collate_token_channel(
                all_cuts_combined, self.tokenizer, self.frame_length, roles=self.input_roles
            )

            try:
                target_first_turn_audio, target_first_turn_audio_lens = collate_first_turn_audio(
                    all_cuts_combined.resample(self.target_sample_rate), roles=self.output_roles,
                    recording_field="target_audio"
                )
            except Exception as e:
                target_first_turn_audio = None
                target_first_turn_audio_lens = None

            audio_data = {
                "sample_id": [str(cut.id) for cut in all_cuts_combined],
                "source_audio": source_audio,
                "source_audio_lens": source_audio_lens,
                "target_audio": target_audio,
                "target_audio_lens": target_audio_lens,
                "target_tokens": target_tokens,
                "target_token_lens": target_token_lens,
                "source_tokens": source_tokens,
                "source_token_lens": source_token_lens,
                "target_texts": [
                    " ".join(s.text for s in cut.supervisions if s.speaker in self.output_roles)
                    for cut in all_cuts_combined
                ],
                "target_first_turn_audio": target_first_turn_audio,
                "target_first_turn_audio_lens": target_first_turn_audio_lens,
                "formatter": [getattr(cut, "formatter", "s2s_duplex") for cut in all_cuts_combined],
                "aug_by_noise": [getattr(cut, "aug_by_noise", True) for cut in all_cuts_combined],
                "loss_mask": loss_mask
            }

        text_cuts = all_cuts.filter(lambda c: isinstance(c, Formattable))
        text_data = None
        if text_cuts:
            text_tokens = []
            text_token_lens = []
            for c in text_cuts:
                text_ids = c.input_ids
                text_tokens.append(text_ids)
                text_token_lens.append(text_ids.shape[0])

            text_tokens = collate_vectors(
                text_tokens, padding_value=get_pad_id(self.tokenizer)
            )
            text_token_lens = torch.tensor(text_token_lens, dtype=torch.long)
            text_data = {
                "text_tokens": text_tokens,
                "text_token_lens": text_token_lens,
            }

        return {
            "audio_data": audio_data,
            "text_data": text_data,
        }

    def _create_role_swapped_cut(self, cut):

        from lhotse import AudioSource
        from io import BytesIO
        import soundfile as sf
        import numpy as np

        swapped_supervisions = []
        for sup in cut.supervisions:
            if sup.speaker == 'User':
                new_speaker = 'Assistant'
            elif sup.speaker == 'Assistant':
                new_speaker = 'User'
            else:
                continue

            swapped_sup = SupervisionSegment(
                id=sup.id + "_swapped",
                recording_id=sup.recording_id,
                start=sup.start,
                duration=sup.duration,
                channel=sup.channel,
                text=sup.text,
                language=sup.language,
                speaker=new_speaker,
                gender=sup.gender,
                custom=sup.custom,
                alignment=sup.alignment
            )
            swapped_supervisions.append(swapped_sup)

        swapped_supervisions = sorted(swapped_supervisions, key=lambda s: s.start)

        first_agent_idx = None
        last_user_idx = None

        for i, sup in enumerate(swapped_supervisions):
            if sup.speaker == 'Assistant' and first_agent_idx is None:
                first_agent_idx = i
            if sup.speaker == 'User':
                last_user_idx = i

        filtered_supervisions = []
        for i, sup in enumerate(swapped_supervisions):
            if i != first_agent_idx and i != last_user_idx:
                filtered_supervisions.append(sup)

        if not filtered_supervisions:
            return None

        first_remaining_start = filtered_supervisions[0].start
        last_remaining_end = max(s.start + s.duration for s in filtered_supervisions)
        new_duration = last_remaining_end - first_remaining_start

        adjusted_supervisions = []
        for sup in filtered_supervisions:
            adjusted_sup = SupervisionSegment(
                id=sup.id,
                recording_id=sup.recording_id,
                start=sup.start - first_remaining_start,  # 减去offset
                duration=sup.duration,
                channel=sup.channel,
                text=sup.text,
                language=sup.language,
                speaker=sup.speaker,
                gender=sup.gender,
                custom=sup.custom,
                alignment=sup.alignment
            )
            adjusted_supervisions.append(adjusted_sup)

        total_duration = max(s.start + s.duration for s in adjusted_supervisions)
        total_samples = int(total_duration * cut.sampling_rate)

        new_source_audio = np.zeros(total_samples, dtype=np.float32)
        new_target_audio = np.zeros(total_samples, dtype=np.float32)

        for sup in adjusted_supervisions:
            start_sample = int(sup.start * cut.sampling_rate)
            end_sample = int((sup.start + sup.duration) * cut.sampling_rate)

            if sup.speaker == 'User':

                original_start = sup.start + first_remaining_start
                agent_audio = cut.custom['target_audio'].to_cut().truncate(
                    offset=original_start,
                    duration=sup.duration
                ).load_audio()
                if len(agent_audio.shape) > 1:
                    agent_audio = agent_audio.squeeze()
                actual_end = min(end_sample, start_sample + len(agent_audio))
                new_source_audio[start_sample:actual_end] = agent_audio[:actual_end - start_sample]

            elif sup.speaker == 'Assistant':
                original_start = sup.start + first_remaining_start
                user_audio = cut.recording.to_cut().truncate(
                    offset=original_start,
                    duration=sup.duration
                ).load_audio()
                if len(user_audio.shape) > 1:
                    user_audio = user_audio.squeeze()
                actual_end = min(end_sample, start_sample + len(user_audio))
                new_target_audio[start_sample:actual_end] = user_audio[:actual_end - start_sample]

        source_buffer = BytesIO()
        sf.write(source_buffer, new_source_audio, cut.sampling_rate, format='wav')
        source_buffer.seek(0)

        new_source_recording = Recording(
            id=f"{cut.id}_swapped_source",
            sampling_rate=cut.sampling_rate,
            num_samples=len(new_source_audio),
            duration=total_duration,
            sources=[AudioSource(
                type="memory",
                channels=[0],
                source=source_buffer.getvalue()
            )]
        )

        target_buffer = BytesIO()
        sf.write(target_buffer, new_target_audio, cut.sampling_rate, format='wav')
        target_buffer.seek(0)

        new_target_recording = Recording(
            id=f"{cut.id}_swapped_target",
            sampling_rate=cut.sampling_rate,
            num_samples=len(new_target_audio),
            duration=total_duration,
            sources=[AudioSource(
                type="memory",
                channels=[0],
                source=target_buffer.getvalue()
            )]
        )

        swapped_cut = MonoCut(
            id=f"{cut.id}_swapped",
            start=0,
            duration=total_duration,
            channel=0,
            supervisions=adjusted_supervisions,
            recording=new_source_recording,
            custom={
                **cut.custom,
                'total_turns': len(adjusted_supervisions),
                'role_swapped': True,
                'target_audio': new_target_recording,
            }
        )

        return swapped_cut


def collate_first_turn_audio(
    cuts: CutSet,
    roles: set[str],
    recording_field: str = "target_audio",
) -> tuple[torch.Tensor, torch.Tensor]:
    first_turn_audios = []
    first_turn_audios_lens = []
    for cut in cuts:
        first_supervision = [s for s in cut.supervisions if s.speaker in roles][0]
        truncated_audio = cut.truncate(offset=max(0, first_supervision.start), 
                                        duration=first_supervision.duration).load_custom(recording_field)
        first_turn_audios.append(truncated_audio.squeeze(0))
        first_turn_audios_lens.append(truncated_audio.shape[-1])

    return collate_vectors(first_turn_audios, padding_value=0), torch.tensor(first_turn_audios_lens)

def collate_target_token_channel(
    cuts: CutSet,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
    loss_mask_forcot: float = 1,
    loss_mask_forres: float = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = get_pad_id(tokenizer)
    tokens = []
    loss_masks = []
    for c in cuts:
        get_token, get_mask = build_target_token_channel(c, 
                            tokenizer=tokenizer, frame_length=frame_length, roles=roles, 
                            pad_id=pad_id,
                            loss_mask_forcot = loss_mask_forcot,
                            loss_mask_forres = loss_mask_forres,
                            )
        tokens.append(get_token)
        loss_masks.append(get_mask)
    token_lens = torch.tensor([len(tt) for tt in tokens])
    tokens = collate_vectors(tokens, padding_value=pad_id)
    loss_masks = collate_vectors(loss_masks, padding_value=1)
    return tokens, token_lens, loss_masks


def collate_token_channel(
        cuts: CutSet,
        tokenizer: TokenizerSpec,
        frame_length: Seconds,
        roles: set[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = get_pad_id(tokenizer)
    tokens = [
        build_token_channel(c, tokenizer=tokenizer, frame_length=frame_length, roles=roles, pad_id=pad_id)
        for c in cuts
    ]
    token_lens = torch.tensor([len(tt) for tt in tokens])
    tokens = collate_vectors(tokens, padding_value=pad_id)
    return tokens, token_lens


def build_token_channel(
    cut: Cut,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
    pad_id: int = -1,
) -> torch.Tensor:
    diagnostic = f"Extra info: {cut.id=}"
    if getattr(cut, "shard_origin", None) is not None:
        diagnostic = f"{diagnostic} {cut.shard_origin=}"

    total = compute_num_frames(cut.duration, frame_length, cut.sampling_rate)
    tokens = torch.ones(total, dtype=torch.long) * pad_id

    for supervision in cut.supervisions:
        if supervision.speaker in roles:
            get_full_text = supervision.text  #<cot>xxx</cot><cl_cot>xxx</cl_cot>(response)
            get_cot, get_clcot, get_res_text = parse_three_segments(get_full_text)
            get_res_text = get_res_text.strip().strip('*').strip()
            # print(get_res_text)
            text_ids = torch.as_tensor([tokenizer.bos] + tokenizer.text_to_ids(get_res_text))

            # Determine the frame offset for the start of the supervision to insert the text tokens.
            pos = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
            if pos >= len(tokens):
                logging.warning(
                    f"Ill-constructed example: the beginning offset of a supervision {pos} is larger than the example's length {len(tokens)}. {diagnostic}"
                )
                continue

            eospos = compute_num_frames(supervision.end, frame_length, cut.sampling_rate)

            available_frames_for_text = eospos - pos

            if available_frames_for_text > 0 and len(text_ids) > available_frames_for_text:
                # Truncate text_ids to fit before the eos position.
                text_ids = text_ids[:available_frames_for_text]
            elif available_frames_for_text <= 0:
                # If there's no space for text (e.g., start >= end), use an empty sequence.
                text_ids = torch.tensor([], dtype=torch.long)

            # Determine the frame offset for the last non-EOS text token to form a valid range for insertion;
            # Note that EOS will be placed possibly much later, at the frame that coincides with end of speech,
            # rather than end of text. The gap between last non-EOS token and EOS token will be filled with `pad_id`.
            endpos = pos + len(text_ids)
            if endpos > len(tokens):
                trunc_len = len(tokens) - pos
                logging.warning(
                    f"Truncating training example's text_ids of length {len(text_ids)} by {trunc_len} because {endpos=} > {len(tokens)=}. {diagnostic}"
                )
                text_ids = text_ids[:trunc_len]
                endpos = pos + len(text_ids)

            try:
                tokens[pos:endpos] = text_ids
            except Exception as e:
                raise RuntimeError(f"{tokens.shape=} {pos=} {endpos=} {text_ids.shape=} {diagnostic}") from e
            # Insert EOS at the end of the supervision segment.
            if eospos < len(tokens):
                # Normal case: place EOS at the intended position
                tokens[eospos] = tokenizer.eos
            else:
                # Interruption case: place EOS at the last valid position
                # This ensures the model learns to stop when interrupted by user
                if endpos < len(tokens):
                    # Case 1: text finished, interrupted during sil/audio generation
                    # Place EOS right after the last text token (or at sequence end if closer)
                    actual_eos_pos = min(endpos, len(tokens) - 1)
                    tokens[actual_eos_pos] = tokenizer.eos
                elif len(tokens) > 0:
                    # Case 2: text truncated due to interruption
                    # Place EOS at the very end of the sequence
                    tokens[-1] = tokenizer.eos
    return tokens


def build_target_token_channel(
    cut: Cut,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
    pad_id: int = -1,
    loss_mask_forcot: float = 1,
    loss_mask_forres: float = 1,
) -> torch.Tensor:
    diagnostic = f"Extra info: {cut.id=}"
    if getattr(cut, "shard_origin", None) is not None:
        diagnostic = f"{diagnostic} {cut.shard_origin=}"

    total = compute_num_frames(cut.duration, frame_length, cut.sampling_rate)
    tokens = torch.ones(total, dtype=torch.long) * pad_id
    last_start = 0
    last_end = 0
    last_ends = 0
    loss_mask = torch.ones_like(tokens)

    rec_last_user_asr = ''
    
    for supervision in cut.supervisions:
        if supervision.speaker in roles:
            get_full_text = supervision.text  #<cot>xxx</cot><cl_cot>xxx</cl_cot>(response)
            get_cot, get_clcot, get_res_text = parse_three_segments(get_full_text)
            get_res_text = get_res_text.strip().strip('*').strip()
            # print(get_res_text)
            text_ids = torch.as_tensor([tokenizer.bos] + tokenizer.text_to_ids(get_res_text))

            # Determine the frame offset for the start of the supervision to insert the text tokens.
            pos = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
            if pos > len(tokens):
                logging.warning(
                    f"Ill-constructed example: the beginning offset of a supervision {pos} is larger than the example's length {len(tokens)}. {diagnostic}"
                )
                continue

            eospos = compute_num_frames(supervision.end, frame_length, cut.sampling_rate)

            available_frames_for_text = eospos - pos

            if available_frames_for_text > 0 and len(text_ids) > available_frames_for_text:
                # Truncate text_ids to fit before the eos position.
                text_ids = text_ids[:available_frames_for_text]
            elif available_frames_for_text <= 0:
                # If there's no space for text (e.g., start >= end), use an empty sequence.
                text_ids = torch.tensor([], dtype=torch.long)

            endpos = pos + len(text_ids)
            if endpos > len(tokens):
                trunc_len = len(tokens) - pos
                logging.warning(
                    f"Truncating training example's text_ids of length {len(text_ids)} by {trunc_len} because {endpos=} > {len(tokens)=}. {diagnostic}"
                )
                text_ids = text_ids[:trunc_len]
                endpos = pos + len(text_ids)
            try:
                tokens[pos:endpos] = text_ids
                loss_mask[pos:endpos] = loss_mask_forres
            except Exception as e:
                raise RuntimeError(f"{tokens.shape=} {pos=} {endpos=} {text_ids.shape=} {diagnostic}") from e
            # Insert EOS at the end of the supervision segment.
            if eospos < len(tokens):  # skip otherwise - unfinished turn
                tokens[eospos] = tokenizer.eos
                loss_mask[pos:eospos+1] = loss_mask_forres
            else:
                loss_mask[pos:len(tokens)] = loss_mask_forres
                # Interruption case: place EOS at the last valid position
                # This ensures the model learns to stop when interrupted by user
                if endpos < len(tokens):
                    # Case 1: text finished, interrupted during sil/audio generation
                    # Place EOS right after the last text token (or at sequence end if closer)
                    actual_eos_pos = min(endpos, len(tokens) - 1)
                    tokens[actual_eos_pos] = tokenizer.eos
                elif len(tokens) > 0:
                    # Case 2: text truncated due to interruption
                    # Place EOS at the very end of the sequence
                    tokens[-1] = tokenizer.eos


            write_end = pos
            write_start = max(last_start + int(0.64 * 12.5), last_end + 1, last_ends + 1)
            # window = tokens[write_start:write_end]
            if write_end - write_start > 2 and loss_mask_forcot == 0:
                # print(rec_last_user_asr)
                latent_ids = tokenizer.text_to_ids(rec_last_user_asr)

                boc_text_id = tokenizer.tokens_to_ids("<|cot_start|>")
                eoc_text_id = tokenizer.tokens_to_ids("<|cot_end|>")

                # expanded_latent_ids = expand_transcription_to_audio_length(latent_ids, len(window) - 2, pad_id)
                # print([boc_text_id])
                # print(expanded_latent_ids)
                # print([eoc_text_id])
                # window[:] = torch.as_tensor([boc_text_id] + expanded_latent_ids + [eoc_text_id])

                # tokens[write_start:write_end] = window
                tokens[write_start:write_start + 1] = torch.as_tensor([boc_text_id])
                tokens[write_end - 1:write_end] = torch.as_tensor([eoc_text_id])
                loss_mask[write_start:write_end] = loss_mask_forcot
                loss_mask[write_start] = loss_mask_forres
                loss_mask[write_end - 1] = loss_mask_forres
            last_end = endpos
            last_ends = eospos
        else:
            rec_last_user_asr = supervision.text
        last_start = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
    # print(tokenizer.ids_to_text(tokens, remove_special_tokens = False))
    # print(loss_mask)
    # print(tokenizer.ids_to_text(tokens * loss_mask, remove_special_tokens = False))
    return tokens, loss_mask


def remove_brackets_and_content(text):
    """
    移除字符串中的方括号 [] 及其中的所有内容。
    
    参数:
        text (str): 待处理的原始字符串。
        
    返回:
        str: 处理后的新字符串。
    """
    # 正则表达式模式 r'\[.*?\]' 用于匹配方括号及其中的内容
    # \[ 和 \] 表示匹配左右方括号本身
    # .*? 表示非贪婪模式，匹配任意字符（除换行符外）零次或多次
    pattern = r' \[.*?\]'
    # 使用 re.sub 将匹配到的内容替换为空字符串
    result = re.sub(pattern, '', text)
    return result

def remove_curly_braces(content):
    """
    移除字符串中的所有花括号 {} 及其内部的内容。

    参数:
        content (str): 待处理的原始字符串。

    返回:
        str: 处理后的新字符串。
    """
    pattern = r'\{.*?\}'
    return re.sub(pattern, '', content)

def _strip_timestamps(
    text: str, _TIMESTAMP_PATTERN=re.compile(r"<\|\d+\|>"), _SPACE_PATTERN=re.compile(r"\s+")
) -> str:
    """
    Strips timestamp tokens from text, e.g. turns:
      '<|0|> Hey <|3|> <|3|> how <|5|> <|7|> are <|8|> <|8|> <|10|> you? <|12|>'
      into:
      'Hey how are you?'
    """
    # Regexp pattern args are cached compiled patterns (micro-optimization).
    text = _TIMESTAMP_PATTERN.sub("", text)  # strip timestamp tokens if present
    return _SPACE_PATTERN.sub(" ", text).strip()  # strip multi-whitespaces


def expand_transcription_to_audio_length(transcription_ids: list,
                                           target_length: int, 
                                           pad_id) -> list:
        """
        将转录文本embedding按比例扩展到音频长度

        Args:
            transcription_emb: [seq_len, hidden_size] 转录文本embedding
            target_length: 目标音频长度

        Returns:
            扩展后的embedding [target_length, hidden_size]
        """
        seq_len = len(transcription_ids)

        if seq_len == 0:
            # 如果转录为空，返回零向量
            return [pad_id for _ in range(target_length)]
        
        if target_length <= seq_len:
            return transcription_ids[:target_length]

        # 计算每个转录token需要重复的次数
        repeat_counts = []
        base_repeat = target_length // seq_len
        remainder = target_length % seq_len

        for i in range(seq_len):
            repeat_count = base_repeat + (1 if i < remainder else 0)
            # if i == seq_len - 1:
                # repeat_count = repeat_count + remainder
            repeat_counts.append(repeat_count)

        # 按计算的重复次数扩展
        expanded_id = []
        for i, repeat_count in enumerate(repeat_counts):
            token_id = transcription_ids[i]  # [1, hidden_size]
            expanded_id.extend([token_id] * repeat_count)

        return expanded_id