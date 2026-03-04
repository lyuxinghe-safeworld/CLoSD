# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import os
import re

import torch
from closd.env.tasks import closd_task
from isaacgym.torch_utils import *
from closd.utils.closd_util import STATES

class CLoSDT2M(closd_task.CLoSDTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.init_state = STATES.TEXT2MOTION
        self.hml_data_buf_size = max(self.fake_mdm_args.context_len, self.planning_horizon_20fps)
        self.hml_prefix_from_data = torch.zeros([self.num_envs, 263, 1, self.hml_data_buf_size], dtype=torch.float32, device=self.device)
        self.segment_hold_last = bool(self.cfg["env"].get("segment_hold_last", True))
        self.segment_schedule_path = self.cfg["env"].get("segment_schedule_path", "")
        self.prompt_per_horizon = []
        self._last_prompt_log_horizon = None
        self._last_prompt_log_value = None
        self._load_segment_schedule()
        return

    def _load_segment_schedule(self):
        if not self.segment_schedule_path:
            return

        schedule_path = os.path.expanduser(self.segment_schedule_path)
        if not os.path.isabs(schedule_path):
            schedule_path = os.path.abspath(schedule_path)

        with open(schedule_path, "r", encoding="utf-8") as f:
            schedule = json.load(f)

        prompt_per_horizon = []
        for segment in schedule.get("segments", []):
            prompt = self._canonicalize_scheduled_prompt(str(segment.get("prompt", "")).strip())
            if not prompt:
                continue
            try:
                num_horizons = int(segment.get("num_horizons", 1))
            except Exception:
                num_horizons = 1
            num_horizons = max(1, num_horizons)
            prompt_per_horizon.extend([prompt] * num_horizons)

        # Optional direct mapping support in schedule JSON.
        if not prompt_per_horizon and isinstance(schedule.get("prompt_per_horizon"), list):
            prompt_per_horizon = []
            for p in schedule["prompt_per_horizon"]:
                norm_prompt = self._canonicalize_scheduled_prompt(str(p).strip())
                if norm_prompt:
                    prompt_per_horizon.append(norm_prompt)

        if not prompt_per_horizon:
            raise ValueError(f"Schedule file [{schedule_path}] did not produce any valid prompt.")

        self.prompt_per_horizon = prompt_per_horizon
        print(
            f"[CLoSDT2M] Loaded segment schedule: path={schedule_path}, "
            f"total_horizons={len(self.prompt_per_horizon)}, hold_last={self.segment_hold_last}"
        )

    def _canonicalize_scheduled_prompt(self, prompt):
        text = str(prompt).strip()
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[ \t\r\n\.\!\?;:,]+$", "", text)
        lowered = text.lower()
        for prefix in ["a person is ", "person is ", "the person is ", "someone is ", "a human is "]:
            if lowered.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        if not text:
            text = "moving"
        return f"a person is {text}."

    def _get_scheduled_prompt(self, horizon_idx):
        if not self.prompt_per_horizon:
            return None
        if horizon_idx < len(self.prompt_per_horizon):
            return self.prompt_per_horizon[horizon_idx]
        if self.segment_hold_last:
            return self.prompt_per_horizon[-1]
        return None
    
    def update_mdm_conditions(self, env_ids):  
        super().update_mdm_conditions(env_ids)
        custom_prompt = self.cfg['env'].get('custom_prompt', '')
        
        # updates prompts and lengths
        try:
            gt_motion, model_kwargs = next(self.mdm_data_iter)
        except StopIteration:
            del self.mdm_data_iter
            self.mdm_data_iter = iter(self.mdm_data) # re-initialize
            gt_motion, model_kwargs = next(self.mdm_data_iter)
        scheduled_prompt = self._get_scheduled_prompt(0)
        for i in env_ids:
            if scheduled_prompt is not None:
                self.hml_prompts[int(i)] = scheduled_prompt
            elif custom_prompt:
                self.hml_prompts[int(i)] = custom_prompt
            else:
                self.hml_prompts[int(i)] = model_kwargs['y']['text'][int(i)]
            self.hml_lengths[int(i)] = model_kwargs['y']['lengths'][int(i)]  
            self.hml_tokens[int(i)] = model_kwargs['y']['tokens'][int(i)]  
            self.db_keys[int(i)] = model_kwargs['y']['db_key'][int(i)]  
        self.hml_prefix_from_data[env_ids] = gt_motion[..., :self.hml_data_buf_size].to(self.device)[env_ids]  # will be used by the first MDM iteration
        if self.cfg['env']['dip']['debug_hml']:
            print(f'in update_mdm_conditions: 1st 10 env_ids={env_ids[:10].cpu().numpy()}, prompts={self.hml_prompts[:2]}')
        return

    def get_text_prompts(self):
        horizon_idx = self.frame_idx // self.planning_horizon_30fps
        scheduled_prompt = self._get_scheduled_prompt(horizon_idx)
        if scheduled_prompt is not None:
            prompts = [scheduled_prompt] * self.num_envs
            self.hml_prompts = prompts
            if self._last_prompt_log_horizon != horizon_idx or self._last_prompt_log_value != scheduled_prompt:
                switch_tag = "SWITCH" if self._last_prompt_log_value not in [None, scheduled_prompt] else "USE"
                print(
                    f"[CLoSDT2M][{switch_tag}] frame={self.frame_idx} horizon={horizon_idx} "
                    f"prompt={scheduled_prompt}"
                )
                self._last_prompt_log_horizon = horizon_idx
                self._last_prompt_log_value = scheduled_prompt
            return prompts
        if self.prompt_per_horizon and not self.segment_hold_last:
            fallback_prompt = str(self.cfg["env"].get("custom_prompt", "")).strip()
            if not fallback_prompt:
                fallback_prompt = self.prompt_per_horizon[0]
            prompts = [fallback_prompt] * self.num_envs
            self.hml_prompts = prompts
            if self._last_prompt_log_horizon != horizon_idx or self._last_prompt_log_value != fallback_prompt:
                print(
                    f"[CLoSDT2M][EXHAUSTED] frame={self.frame_idx} horizon={horizon_idx} "
                    f"prompt={fallback_prompt}"
                )
                self._last_prompt_log_horizon = horizon_idx
                self._last_prompt_log_value = fallback_prompt
            return prompts
        return super().get_text_prompts()
    
    def get_cur_done(self):
        # Done signal is not in use for this task
        return torch.zeros([self.num_envs], device=self.device, dtype=bool)
    
