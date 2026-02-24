# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Transformer-Walk-Direct-v0",  # ✅ FIXED: Changed name
    entry_point=f"{__name__}.transformer_nam_env:TransformerWalkEnv",  # ✅ FIXED: Correct class
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.transformer_nam_env:TransformerWalkEnvCfg",  # ✅ FIXED: Same file
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TransformerWalkPPORunnerCfg",  # ✅ FIXED: Correct config
    },
)