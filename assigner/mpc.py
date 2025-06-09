import numpy as np
from sb3_contrib import MaskablePPO

from assigner.base import BaseAssigner


class RLAssigner(BaseAssigner):
    """docstring for AlphaAssigner"""

    def __init__(self, lookup_dict, info, model):
        super().__init__(lookup_dict)
        pretrained = {
            0: "pretrained/ta/MaskablePPO_cfg0_13e4",
            1: "pretrained/ta/MaskablePPO_cfg1_3e5",
            2: "pretrained/ta/MaskablePPO_cfg2_3e5",
            3: "pretrained/ta/MaskablePPO_cfg2_3e5_360",
            4: "pretrained/ta/MaskablePPO_cfg3_9e5_360",
            5: "best_models/ta/best_model_cfg3_9e5_360",
            6: "pretrained/ta/MaskablePPO_cfg3_9e5_360_t0r0",
            7: "best_models/ta/best_model_cfg3_9e5_360_t0r0",
            8: "pretrained/ta/MaskablePPO_cfg0_9e5_360_t0r0",
            9: "best_models/ta/best_model_cfg0_9e5_360_t0r0",
            10: "pretrained/ta/MaskablePPO_cfg1_18e5_360",
            11: "best_models/ta/best_model_cfg1_18e5_360",
            12: "pretrained/ta/MaskablePPO_cfg0_18e5_360_t0r0",
            13: "best_models/ta/best_model_cfg0_18e5_360_t0r0",
            # {num_of_agents}{cfg}{train_rand}{final:1, best:0}
            # 50a
            50011: "pretrained/ta/50a/MaskablePPO_cfg0_iter10e5_t0r0_train_rand_eval_rand",
            50010: "best_models/ta/50a/cfg0_t0r0_train_rand_eval_rand/2025-01-09-204217/best_model",
            50111: "pretrained/ta/50a/MaskablePPO_cfg1_iter10e5_t0r0_train_rand_eval_rand",
            50110: "best_models/ta/50a/cfg1_t0r0_train_rand_eval_rand/2025-01-09-204444/best_model",
            50311: "pretrained/ta/50a/MaskablePPO_cfg3_iter10e5_t0r0_train_rand_eval_rand",
            50310: "best_models/ta/50a/cfg3_t0r0_train_rand_eval_rand/2025-01-09-204820/best_model",
            50001: "pretrained/ta/50a/MaskablePPO_cfg0_iter10e5_t0r0_train_fix_eval_fix",
            50000: "best_models/ta/50a/cfg0_t0r0_train_fix_eval_fix/2025-01-10-153002/best_model",
            50101: "pretrained/ta/50a/MaskablePPO_cfg1_iter10e5_t0r0_train_fix_eval_fix",
            50100: "best_models/ta/50a/cfg1_t0r0_train_fix_eval_fix/2025-01-10-153200/best_model",
            50301: "pretrained/ta/50a/MaskablePPO_cfg3_iter10e5_t0r0_train_fix_eval_fix",
            50300: "best_models/ta/50a/cfg3_t0r0_train_fix_eval_fix/2025-01-10-153240/best_model",

            # 30a
            30011: "pretrained/ta/30a/MaskablePPO_cfg0_iter60e4_t0r0_train_rand_eval_rand",
            30010: "best_models/ta/30a/cfg0_t0r0_train_rand_eval_rand/2025-01-17-090151/best_model",
            30111: "pretrained/ta/30a/MaskablePPO_cfg1_iter60e4_t0r0_train_rand_eval_rand",
            30110: "best_models/ta/30a/cfg1_t0r0_train_rand_eval_rand/2025-01-17-090152/best_model",
            30311: "pretrained/ta/30a/MaskablePPO_cfg3_iter60e4_t0r0_train_rand_eval_rand",
            30310: "best_models/ta/30a/cfg3_t0r0_train_rand_eval_rand/2025-01-17-090208/best_model",
            30001: "pretrained/ta/30a/MaskablePPO_cfg0_iter60e4_t0r0_train_fix_eval_fix",
            30000: "best_models/ta/30a/cfg0_t0r0_train_fix_eval_fix/2025-01-15-225731/best_model",
            30101: "pretrained/ta/30a/MaskablePPO_cfg1_iter60e4_t0r0_train_fix_eval_fix",
            30100: "best_models/ta/30a/cfg1_t0r0_train_fix_eval_fix/2025-01-15-225731/best_model",
            30301: "pretrained/ta/30a/MaskablePPO_cfg3_iter60e4_t0r0_train_fix_eval_fix",
            30300: "best_models/ta/30a/cfg3_t0r0_train_fix_eval_fix/2025-01-15-225757/best_model",

            # 40a
            40011: "pretrained/ta/40a/MaskablePPO_cfg0_iter80e4_t0r0_train_rand_eval_rand",
            40010: "best_models/ta/40a/cfg0_t0r0_train_rand_eval_rand/2025-01-17-090159/best_model",
            40111: "pretrained/ta/40a/MaskablePPO_cfg1_iter80e4_t0r0_train_rand_eval_rand",
            40110: "best_models/ta/40a/cfg1_t0r0_train_rand_eval_rand/2025-01-17-090205/best_model",
            40311: "pretrained/ta/40a/MaskablePPO_cfg3_iter80e4_t0r0_train_rand_eval_rand",
            40310: "best_models/ta/40a/cfg3_t0r0_train_rand_eval_rand/2025-01-17-090213/best_model",
            40001: "pretrained/ta/40a/MaskablePPO_cfg0_iter80e4_t0r0_train_fix_eval_fix",
            40000: "best_models/ta/40a/cfg0_t0r0_train_fix_eval_fix/2025-01-15-225731/best_model",
            40101: "pretrained/ta/40a/MaskablePPO_cfg1_iter80e4_t0r0_train_fix_eval_fix",
            40100: "best_models/ta/40a/cfg1_t0r0_train_fix_eval_fix/2025-01-15-225737/best_model",
            40301: "pretrained/ta/40a/MaskablePPO_cfg3_iter80e4_t0r0_train_fix_eval_fix",
            40300: "best_models/ta/40a/cfg3_t0r0_train_fix_eval_fix/2025-01-15-225757/best_model",

            # 60a
            60011: "pretrained/ta/60a/MaskablePPO_cfg0_iter120e4_t0r0_train_rand_eval_rand",
            60010: "best_models/ta/60a/cfg0_t0r0_train_rand_eval_rand/2025-01-17-090158/best_model",
            60111: "pretrained/ta/60a/MaskablePPO_cfg1_iter120e4_t0r0_train_rand_eval_rand",
            60110: "best_models/ta/60a/cfg1_t0r0_train_rand_eval_rand/2025-01-17-090202/best_model",
            60311: "pretrained/ta/60a/MaskablePPO_cfg3_iter120e4_t0r0_train_rand_eval_rand",
            60310: "best_models/ta/60a/cfg3_t0r0_train_rand_eval_rand/2025-01-17-090213/best_model",
            60001: "pretrained/ta/60a/MaskablePPO_cfg0_iter120e4_t0r0_train_fix_eval_fix",
            60000: "best_models/ta/60a/cfg0_t0r0_train_fix_eval_fix/2025-01-15-225736/best_model",
            60101: "pretrained/ta/60a/MaskablePPO_cfg1_iter120e4_t0r0_train_fix_eval_fix",
            60100: "best_models/ta/60a/cfg1_t0r0_train_fix_eval_fix/2025-01-15-225743/best_model",
            60301: "pretrained/ta/60a/MaskablePPO_cfg3_iter120e4_t0r0_train_fix_eval_fix",
            60300: "best_models/ta/60a/cfg3_t0r0_train_fix_eval_fix/2025-01-15-225757/best_model",

            # 70a
            70011: "pretrained/ta/70a/MaskablePPO_cfg0_iter140e4_t0r0_train_rand_eval_rand",
            70010: "best_models/ta/70a/cfg0_t0r0_train_rand_eval_rand/2025-01-17-090201/best_model",
            70111: "pretrained/ta/70a/MaskablePPO_cfg1_iter140e4_t0r0_train_rand_eval_rand",
            70110: "best_models/ta/70a/cfg1_t0r0_train_rand_eval_rand/2025-01-17-090205/best_model",
            70311: "pretrained/ta/70a/MaskablePPO_cfg3_iter140e4_t0r0_train_rand_eval_rand",
            70310: "best_models/ta/70a/cfg3_t0r0_train_rand_eval_rand/2025-01-17-090213/best_model",
            70001: "pretrained/ta/70a/MaskablePPO_cfg0_iter140e4_t0r0_train_fix_eval_fix",
            70000: "best_models/ta/70a/cfg0_t0r0_train_fix_eval_fix/2025-01-15-225736/best_model",
            70101: "pretrained/ta/70a/MaskablePPO_cfg1_iter140e4_t0r0_train_fix_eval_fix",
            70100: "best_models/ta/70a/cfg1_t0r0_train_fix_eval_fix/2025-01-15-225742/best_model",
            70301: "pretrained/ta/70a/MaskablePPO_cfg3_iter140e4_t0r0_train_fix_eval_fix",
            70300: "best_models/ta/70a/cfg3_t0r0_train_fix_eval_fix/2025-01-15-225757/best_model",

        }
        print(pretrained[model])
        self.policy = MaskablePPO.load(pretrained[model])
        self.layout = info['layout']

    def assign(self, task, state):
        locations = state['locations']
        directions = state['directions']
        obs = np.concatenate(
            [locations / np.array(self.layout.shape),
             (np.array(directions) / 360).reshape(-1, 1)],
            axis=1
        ).reshape(-1)

        candidates = self.lookup_dict[task]
        # print(candidates)
        action_mask = np.zeros(151, dtype=np.int8)
        action_mask[candidates] = 1

        chosen = self.policy.predict(obs, action_masks=action_mask, deterministic=True)[0]
        # print(chosen)
        i = candidates.index(chosen)
        del self.lookup_dict[task][i]

        return chosen
