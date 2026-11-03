import os

DEFAULTS = {
    "SN56_RESTORE_SCALING": "1",
    "SN56_DEPTH_FIRST": "1",
    "SN56_A0_SHRINK": "1",
    "SN56_STOP_TO_DECAY": "0",
    "SN56_EE_PRICED": "0",
    "SN56_PACKED_FILL": "1",
    "SN56_PRECISE_WRITEBACK": "carry",
    "SN56_CARRY_LR_CAL": "1",
    "SN56_LR_COLDEDGE_FUND": "1",
    "SN56_FP32_MASTER": "1",
    "SN56_DEALCAL": "1",
    "SN56_EVAL_PHASE_DEALIAS": "1",
    "SN56_LOGIT_CALIB": "0",
    "SN56_COVERAGE_CLOCK": "1",
    "SN56_TOKEN_BUDGET": "1",
    "SN56_LR_ALLDIVERGE_FIX": "1",
    "SN56_FP32_OOM_FIX": "1",
    "SN56_SB_REJECT_RESCUE": "1",
    "SN56_A0_COSINE": "0",
    "SN56_PHI_RTN_COSINE": "1",
    "SN56_LR_MUP_RMS": "1",
}


def raw(name: str) -> str:
    return os.environ.get(name, DEFAULTS[name])
