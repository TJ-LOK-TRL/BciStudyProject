#!/bin/bash
# 1. FORCE DEACTIVATE: Close any active env (repeated to ensure clean state)
# "2>/dev/null" suppresses errors if there is nothing to deactivate
conda deactivate 2>/dev/null
conda deactivate 2>/dev/null

# 2. HARD CLEAN: Remove environment variables that cause conflicts
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset PYTHONHOME
unset PYTHONPATH
unset CONDA_SHLVL

# 3. INITIALIZE: Wake up the Server Conda
# Using an 'if' check to be safe
CONDA_SCRIPT="/opt/bci_eeg_conda/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SCRIPT" ]; then
    source "$CONDA_SCRIPT"
else
    echo "ERROR: Conda not found at /opt!"
    return 1 2>/dev/null || exit 1
fi

# 4. ACTIVATE: Enter the team environment
conda activate /opt/bci_eeg_conda/envs/eeg_team

# 5. CONFIG: Set Alex's disk for heavy data
export MNE_DATA=/mnt/alexdisk/mne_data

echo "-------------------------------------------------------"
echo "✅ EEG-AI Environment Ready!"
echo "Python: $(which python)"
echo "Active Env: $CONDA_DEFAULT_ENV"
echo "Data Path: $MNE_DATA"
echo "-------------------------------------------------------"