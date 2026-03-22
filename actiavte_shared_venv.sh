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

# 5. CONFIG: Set the variables of the datasets
export MNE_DATA=/data/bci_eeg_data/mne_data
export BCI_DATASETS=/data/bci_eeg_data/datasets

# 6. Sync MNE/MOABB config if outdated
python - <<EOF
import os
import mne
from moabb.utils import set_config, get_config

target = os.environ.get("MNE_DATA")
cfg = mne.get_config()
outdated = [k for k, v in cfg.items() if 'mne_data' in str(v).lower() and v != target]

if outdated or get_config('MNE_DATA') != target:
    set_config('MNE_DATA', target)
    for k in outdated:
        set_config(k, target)
    print(f"🔄 MNE/MOABB config actualizado ({len(outdated)} keys)")
EOF

# 7. LOCAL PACKAGE (Dynamic): Resolve project root from script location
# Works because this script lives alongside pyproject.toml and src/
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
if [ -d "$PROJECT_ROOT/src" ]; then
    export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
    echo "📦 Local package: $PROJECT_ROOT/src"
else
    echo "⚠️  Could not find src/ next to this script at: $PROJECT_ROOT"
fi

echo "-------------------------------------------------------"
echo "✅ EEG-AI Environment Ready!"
echo "Python: $(which python)"
echo "Active Env: $CONDA_DEFAULT_ENV"
echo "MNE Data Path: $MNE_DATA"
echo "Manual Data Path: $BCI_DATASETS "
echo "-------------------------------------------------------"