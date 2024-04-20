# Get the absolute path of the script directory automatically
# If it does not work, specify ProjDir manually
ProjDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "Project directory: $ProjDir"

accelerate launch --config_file $ProjDir/stdiff/configs/accelerate_config.yaml $ProjDir/stdiff/train_stdiff.py --train_config ./configs/train_config.yaml