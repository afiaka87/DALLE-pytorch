#!/bin/bash
#! taming_transformers.sh - Downloads new vqgan, installs repo.

git clone https://github.com/CompVis/taming-transformers
cd taming-transformers
mkdir -p logs/2020-11-09T13-31-51_sflckr/checkpoints
wget 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1' -O 'logs/2020-11-09T13-31-51_sflckr/checkpoints/last.ckpt'
mkdir logs/2020-11-09T13-31-51_sflckr/configs
wget 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1' -O 'logs/2020-11-09T13-31-51_sflckr/configs/2020-11-09T13-31-51-project.yaml'
