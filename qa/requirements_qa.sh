#!/usr/bin/env bash
git clone https://github.com/huggingface/transformers
cd transformers
git checkout cefd51c50cc08be8146c1151544495968ce8f2ad
pip install .
pip install learn2learn==0.1.3
pip install seqeval
pip install tensorboardx