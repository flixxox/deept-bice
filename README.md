# DeepT BiCE: Training spiking neural networks

This repository contains our code for the BiCE project which is based on the DeepT framework.

## Requirements

<ul>
    <li> Python 3.11.6
</ul>

## Install

```bash
export DEEPT_VENV_FOLDER=<YOUR_VENV_FOLDER>
export DEEPT_SNN_ROOT=<YOUR_DEEPT_SNN_ROOT>
```
```bash
python3 -m venv $DEEPT_VENV_FOLDER
source $DEEPT_VENV_FOLDER/bin/activate
pip3 install --upgrade pip
cd $DEEPT_SNN_ROOT
poetry install
```