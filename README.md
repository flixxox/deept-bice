# DeepT BiCE: Training spiking neural networks

This repository contains our code for the BiCE project which is based on the DeepT framework.

## Requirements

<ul>
    <li> Python 3.11.6
</ul>

## Install

```bash
export DEEPT_VENV_FOLDER=<YOUR_VENV_FOLDER>
export DEEPT_ROOT=<YOUR_DEEPT_SNN_ROOT>
export DEEPT_WORKFLOW_ROOT=<YOUR_DEEPT_WORKFLOW_ROOT>
export DEEPT_BICE_ROOT=<YOUR_DEEPT_BICE_ROOT>
```
```bash
python3 -m venv $DEEPT_VENV_FOLDER
source $DEEPT_VENV_FOLDER/bin/activate
pip3 install --upgrade pip

cd $DEEPT_BICE_ROOT
poetry install
cd $DEEPT_ROOT
poetry install
cd $DEEPT_WORKFLOW_ROOT
poetry install

```