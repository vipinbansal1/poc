.ipynb file used for googlecolab
.py just for reference.

In order to launch tensorboard on googlecolab, follow two steps:
1. Register extension, "%load_ext tensorboard"
2. Launch tensorboard, "%tensorboard --logdir "out" --port 4321 --debugger_port 7003"

"out", directory having event file