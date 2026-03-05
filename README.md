# Transform_bipedal
URDF folder: Transfomer_nam, SimpleTrans, Namtransformer, Bullettransformer, 3DOFTrans, Transformer_IsaacLab
IsaacSim training: transformer_nam

Walking gait demo: ![Demo Robot](Walking-demo.gif)

# Create project

Run through ~/IsaacLab python.sh not python only 

Build project:
Create project (IsaacLab)
```
 ./isaaclab.sh --new
```
Build:
```
~/IsaacLab/isaac-sim/python.sh -m pip install -e source/Transform_process
```
Check:
```
~/IsaacLab/isaac-sim/python.sh scripts/list_envs.py
```

# Training

Install IsaacSim & IssacLab 
conda env: env_isaaclab 
Create & Subscribe task: Transformer-Walk-Direct-v0

Train policy:
```
~/IsaacLab/isaac-sim/python.sh scripts/rsl_rl/train.py     --task Transformer-Walk-Direct-v0     --num_envs xxx     --headless     --max_iterations xxx
```
Replay environment:
```
~/IsaacLab/isaac-sim/python.sh scripts/rsl_rl/play.py     --task Transformer-Walk-Direct-v0     --num_envs 1     --load_run xxx 
```
Visualize chart:
```
tensorboard --logdir=logs/rsl_rl/transformer_walk --port=6006
```
