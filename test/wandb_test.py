import wandb

# 初始化一个实验
wandb.init(project="test-project")

# 记录一些数据
wandb.log({"accuracy": 0.9, "loss": 0.1})

# 结束实验
wandb.finish()