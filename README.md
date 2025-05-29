# Virtual Machine Migration Reinforcement Learning

## 项目简介

这是一个我的毕设项目，该项目提供了一个虚拟机迁移的强化学习环境。最浅层的文件（`config.py`、`contrast.py`、`DQN.py`、`env.py`、`main.py`、`PPO.py`、`reward.py`、`utils.py`）可以通过main.py独立运行，适用于进行算法的直接实验和调试。

项目使用强化学习算法来解决虚拟机迁移问题，您可以使用这些模块进行实验、调试和优化模型。

## 项目结构
Graduation design
├── system
│ ├── polls
│ └── system
│ ├── init.py
│ ├── asgi.py
│ ├── settings.py
│ ├── urls.py
│ ├── wsgi.py
├── VirtualMachineRentalSystem
│ ├── migrations
│ ├── rl_core
│ ├── static
│ ├── templates
│ ├── init.py
│ ├── admin.py
│ ├── apps.py
│ ├── forms.py
│ ├── models.py
│ ├── serializers.py
│ ├── tests.py
│ ├── urls.py
│ └── views.py
├── db.sqlite3
└── manage.py
config.py
contrast.py
DQN.py
env.py
main.py
PPO.py
reward.py
utils.py

## 使用说明

### 独立运行虚拟机迁移强化学习环境

1. 直接运行 `config.py`、`contrast.py`、`DQN.py`、`env.py`、`main.py`、`PPO.py`、`reward.py`、`utils.py` 这些文件可以启动虚拟机迁移的强化学习环境，适合用来进行算法实验和调试。

2. 请根据需求修改配置文件（`config.py`）来调整环境设置。

### 系统开发

1. 系统部分的开发请转到 `manage.py` 同级目录。
2. 运行以下命令启动开发服务器：
   ```bash
   py manage.py runserver
3. 访问 http://127.0.0.1:8000 即可访问系统界面。

安装依赖
确保您的环境中安装了以下依赖：
pip install -r requirements.txt

许可证
MIT License

您可以根据需要对这个 README 进行修改和扩展，加入更多的细节或使用示例。
