import uuid

from django.db import models
from django.contrib.auth.hashers import make_password, check_password
from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    uid = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    GENDER_CHOICES = [
        ('M', '男'),
        ('F', '女'),
        ('O', '保密'),
    ]
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, default='O')
    message = models.TextField(default="", blank=True)

    def __str__(self):
        return self.username

    def set_password(self, raw_password):
        self.password = make_password(raw_password)

    def check_password(self, raw_password):
        return check_password(raw_password, self.password)


class PhysicalMachine(models.Model):
    pid = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    cpu = models.IntegerField()
    memory = models.IntegerField()
    name = models.CharField(max_length=50, default="None")

    @property
    def used_cpu(self):
        return sum(d.vm.cpu for d in self.deployments.all() if d.vm)

    @property
    def used_mem(self):
        return sum(d.vm.memory for d in self.deployments.all() if d.vm)

    def __str__(self):
        return f"PM-{self.pid}-{self.name}"


class VirtualMachine(models.Model):
    vid = models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True)
    cpu = models.IntegerField()
    category = models.IntegerField(default=-1)
    memory = models.IntegerField()
    name = models.CharField(max_length=50, default="None")
    deploy_on = models.ForeignKey(PhysicalMachine, on_delete=models.CASCADE, related_name="vm")
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="vm", null=True, blank=True)

    def __str__(self):
        return f"VM-{self.vid}-{self.name}"


class Deploy(models.Model):
    pm = models.ForeignKey(PhysicalMachine, on_delete=models.CASCADE, related_name="deployments")
    vm = models.ForeignKey(VirtualMachine, on_delete=models.CASCADE, null=True, blank=True, related_name="deployments")
    create_on = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="deploy", null=True, blank=True)
    
    message = models.TextField(default="", blank=True)
    method = models.CharField(max_length=50, default="auto")

    def __str__(self):
        return f"Deploy VM-{self.vm.name} to PM-{self.pm.name if self.pm else 'None'}"


class RLHyperParameter(models.Model):
    lr = models.FloatField(default=0.003, help_text='学习率（DQN 使用），控制权重更新速度')
    actor_lr = models.FloatField(default=0.003, help_text='Actor 网络的学习率')
    critic_lr = models.FloatField(default=0.003, help_text='Critic 网络的学习率')
    gamma = models.FloatField(default=0.98, help_text='折扣因子 γ，衡量未来奖励的重要性')
    lmbda = models.FloatField(default=0.95,
                              help_text='GAE（Generalized Advantage Estimation）中的 λ，用于平衡 bias 和 variance')
    eps = models.FloatField(default=0.2, help_text='PPO 剪切范围 ε，控制策略更新幅度，防止策略崩溃')
    epochs = models.IntegerField(default=10, help_text='每轮更新中优化策略网络的次数')
    hidden_dim = models.IntegerField(default=128, help_text='隐藏层维度，控制网络容量')
    target_update = models.IntegerField(default=10, help_text='目标网络更新频率，每隔 N 步复制一次权重')
    n_step = models.IntegerField(default=3, help_text='累计奖励的步长')

    name = models.CharField(max_length=100, unique=True)
    create_on = models.DateTimeField(auto_now_add=True)
    update_on = models.DateTimeField(auto_now=True)

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="hyperparameter", null=True, blank=True)

    def __str__(self):
        return f"用户{self.user.name}的模型{self.name}"
