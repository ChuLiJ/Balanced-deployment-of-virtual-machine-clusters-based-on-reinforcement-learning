from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import PhysicalMachine, VirtualMachine, Deploy, User, RLHyperParameter


@admin.register(PhysicalMachine)
class PhysicalMachineAdmin(admin.ModelAdmin):
    list_display = ('name', 'pid', 'cpu', 'memory')


@admin.register(VirtualMachine)
class VirtualMachineAdmin(admin.ModelAdmin):
    list_display = ('name', 'vid', 'cpu', 'memory', 'category', 'deploy_on', 'user')


@admin.register(Deploy)
class DeployAdmin(admin.ModelAdmin):
    list_display = ('vm', 'pm', 'method', 'create_on', 'message', 'user')
    list_filter = ('method', 'create_on')
    search_fields = ('vm__name', 'pm__name', 'message')


@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display = ['username', 'email', 'gender', 'is_staff']
    list_filter = ['gender']
    search_fields = ['username']


@admin.register(RLHyperParameter)
class RLHyperParameter(admin.ModelAdmin):
    list_display = ('user', 'lr', 'actor_lr', 'critic_lr', 'gamma', 'lmbda', 'eps', 'epochs', 'hidden_dim',
                    'target_update', 'n_step', 'name', 'create_on', 'update_on')
    list_filter = ('create_on', 'update_on')
    search_fields = ('user', 'name')

