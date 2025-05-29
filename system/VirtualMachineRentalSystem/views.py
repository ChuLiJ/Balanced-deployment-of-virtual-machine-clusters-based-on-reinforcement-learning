import json
import datetime

from django.contrib import messages
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, update_session_auth_hash, login as auth_login
from django.contrib.auth.decorators import login_required
from django.urls import reverse

from .forms import RLHyperParameterForm
from .models import VirtualMachine, PhysicalMachine, Deploy, User
from .rl_core.train import train_model
from .rl_core.env import generate_pms, generate_vms, classify_new_vm


def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)
        if user is not None:
            auth_login(request, user)
            return redirect('dashboard')
        else:
            error = "用户名或密码错误"
            return render(request, 'login.html', {'error': error})
    else:
        return render(request, 'login.html')


def register(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        gender = request.POST.get('gender', 'O')
        message = request.POST.get('message', "")
        confirm_password = request.POST.get('confirm_password')

        if User.objects.filter(username=username).exists():
            error = "用户名已存在"
            return render(request, 'register.html', {'error': error})

        if password != confirm_password:
            return render(request, 'register.html', {'error': '两次密码输入不一致！'})

        user = User(username=username, gender=gender, message=message)
        user.set_password(password)
        user.save()
        return redirect('login')

    return render(request, 'register.html')


@login_required(login_url='login')
def dashboard(request):
    user = request.user

    pm_count = PhysicalMachine.objects.count()
    vm_count = VirtualMachine.objects.count()

    pms = PhysicalMachine.objects.all()
    vms = VirtualMachine.objects.all()

    total_cpu = 0
    total_mem = 0
    used_cpu = 0
    used_mem = 0

    for pm in pms:
        total_cpu += pm.cpu
        total_mem += pm.memory
    for vm in vms:
        used_cpu += vm.cpu
        used_mem += vm.memory

    if total_cpu and total_mem:
        cpu_load = int(used_cpu / total_cpu * 100)
        mem_load = int(used_mem / total_mem * 100)
        avg_load = int((cpu_load + mem_load) / 2)
    else:
        cpu_load = 0
        mem_load = 0
        avg_load = 0

    context = {
        'name': user.username,
        'physical_machine_count': pm_count,
        'virtual_machine_count': vm_count,
        'avg_load': avg_load,
        'cpu_usage': cpu_load,
        'memory_usage': mem_load,
        'disk_usage': 0,
    }
    return render(request, 'dashboard.html', context)


@login_required(login_url='login')
def main_margin(request):
    user = request.user
    deploy_list = Deploy.objects.filter(user=user)
    return render(request, 'main_margin.html', {'deployments': deploy_list})


@login_required(login_url='login')
def pm_display(request):
    user = request.user
    pm_list = PhysicalMachine.objects.all()
    pms = []
    for pm in pm_list:
        remain_cpu = pm.cpu - pm.used_cpu
        remain_mem = pm.memory - pm.used_mem
        over_cpu = int(pm.used_cpu / pm.cpu * 100)
        over_mem = int(pm.used_mem / pm.memory * 100)
        pms.append({
            'pid': str(pm.pid),
            'name': pm.name,
            'remain_cpu': remain_cpu,
            'over_cpu': over_cpu,
            'remain_mem': remain_mem,
            'over_mem': over_mem,
        })
    return render(request, "pm_display.html", {'pm': pms})


@login_required(login_url='login')
def user_profile(request):
    user = request.user
    return render(request, 'user.html', {'user': user})


@login_required(login_url='login')
def edit_profile(request):
    if request.method == 'POST':
        user = request.user
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')

        user.username = username
        user.email = email

        if password:
            user.set_password(password)
            update_session_auth_hash(request, user)

        user.save()
        messages.success(request, "个人信息已更新！")
        return redirect('dashboard')
    else:
        return render(request, 'user.html')


def test(request):
    return render(request, 'test.html')


@login_required(login_url='login')
def pm_detail(request, pm_pid):
    pm_obj = get_object_or_404(PhysicalMachine, pid=pm_pid)
    return render(request, "pm_detail.html", {'pm': pm_obj})


@login_required(login_url='login')
def clear(request):
    user = request.user
    if request.method == 'POST':
        Deploy.objects.all().delete()
        VirtualMachine.objects.all().delete()
        PhysicalMachine.objects.all().delete()
        return HttpResponseRedirect(reverse('vms'))
    else:
        return HttpResponse("Invalid request method.", status=400)


@login_required(login_url='login')
def deploy(request):
    user = request.user
    if request.method == 'POST':
        pm_objs = generate_pms(50, (32, 128), (128, 512))
        generate_vms(200, pm_objs, user=user)
        return HttpResponseRedirect(reverse('vms'))
    else:
        return HttpResponse("Invalid request method.", status=405)


@login_required(login_url='login')
def deploy_chart(request):
    pm_objs = PhysicalMachine.objects.all()
    data = []
    for pm in pm_objs:
        used_cpu = sum(d.vm.cpu for d in pm.deployments.all() if d.vm)
        used_mem = sum(d.vm.memory for d in pm.deployments.all() if d.vm)
        data.append({
            'name': pm.name,
            'total_cpu': pm.cpu,
            'used_cpu': used_cpu,
            'total_mem': pm.memory,
            'used_mem': used_mem,
            'pid': str(pm.pid),
        })
    return render(request, 'deployment_chart.html', {'data': json.dumps(data)})


@login_required(login_url='login')
def add(request, pm_pid):
    pm = get_object_or_404(PhysicalMachine, pid=pm_pid)
    context = {
        'pid': str(pm.pid),
        'name': pm.name,
        'available_cpu': pm.cpu - pm.used_cpu,
        'available_memory': pm.memory - pm.used_mem,
    }
    return render(request, 'add.html', {'pm': context})


@login_required(login_url='login')
def add_vm(request, pm_pid):
    user = request.user
    pm_obj = get_object_or_404(PhysicalMachine, pid=pm_pid)
    if request.method == 'POST':
        cpu = request.POST.get('cpu')
        memory = request.POST.get('memory')
        vm_name = request.POST.get('vm_name')
        remark = request.POST.get('remarks')
        category = classify_new_vm(cpu, memory)

        vm_obj = VirtualMachine.objects.create(
            cpu=cpu,
            memory=memory,
            name=vm_name,
            category=category,
            deploy_on=pm_obj,
            user=user,
        )
        Deploy.objects.create(
            pm=pm_obj,
            vm=vm_obj,
            create_on=datetime.datetime.now(),
            user=user,
            message=remark,
            method="自建",
        )

        return redirect('dashboard')
    else:
        return render(request, 'add.html')


def train_button(request):
    return render(request, 'train_button.html')


def train(request):
    if request.method == 'POST':
        train_model()
        return HttpResponse("Training started successfully!")
    else:
        return HttpResponse("Invalid request method.", status=400)


@login_required(login_url='login')
def set_hyperparams(request):
    if request.method == 'POST':
        form = RLHyperParameterForm(request.POST)
        if form.is_valid():
            hyperparams = form.save(commit=False)
            hyperparams.user = request.user
            hyperparams.save()
            return redirect('dashboard')
    else:
        form = RLHyperParameterForm()

    return render(request, 'setting.html', {'form': form})


@login_required(login_url='login')
def rl_migration(request):
    user = request.user
    if request.method == 'POST':
        train_model(user)
        messages.success(request, "迁移完成！")

        return redirect('dashboard')

    return render(request, "rl_migration.html")
