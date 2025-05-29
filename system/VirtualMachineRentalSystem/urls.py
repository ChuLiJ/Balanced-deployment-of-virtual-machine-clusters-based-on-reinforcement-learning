from django.urls import path
from . import views

urlpatterns = [
    path('', views.login, name='login'),
    path('register/', views.register, name='register'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('deploying/', views.deploy, name='deploying'),
    path('vms/', views.main_margin, name="vms"),
    path('pms/', views.pm_display, name="pms"),
    path('pms/add/<str:pm_pid>/', views.add, name='add'),
    path('pms/adding/<str:pm_pid>/', views.add_vm, name='adding'),
    path('test/', views.test, name='test'),
    path('migration/', views.train_button, name="train_button"),
    path('train/', views.train, name="train"),
    path('vms/clear/', views.clear, name="clear"),
    path('vms/chart/', views.deploy_chart, name="deploy_chart"),
    path('vms/<str:pm_pid>/', views.pm_detail, name="pm_detail"),
    path('profile/', views.user_profile, name="profile"),
    path('profile/edit/', views.edit_profile, name="edit_profile"),
    path('setting/', views.set_hyperparams, name='setting'),
]
