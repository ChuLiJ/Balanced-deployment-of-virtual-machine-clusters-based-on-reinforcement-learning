# forms.py
from django import forms
from .models import RLHyperParameter


class RLHyperParameterForm(forms.ModelForm):
    class Meta:
        model = RLHyperParameter
        fields = [
            'name', 'lr', 'actor_lr', 'critic_lr', 'gamma', 'lmbda',
            'eps', 'epochs', 'hidden_dim', 'target_update', 'n_step'
        ]
        widgets = {
            field: forms.NumberInput(attrs={'class': 'form-control'}) for field in fields if field != 'name'
        }
        widgets['name'] = forms.TextInput(attrs={'class': 'form-control'})
