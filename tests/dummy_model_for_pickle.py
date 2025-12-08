'''
Classe DummyModel de nível de módulo usada para criar um arquivo .pkl picklable durante os 
testes (evita problemas de pickle com classes locais).
'''

class DummyModel:
    def eval(self):
        pass

    def __call__(self, x):
        import torch
        return torch.zeros((x.size(0), 1))
