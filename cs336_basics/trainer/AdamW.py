import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 获取当前梯度
                grad = p.grad.data
                
                # 获取状态信息
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']
                state['step'] += 1
                t = state['step']

                # 1. 更新权重衰减 (L2 惩罚的解耦版本)
                # theta = theta - alpha * lambda * theta
                p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

                # 2. 更新动量
                # m = beta1 * m + (1 - beta1) * g
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v = beta2 * v + (1 - beta2) * g^2
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 3. 计算偏差修正后的学习率
                # alpha_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = group['lr'] * (math.sqrt(bias_correction2) / bias_correction1)

                # 4. 更新参数
                # theta = theta - step_size * m / (sqrt(v) + eps)
                denom = v.sqrt().add_(group['eps'])
                p.data.addcdiv_(m, denom, value=-step_size)

        return loss