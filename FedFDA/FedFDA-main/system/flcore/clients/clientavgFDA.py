
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from torch.autograd import Variable
import torch.nn.functional as F


class DualPathAttention(nn.Module):
    """双路注意力融合模块"""

    def __init__(self, feature_dim=512, num_heads=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        # 第一路：当前特征→历史特征
        self.curr_to_hist = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 第二路：历史特征→当前特征
        self.hist_to_curr = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 门控权重生成器
        self.gate = nn.Sequential(
            nn.Linear(2 * feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, curr, hist):
        """
        Args:
            curr: 当前特征 [B, D]
            hist: 历史特征 [B, D]
        Returns:
            fused: 融合特征 [B, D]
        """
        # 路径1：当前特征关注历史上下文
        attn1, _ = self.curr_to_hist(
            query=curr.unsqueeze(1),  # [B, 1, D]
            key=hist.unsqueeze(1),  # [B, 1, D]
            value=hist.unsqueeze(1)
        )
        attn1 = attn1.squeeze(1)  # [B, D]

        # 路径2：历史特征关注当前上下文
        attn2, _ = self.hist_to_curr(
            query=hist.unsqueeze(1),  # [B, 1, D]
            key=curr.unsqueeze(1),  # [B, 1, D]
            value=curr.unsqueeze(1)
        )
        attn2 = attn2.squeeze(1)  # [B, D]

        # 动态门控融合
        gate_input = torch.cat([curr, hist], dim=1)  # [B, 2D]
        gate_weight = self.gate(gate_input)  # [B, 1]

        # 加权融合
        fused = gate_weight * attn1 + (1 - gate_weight) * attn2 + curr
        # # 让 curr_to_hist 的权重大于 hist_to_curr
        # dominant_gate_weight = gate_weight * 0.75 + 0.15  # 将 curr_to_hist 的权重增大
        # fused = dominant_gate_weight * attn1 + (1 - dominant_gate_weight) * attn2
        return fused

# ====== 梯度反转层（GRL） ======
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)


# ====== 领域分类器 ======
class DomainClassifier(nn.Module):
    """领域分类器：输入特征维度，输出领域标签（0=本地，1=全局）"""

    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出领域概率
        )

    def forward(self, x):
        return self.net(x)

class MINE(nn.Module):
    """
    Mutual Information Neural Estimation (MINE) model to estimate the mutual information
    between two random variables X and Y.
    """

    def __init__(self, input_dim, hidden_dim=128):
        super(MINE, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # Concatenate X and Y
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)

        )

    # def forward(self, x, y):
    #     # Concatenate x and y to form the input to the network
    #     inputs = torch.cat((x, y), dim=-1)
    #     return self.network(inputs)
    #
    def forward(self, x, y):
        batch_size = x.size(0)

        # Shuffle y to create the negative sample
        idx = torch.randperm(batch_size)
        shuffled_y = y[idx]

        # Only concatenate x with original y and shuffled y, without doubling batch size
        logits_1 = self.network(torch.cat([x, y], dim=-1))  # Original pair (x, y)
        logits_2 = self.network(torch.cat([x, shuffled_y], dim=-1))  # Negative pair (x, shuffled_y)

        # Compute the loss directly using the logits from both real and shuffled pairs
        pred_xy = logits_1
        pred_x_y = logits_2

        # Use logsumexp instead of log(mean(exp(...)))
        log_mean_exp_pred_x_y = torch.logsumexp(pred_x_y, dim=0) - torch.log(
            torch.tensor(batch_size, dtype=torch.float32))

        #torch.log(torch.mean(torch.exp(pred_x_y)))
        # Compute the loss
        loss = - torch.mean(pred_xy) + log_mean_exp_pred_x_y

        return loss


class clientAvgFDA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.klw = 100
        self.momentum = args.momentum
        self.global_mean = None

        trainloader = self.load_train_data()
        for x, y in trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                rep = self.model.base(x).detach()
            break
        self.running_mean = torch.zeros_like(rep[0])
        self.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=self.device)

        self.client_mean = nn.Parameter(Variable(torch.zeros_like(rep[0])))
        self.opt_client_mean = torch.optim.SGD([self.client_mean], lr=self.learning_rate)

        # 初始化领域对抗训练参数
        self.domain_classifier = DomainClassifier(input_dim=512).to(self.device)
        self.opt_domain = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=0.001,  # 领域分类器学习率
            betas=(0.5, 0.999)
        )
        self.adv_weight = 0.3  # 领域对抗损失权重
        self.ewc_lambda = 1

        # 替换为双路注意力模块
        self.dual_attention = DualPathAttention(feature_dim=512).to(self.device)

        # 更新优化器参数组
        self.optimizer.add_param_group({'params': self.dual_attention.parameters()})

        self.feature_means = []  # 新增：存储各epoch的特征均值

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        mine_model = MINE(input_dim=512).to(self.device)
        self.domain_classifier.train()  # 确保领域分类器处于训练模式

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        self.reset_running_stats()

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # ====== begin
                rep = self.model.base(x)
                running_mean = torch.mean(rep, dim=0)

                if self.num_batches_tracked is not None:
                    self.num_batches_tracked.add_(1)

                self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * running_mean

                if self.global_mean is not None :

                    # 生成全局特征样本（例如从高斯分布采样）
                    global_features = torch.randn_like(rep) * 0.1 + self.global_mean

                    # 拼接本地和全局特征
                    features = torch.cat([rep.detach(), global_features.detach()], dim=0)
                    domain_labels = torch.cat([
                        torch.zeros(rep.size(0), 1).to(self.device),  # 本地特征标签=0
                        torch.ones(global_features.size(0), 1).to(self.device)  # 全局特征标签=1
                    ], dim=0)
                    #
                    # 计算领域分类损失
                    domain_preds = self.domain_classifier(features)
                    domain_loss = F.binary_cross_entropy(domain_preds, domain_labels)

                    # 更新领域分类器
                    self.opt_domain.zero_grad()
                    domain_loss.backward()
                    self.opt_domain.step()
                    #
                    reversed_features = grad_reverse(rep, alpha=1)
                    domain_preds = self.domain_classifier(reversed_features)
                    adversarial_loss = F.binary_cross_entropy(
                        domain_preds,
                        torch.ones(rep.size(0), 1).to(self.device)  # 欺骗领域分类器
                    ) + torch.mean(0.5 * (self.running_mean - self.global_mean) ** 2) * self.klw
                    # #print(adversarial_loss)
                    #
                    # #reg_loss = torch.mean(0.5 * (self.running_mean - self.global_mean) ** 2)  # 本地特征接近全局特征的损失
                    #
                    min_loss = self.mine_loss(self.running_mean, self.global_mean, mine_model)
                    # #print('min_loss:',min_loss)
                    # # 注意力特征融合（新增核心部分）
                    fused_rep = self.fuse_features(rep)

                    # 将原始head计算改为使用融合后的特征
                    output = self.model.head(fused_rep*0.1 + rep+self.client_mean)
                    loss = self.loss(output, y)

                    #loss = self.loss(self.model.head(rep + self.client_mean), y)
                    # print('loss:',loss)
                    # print('att_loss:',min_loss)
                    total_loss = loss + adversarial_loss * self.adv_weight + min_loss * 0.1
                else:
                    output = self.model.head(rep)
                    total_loss = self.loss(output, y)
                # ====== end

                self.opt_client_mean.zero_grad()
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)  # 最大梯度范数阈值
                self.optimizer.step()
                self.opt_client_mean.step()
                self.detach_running()


        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()


        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def reset_running_stats(self):
        self.running_mean.zero_()
        self.num_batches_tracked.zero_()

    def detach_running(self):
        self.running_mean.detach_()

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                # output = self.model.head(rep + self.client_mean)
                # loss = self.loss(output, y)
                fused_rep = self.fuse_features(rep)

                # 将原始head计算改为使用融合后的特征
                output = self.model.head(fused_rep*0.1 +rep+self.client_mean)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        reps = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)

                fused_rep = self.fuse_features(rep)

                # 将原始head计算改为使用融合后的特征
                output = self.model.head(fused_rep*0.1 +rep+self.client_mean)
                #loss = self.loss(output, y)

                #output = self.model.head(rep + self.client_mean)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)
                reps.extend(rep.detach())

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc


# def mutual_information_loss(local_features, global_features):
#     """
#     计算本地特征和全局特征之间的互信息损失，通过最小化它们之间的点积来减少冗余信息。
#     """
#     # 归一化特征向量，使其具有相同的尺度
#     local_features = F.normalize(local_features, p=2, dim=-1)
#     global_features = F.normalize(global_features, p=2, dim=-1)
#
#     # 计算本地特征和全局特征的点积，作为它们之间的相似性度量
#     similarity = torch.sum(local_features * global_features, dim=-1)  # 点积
#
#     # 我们希望减少它们之间的相似性，所以最小化点积
#     mi_loss = -similarity.mean()  # 负的点积作为互信息的近似
#     return mi_loss
    def mine_loss(self,x, y, model):
        """
        MINE Loss: Maximizes mutual information between two variables X and Y
        """
        # Compute the forward pass of the MINE model
        x = x.to(self.device)
        y = y.to(self.device)
        mi_score = model(x, y)

        # Maximize the mutual information by taking the average of the MI score
        #mi_loss = -mi_score.mean()  # Negative of the average score to maximize MI

        return - mi_score  # Negate the loss to maximize mutual information

    def fuse_features(self, current_features):
        """
        双路注意力特征融合
        Args:
            current_features: 当前特征 [B, D]
        Returns:
            fused_features: 融合特征 [B, D]
        """
        # 扩展历史特征维度 [D] -> [B, D]
        historical_features = self.client_mean.unsqueeze(0).expand_as(current_features)

        # 执行双路注意力融合
        fused_features = self.dual_attention(
            curr=current_features,
            hist=historical_features
        )
        return fused_features

    # 在客户端训练代码中添加
    def get_feature_statistics(self, dataloader):
        features = []
        self.model.eval()
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                z = self.model.base(x)
                features.append(z)
        features = torch.cat(features)
        mu = torch.mean(features, dim=0)
        sigma = torch.std(features, dim=0)
        return mu.cpu().numpy(), sigma.cpu().numpy()