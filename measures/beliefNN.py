import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import itertools
import matplotlib.pyplot as plt
from mpmath import mp

device='cuda:1'


class EvidenceModel:
    def __init__(self, Beta, bias):
        '''
        input:
            Beta (B): a neural network model head paramenters
            Phi (M):  feature vectors
        '''
        self.B = Beta
        self.bias = bias
        self.evidence_weights = None
        self.w_pos = None
        self.w_neg = None
        self.eta_pos_temp = None
        self.eta_neg_temp = None
        self.K = None
        self.weight = None


    def get_evidence_weights(self,Phi):
        # import pdb;pdb.set_trace()
        J, K = self.B.shape
    
        with torch.no_grad():
            B = self.B
            Phi = Phi
            M = Phi.reshape(1,-1)
            # M = Phi.unsqueeze(0)
            M = torch.mean(M, dim=0, keepdims=True)
            B_star = B - B.mean(axis=0, keepdims=True)
            # import pdb;pdb.set_trace()
            
            self.weight = B_star*M.reshape(J,1).to(device) 
            W = ((self.bias.reshape(1,K) + torch.mm(M, B_star)) / J) - self.weight
            # W = ((torch.mm(M, B_star.T)) / B.shape[0]).expand(J, K)
        # import pdb;pdb.set_trace()
        self.evidence_weights = self.weight.unsqueeze(0) + W.reshape(1,J,K)
        self._calculate_basic_terms(Phi)
        return self.evidence_weights

    def _calculate_basic_terms(self,phi):
        # import pdb;pdb.set_trace()
        omega_jk_positive = torch.relu(self.evidence_weights)
        omega_jk_negative = torch.relu(-self.evidence_weights)
        self.w_pos1 = omega_jk_positive.sum(1)[0].unsqueeze(0)
        self.w_neg1 = omega_jk_negative.sum(1)[0].unsqueeze(0)
        self.K = self.w_pos1.shape[-1]
        self.eta_pos_temp = 1 / (torch.exp(self.w_pos1).sum(dim=1) - self.K + 1)
        self.eta_neg_temp = 1 / (1 - torch.prod(1 - torch.exp(-self.w_neg1), dim=1))

        # Handle zeros in w_pos
        sorted_w_pos = torch.sort(self.w_pos1.flatten())[0]
        second_smallest = sorted_w_pos[torch.nonzero(sorted_w_pos > 0, as_tuple=True)[0][0]]
        w_pos1_copy = self.w_pos1.clone()  # 创建 w_pos1 的副本
        w_pos1_copy[w_pos1_copy == 0] = second_smallest  # 修改副本中的值
        self.w_pos2 = w_pos1_copy  # 将修改后的副本赋值给 w_pos2
        # Handle zeros in w_neg
        sorted_w_neg = torch.sort(self.w_neg1.flatten())[0]
        second_smallest = sorted_w_neg[torch.nonzero(sorted_w_neg > 0, as_tuple=True)[0][0]]
        w_neg1_copy = self.w_neg1.clone()  # 创建 w_neg1 的副本
        w_neg1_copy[w_neg1_copy == 0] = second_smallest  # 修改副本中的值
        self.w_neg2 = w_neg1_copy  # 将修改后的副本赋值给 w_neg2
        # Calculate kappa
        self.kappa = torch.sum(self.eta_pos_temp.reshape(-1, 1) * (torch.exp(self.w_pos2) - 1) * (1 - self.eta_neg_temp.reshape(-1, 1) * torch.exp(-self.w_neg1)), dim=1)
        self.eta_temp = 1 / (1 - self.kappa)

    def get_evidence_conflict(self):
        return self.kappa

    def get_evidence_ignorance(self):
        # Calculate ignorance value using precomputed terms
        mp.dps=20
        w_neg_np = self.w_neg2.detach().cpu().numpy()
        # exp_results = mp.exp(-mp.fsum(w_neg_np))
        exp_results = [mp.exp(-mp.fsum(row)) for row in w_neg_np]
        w_neg_sum = np.array(exp_results).sum()
        ig = self.eta_temp.detach().cpu().numpy() * self.eta_pos_temp.detach().cpu().numpy() * self.eta_neg_temp.detach().cpu().numpy() * w_neg_sum
        # exp_results = torch.exp(-torch.sum(self.w_neg2, dim=1))
        # w_neg_sum = exp_results.sum()
        # ig = self.eta_temp * self.eta_pos_temp * self.eta_neg_temp * w_neg_sum
        return ig

    def get_nonspecific(self):
        # Calculate non-specificity value using precomputed terms
        eta_mul = self.eta_temp * self.eta_pos_temp * self.eta_neg_temp
        mp.dps = 20
        first_term = eta_mul.reshape(-1, 1) * torch.exp(-self.w_neg2)
        # import pdb;pdb.set_trace()
        w_neg_np = self.w_neg2.detach().cpu().numpy()
        # prod_term = mp.fprod(1 - mp.exp(-w_neg_np), dim=1, keepdim=True) / (1 - mp.exp(-w_neg_np))
        prod_term = (np.array([mp.fprod([(1-mp.exp(-x)) for x in row]) for row in w_neg_np]))/(1-np.array([[mp.exp(-x) for x in row] for row in w_neg_np]))
        w_pos_np=self.w_pos1.detach().cpu().numpy()
        second_term=[[(mp.exp(x)-1) for x in row] for row in w_pos_np]+prod_term
        
        m_theta = first_term.detach().cpu().numpy() * second_term

        # prod_term = torch.prod(1 - torch.exp(-self.w_neg2), dim=1, keepdim=True) / (1 - torch.exp(-self.w_neg2))
        # second_term = (torch.exp(self.w_pos1) - 1) + prod_term
        # first_term = eta_mul.reshape(-1, 1) * torch.exp(-self.w_neg2)
        # m_theta = first_term * second_term
        return 1 - m_theta.sum()

    def compute_m(self, labels):
        w_neg = self.w_neg1.squeeze()
        w_pos = self.w_pos1.squeeze()
        mass_function = dict()
        subsets = self.generate_subsets(labels)
        for theta_set in subsets:
            if len(theta_set) == 0:
                m_value = 0
            elif len(theta_set) == 1:
                k = theta_set[0]
                m_value = torch.exp(-w_neg[k]) * (torch.exp(w_pos[k]) - 1 + torch.prod(torch.tensor([1 - torch.exp(-w_neg[l]) for l in range(len(w_neg)) if l != k])))
            elif len(theta_set) > 1:
                prod_not_in_A = np.prod([1 - np.exp(-w_neg[k]) for k in range(len(w_neg)) if k not in theta_set])
                prod_in_A = np.prod([np.exp(-w_neg[k]) for k in theta_set])
                m_value = prod_not_in_A * prod_in_A
            mass_function[theta_set] = m_value
        values = list(mass_function.values())
        total = sum(values)
        mass_function = {key: value / total for key, value in mass_function.items()}
        return mass_function

    def generate_subsets(self, input_set):
        subsets = []
        for r in range(len(input_set) + 1):
            subsets.extend(itertools.combinations(input_set, r))
        return [tuple(subset) for subset in subsets]

    def pl_A(self, labels, m_values):
        subsets = self.generate_subsets(labels)
        pl = dict()
        for A in subsets:
            pl_value = 0
            for B, m_B in m_values.items():
                if set(B).intersection(A):
                    pl_value += m_B
            pl[A] = pl_value
        return pl








def get_hidden_layer_output(model,X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        hidden_output = model(X_tensor, return_hidden=True)
    return hidden_output


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 20)  # 输入层到第一隐藏层（20个ReLU单元）
        self.fc2 = nn.Linear(20, 10)  # 第一隐藏层到第二隐藏层（10个ReLU单元）
        self.output = nn.Linear(10, 3,bias=False)  # 第二隐藏层到输出层（3个类别）
        self.dropout = nn.Dropout(0.5)  # Dropout层，dropout率为0.5
        self.softmax = nn.Softmax(dim=1)  # Softmax激活用于多类别分类

    def forward(self, x, return_hidden=False):
        x = torch.relu(self.fc1(x))  # 第一层输出
        x = self.dropout(x)  # Dropout 应用在第一层
        hidden_output = torch.relu(self.fc2(x))  # 第二层输出
        output = self.output(hidden_output)  # 输出层（线性）
        output = self.softmax(output)  # Softmax 输出
        if return_hidden:
            return hidden_output  # 返回倒数第二层的输出
        return output  # 返回最终的 Softmax 输出

def predict(model, X):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)  # 转换输入为 tensor
        output = model(X_tensor)  # 前向传播计算
        _, predicted = torch.max(output, 1)  # 获取类别标签
    return predicted,output  # 转换为 numpy 数组
def plot_decision_boundary(clf, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # 预测整个网格上的结果，替换 clf.predict
    Z,_ = predict(clf, np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=ListedColormap(('black', 'red', 'green')))

    # 绘制数据点
    plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], c='green', marker='+', label='Class 1', s=15, facecolors='none')
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='black', marker='o', label='Class 2', s=15, facecolors='none')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', marker='^', label='Class 3', s=15, facecolors='none')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.legend(loc='best')
    plt.title('Bayes Decision Boundary with Data Points')
    plt.savefig('bayes_decision_boundary.png')




def plot_m1(clf,evidence_model):
    # Generate grid of w_neg and w_pos values for contour plotting
    w_neg_values = np.linspace(-3, 3, 100)
    w_pos_values = np.linspace(-3, 3, 100)
    labels = [0, 1, 2]  # Example label set
    # Compute m values for (1,) subset
    m_values = np.zeros((len(w_neg_values), len(w_pos_values)))
    import pdb; pdb.set_trace()
    for i, w_neg in enumerate(w_neg_values):
        for j, w_pos in enumerate(w_pos_values):
            phi = torch.tensor(get_hidden_layer_output(clf,torch.tensor([w_neg, w_pos]).reshape(1,2)))[0]
            evidence_model.get_evidence_weights(phi)
            # import pdb; pdb.set_trace()
            mass_function = evidence_model.compute_m(labels)
            if (1,) in mass_function:
                m_values[i, j] = mass_function[(1,)]

    # Plot the contour plot
    plt.figure(figsize=(8, 6))
    plt.contourf(w_neg_values, w_pos_values, m_values, levels=50, cmap='viridis')
    plt.colorbar(label=r'$m(\theta_1)$')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(r'Contour Plot of $m(\theta_1)$')
    plt.savefig("m1_contour.png")
def plot_m1(clf, evidence_model):
    # Generate grid of w_neg and w_pos values for contour plotting
    w_neg_values = np.linspace(-3, 3, 100)
    w_pos_values = np.linspace(-3, 3, 100)
    labels = [0, 1, 2]  # Example label set

    # Compute m values for (1,) subset
    m_values = np.zeros((len(w_neg_values), len(w_pos_values)))
    
    for i, w_neg in enumerate(w_neg_values):
        for j, w_pos in enumerate(w_pos_values):
            phi = torch.tensor(get_hidden_layer_output(clf, torch.tensor([w_neg, w_pos]).reshape(1, 2)))[0]
            evidence_model.get_evidence_weights(phi)
            mass_function = evidence_model.compute_m(labels)
            if (1,) in mass_function:
                # print(f"w_neg: {w_neg}, w_pos: {w_pos}, m_value: {mass_function[(0,)]}")
                m_values[i, j] = mass_function[(1,2,)]

    # Plot the contour plot
    plt.figure(figsize=(8, 6))
    plt.contourf(w_pos_values,w_neg_values, m_values, levels=50, cmap='viridis')
    plt.colorbar()
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(r'Contour Plot of $m(\theta_{23})$')
    plt.savefig("m23_contour.png")
    plt.show()


if __name__ == "__main__":
    with open('dataset.pkl', 'rb') as f:
        X, y = pickle.load(f)
    clf = SimpleNN()
    clf.load_state_dict(torch.load('clf.pth'))
    model_weights = clf.output.weight.T #(10,3)
    # import pdb; pdb.set_trace()
    phi = torch.tensor(get_hidden_layer_output(clf,X)) #(900,10); phi[0]:(10)
    evidence_model = EvidenceModel(model_weights)
    flag = 0
    predict_labels, predict_probs = predict(clf, X)
    plot_decision_boundary(clf, X, y)
    # import pdb; pdb.set_trace()
    plot_m1(clf,evidence_model)
    for i, predict_label in enumerate(predict_labels):
        # import pdb; pdb.set_trace()
        evidence_weights = evidence_model.get_evidence_weights(phi[i])
        conflict_value = evidence_model.get_evidence_conflict()
        nonspecific_value = evidence_model.get_nonspecific()
        ig_value = evidence_model.get_evidence_ignorance()
        # print(f"conflict_value: {conflict_value}, ig_value: {ig_value}, nonspecific_value: {nonspecific_value}")
        # import pdb; pdb.set_trace()
        mass_values = evidence_model.compute_m([0, 1, 2])
        # import pdb; pdb.set_trace()
        pl_values = evidence_model.pl_A([0, 1, 2], mass_values)
        pl_sum = pl_values[(0,)] + pl_values[(1,)] + pl_values[(2,)]
        probs = torch.tensor([pl_values[(0,)], pl_values[(1,)], pl_values[(2,)]]) / pl_sum
        if predict_label == probs.argmax():
            flag += 1
        # print(f"predict_label: {predict_label}, probs: {probs.argmax()}")
        print("predict_probs:",predict_probs[i],end=" ")
        print("evidence_probs:",probs)
    print(f"accuracy: {flag / len(X)}")
