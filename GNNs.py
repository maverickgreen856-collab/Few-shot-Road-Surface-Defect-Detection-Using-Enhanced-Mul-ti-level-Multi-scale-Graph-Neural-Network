import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import to_dense_batch
import math

class KANLinear(nn.Module):
    """KAN增强的线性层"""
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # 基础线性变换
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.base_bias = nn.Parameter(torch.Tensor(out_features))
        
        # 自适应样条基函数参数
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size))
        
        # 网格点参数
        self.grid = nn.Parameter(torch.Tensor(in_features, grid_size))
        
        # 可学习的缩放因子
        self.lambda_res = nn.Parameter(torch.tensor(1.0))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.uniform_(self.spline_weight, -0.1, 0.1)
        nn.init.uniform_(self.grid, -1, 1)
        if self.base_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.base_bias, -bound, bound)
    
    def b_spline_basis(self, x, grid, k=3):
        """B样条基函数计算"""
        batch_size = x.size(0)
        n_features = x.size(1)
        grid_size = grid.size(1)
        
        # 扩展x和grid用于向量化计算
        x_expanded = x.unsqueeze(-1).expand(-1, -1, grid_size)
        grid_expanded = grid.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 计算相对距离
        diff = x_expanded - grid_expanded
        
        # 零阶B样条
        basis = (diff >= 0).float() * (diff < 1).float()
        
        # 递归计算高阶B样条
        for order in range(1, k + 1):
            basis_prev = basis.clone()
            for j in range(grid_size - order):
                term1 = (diff[:, :, j] - grid[:, j]) / (grid[:, j + order] - grid[:, j] + 1e-8) * basis_prev[:, :, j]
                term2 = (grid[:, j + order + 1] - diff[:, :, j]) / (grid[:, j + order + 1] - grid[:, j + 1] + 1e-8) * basis_prev[:, :, j + 1]
                basis[:, :, j] = term1 + term2
        
        return basis
    
    def forward(self, x):
        # 基础线性变换
        base_output = F.linear(x, self.base_weight, self.base_bias)
        
        # 样条基函数变换
        spline_basis = self.b_spline_basis(x, self.grid, self.spline_order)
        spline_output = torch.einsum('bik,oik->bo', spline_basis, self.spline_weight)
        
        # 合并输出
        output = base_output + spline_output
        return output

class KANGINEConv(MessagePassing):
    """KAN增强的图卷积层"""
    def __init__(self, in_channels, out_channels, edge_dim=None):
        super(KANGINEConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        
        # 节点特征更新网络
        self.kan_node = KANLinear(in_channels, out_channels)
        
        # 边特征处理网络
        if edge_dim is not None:
            self.kan_edge = KANLinear(edge_dim, out_channels)
        else:
            self.kan_edge = None
        
        # 残差连接的线性投影
        self.res_proj = nn.Linear(in_channels, out_channels)
        
        # 可学习的缩放因子
        self.lambda_res = nn.Parameter(torch.tensor(1.0))
        
        # 层归一化
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x, edge_index, edge_attr=None):
        # 消息传递
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # 节点特征更新
        out = self.kan_node(out)
        out = F.gelu(out)
        out = self.norm(out)
        
        # 残差连接
        res = self.res_proj(x)
        out = out + self.lambda_res * res
        
        return out
    
    def message(self, x_j, edge_attr):
        # 节点特征
        msg = x_j
        
        # 边特征处理
        if edge_attr is not None and self.kan_edge is not None:
            edge_msg = self.kan_edge(edge_attr)
            msg = msg + edge_msg
        
        return msg

class MultiScaleFeatureFusion(nn.Module):
    """多尺度特征融合机制"""
    def __init__(self, num_scales, feature_dim, attention_dim=64):
        super(MultiScaleFeatureFusion, self).__init__()
        self.num_scales = num_scales
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        
        # 尺度级注意力参数
        self.W_q = nn.Linear(feature_dim, attention_dim)
        self.W_k = nn.Linear(feature_dim, attention_dim)
        self.W_v = nn.Linear(feature_dim, attention_dim)
        
        # 缩放因子
        self.scale = math.sqrt(attention_dim)
    
    def forward(self, features_list, batch=None):
        """
        Args:
            features_list: 多尺度特征列表 [H^(1), H^(2), ..., H^(L)]
            batch: 批处理索引
        """
        batch_size = features_list[0].size(0) if batch is None else batch.max().item() + 1
        
        # 转换为密集批次格式
        dense_features = []
        masks = []
        
        for features in features_list:
            dense_feat, mask = to_dense_batch(features, batch)
            dense_features.append(dense_feat)
            masks.append(mask)
        
        # 全局平均池化获取尺度级特征
        scale_features = []
        for i, (feat, mask) in enumerate(zip(dense_features, masks)):
            # 对有效节点进行平均池化
            feat_sum = (feat * mask.unsqueeze(-1)).sum(dim=1)
            node_count = mask.sum(dim=1, keepdim=True)
            scale_feat = feat_sum / torch.clamp(node_count, min=1)
            scale_features.append(scale_feat)
        
        # 构建全局特征矩阵 [batch_size, num_scales, feature_dim]
        G_mean = torch.stack(scale_features, dim=1)
        
        # 尺度级注意力
        Q = self.W_q(G_mean)  # [batch_size, num_scales, attention_dim]
        K = self.W_k(G_mean)  # [batch_size, num_scales, attention_dim]
        V = self.W_v(G_mean)  # [batch_size, num_scales, feature_dim]
        
        # 计算注意力权重
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        A_scale = F.softmax(attention_scores, dim=-1)  # [batch_size, num_scales, num_scales]
        
        # 注意力加权
        G_attended = torch.bmm(A_scale, V)  # [batch_size, num_scales, feature_dim]
        
        # 计算每个尺度的注意力权重
        scale_weights = F.softmax(G_attended, dim=1)  # [batch_size, num_scales, feature_dim]
        
        # 多尺度特征融合
        fused_features = torch.zeros_like(features_list[0])
        
        for i, features in enumerate(features_list):
            # 获取当前尺度的权重
            weight = scale_weights[:, i, :].mean(dim=-1, keepdim=True)  # [batch_size, 1]
            
            # 扩展权重以匹配特征维度
            if batch is not None:
                weight_expanded = weight[batch]
            else:
                weight_expanded = weight.expand(features.size(0), -1)
            
            # 加权求和
            fused_features = fused_features + weight_expanded * features
        
        return fused_features

class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, feature_dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert self.head_dim * num_heads == feature_dim, "feature_dim必须能被num_heads整除"
        
        # QKV投影
        self.W_q = nn.Linear(feature_dim, feature_dim)
        self.W_k = nn.Linear(feature_dim, feature_dim)
        self.W_v = nn.Linear(feature_dim, feature_dim)
        
        # 输出投影
        self.W_o = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, feature_dim = x.size()
        
        # 线性投影得到Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 应用mask（如果有）
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # softmax归一化
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 注意力加权
        attended = torch.matmul(attention_weights, V)
        
        # 合并多头
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, feature_dim)
        
        # 输出投影
        output = self.W_o(attended)
        
        return output

class NodeLevelAttention(nn.Module):
    """节点级注意力机制"""
    def __init__(self, feature_dim, num_heads=8, dropout=0.1):
        super(NodeLevelAttention, self).__init__()
        self.multihead_attention = MultiHeadAttention(feature_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, batch=None, mask=None):
        """
        Args:
            x: 节点特征 [num_nodes, feature_dim]
            batch: 批处理索引
            mask: 注意力mask
        """
        if batch is not None:
            # 转换为密集格式
            x_dense, mask_dense = to_dense_batch(x, batch)
        else:
            x_dense = x.unsqueeze(0)  # [1, num_nodes, feature_dim]
            mask_dense = mask.unsqueeze(0) if mask is not None else None
        
        # 多头自注意力
        attn_output = self.multihead_attention(x_dense, mask_dense)
        
        # 残差连接和层归一化
        output_dense = self.norm(x_dense + self.dropout(attn_output))
        
        # 转换回稀疏格式
        if batch is not None:
            output = output_dense[mask_dense]
        else:
            output = output_dense.squeeze(0)
        
        return output

class DualAttentionFusion(nn.Module):
    """双注意力融合模块"""
    def __init__(self, feature_dim, num_scales, num_heads=8, dropout=0.1):
        super(DualAttentionFusion, self).__init__()
        self.scale_attention = MultiScaleFeatureFusion(num_scales, feature_dim)
        self.node_attention = NodeLevelAttention(feature_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(feature_dim)
    
    def forward(self, features_list, batch=None):
        """
        Args:
            features_list: 多尺度特征列表
            batch: 批处理索引
        """
        # 尺度级注意力融合
        scale_fused = self.scale_attention(features_list, batch)
        
        # 节点级注意力
        node_attended = self.node_attention(scale_fused, batch)
        
        # 最终输出
        output = self.norm(scale_fused + node_attended)
        
        return output

class KANEnhancedGNN(nn.Module):
    """KAN增强的多尺度注意力图神经网络"""
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 num_layers=3,
                 edge_dim=None,
                 num_heads=8,
                 dropout=0.1):
        super(KANEnhancedGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.edge_dim = edge_dim
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 多尺度图卷积层
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim
            self.gnn_layers.append(
                KANGINEConv(in_channels, hidden_dim, edge_dim)
            )
        
        # 双注意力融合模块
        self.dual_attention = DualAttentionFusion(
            hidden_dim, num_layers, num_heads, dropout
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 输入投影
        x = self.input_proj(x)
        
        # 多尺度特征提取
        features_list = []
        current_features = x
        
        for i, gnn_layer in enumerate(self.gnn_layers):
            current_features = gnn_layer(current_features, edge_index, edge_attr)
            current_features = self.dropout(current_features)
            features_list.append(current_features)
        
        # 双注意力融合
        fused_features = self.dual_attention(features_list, batch)
        
        # 全局池化
        graph_features = global_mean_pool(fused_features, batch)
        
        # 分类
        logits = self.classifier(graph_features)
        
        return logits

class MultiScaleKANGNN(nn.Module):
    """完整的多尺度KAN增强GNN模型（用于路面缺陷检测）"""
    def __init__(self, 
                 input_dim=512,  # 假设从CNN backbone提取的特征维度
                 hidden_dim=256,
                 num_classes=3,  # 鳄鱼裂缝、常规裂缝、坑洞
                 num_layers=4,
                 edge_dim=None,
                 num_heads=8,
                 dropout=0.2):
        super(MultiScaleKANGNN, self).__init__()
        
        self.gnn = KANEnhancedGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_layers=num_layers,
            edge_dim=edge_dim,
            num_heads=num_heads,
            dropout=dropout
        )
    
    def forward(self, data):
        return self.gnn(data)

# 使用示例
if __name__ == "__main__":
    # 模型参数
    input_dim = 512  # 从CNN特征提取器得到的特征维度
    hidden_dim = 256
    num_classes = 3  # 三类缺陷
    num_layers = 4
    num_heads = 8
    
    # 创建模型
    model = MultiScaleKANGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("模型结构:")
    print(model)