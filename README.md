pip install torch torchvision rasterio numpy matplotlib albumentations timm  # hao
import rasterio
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 读取影像文件，允许选择波段
def load_image(image_path, bands=[2, 3, 4, 8]):  # 默认为 Sentinel-2: B2, B3, B4, B8
    with rasterio.open(image_path) as src:
        # 读取指定波段
        bands_data = src.read(bands)  # 读取选择的波段
        image = np.stack(bands_data, axis=-1)  # 堆叠成 (H, W, N) 的形状，其中 N 为选择的波段数
        image = image.astype(np.float32)
        image /= 10000.0  # 归一化（根据卫星影像类型，这里假设是 Sentinel-2）
    return image  # 返回选择波段的影像数据

# 示例：读取 Sentinel-2 的波段 B2, B3, B4, B8
image = load_image("sentinel_image.tif", bands=[2, 3, 4, 8])  # 选择 Blue, Green, Red, NIR 波段
print(image.shape)  # 输出图像的形状

# 数据增强
transform = A.Compose([
    A.Resize(256, 256),  # hao
    A.HorizontalFlip(),  # hao
    A.RandomBrightnessContrast(),  # hao
    A.RandomGamma(),  # hao
    A.RandomCrop(width=224, height=224),  # hao
    ToTensorV2()  # hao
])

# 读取和预处理数据
def preprocess_image(image_path, bands=[2, 3, 4, 8]):  # hao
    image = load_image(image_path, bands)  # hao
    augmented = transform(image=image)  # hao
    return augmented['image']  # hao
    import torch
import torch.nn as nn
import torch.nn.functional as F
# CBAM
class CBAM(nn.Module):  # hao
    def __init__(self, in_channels, reduction=16):  # hao
        super(CBAM, self).__init__()  # hao
        self.channel_attention = ChannelAttention(in_channels, reduction)  # hao
        self.spatial_attention = SpatialAttention()  # hao

    def forward(self, x):  # hao
        x = self.channel_attention(x) * x  # hao
        x = self.spatial_attention(x) * x  # hao
        return x  # hao

class ChannelAttention(nn.Module):  # hao
    def __init__(self, in_channels, reduction):  # hao
        super(ChannelAttention, self).__init__()  # hao
        self.mlp = nn.Sequential(  # hao
            nn.AdaptiveAvgPool2d(1),  # hao
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),  # hao
            nn.ReLU(inplace=True),  # hao
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),  # hao
            nn.Sigmoid()  # hao
        )  # hao

    def forward(self, x):  # hao
        return self.mlp(x)  # hao

class SpatialAttention(nn.Module):  # hao
    def __init__(self):  # hao
        super(SpatialAttention, self).__init__()  # hao
        self.conv1 = nn.Conv2d(2, 1, 7, padding=3)  # hao
        self.sigmoid = nn.Sigmoid()  # hao

    def forward(self, x):  # hao
        avg_out = torch.mean(x, dim=1, keepdim=True)  # hao
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # hao
        x_out = torch.cat([avg_out, max_out], dim=1)  # hao
        x_out = self.conv1(x_out)  # hao
        return self.sigmoid(x_out)  # hao
   # ASPP
   class ASPP(nn.Module):  # hao
    def __init__(self, in_channels, out_channels):  # hao
        super(ASPP, self).__init__()  # hao
        self.dilations = [1, 6, 12, 18]  # hao
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d) for d in self.dilations])  # hao
        self.pool = nn.AdaptiveAvgPool2d(1)  # hao
        self.fc = nn.Conv2d(in_channels, out_channels, 1)  # hao
        self.bn = nn.BatchNorm2d(out_channels)  # hao
        self.relu = nn.ReLU(inplace=True)  # hao

    def forward(self, x):  # hao
        res = [conv(x) for conv in self.convs]  # hao
        pool = self.pool(x)  # hao
        pool = self.fc(pool)  # hao
        pool = F.interpolate(pool, size=x.size()[2:], mode='bilinear', align_corners=False)  # hao
        res.append(pool)  # hao
        x = torch.cat(res, dim=1)  # hao
        x = self.bn(x)  # hao
        return self.relu(x)  # hao
    # CBAM-ASPP-U-Net
    class ResidualBlock(nn.Module):  # hao
    def __init__(self, in_channels, out_channels):  # hao
        super(ResidualBlock, self).__init__()  # hao
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # hao
        self.bn1 = nn.BatchNorm2d(out_channels)  # hao
        self.relu = nn.ReLU(inplace=True)  # hao
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)  # hao
        self.bn2 = nn.BatchNorm2d(out_channels)  # hao
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)  # hao

    def forward(self, x):  # hao
        identity = self.skip(x)  # hao
        out = self.conv1(x)  # hao
        out = self.bn1(out)  # hao
        out = self.relu(out)  # hao
        out = self.conv2(out)  # hao
        out = self.bn2(out)  # hao
        out += identity  # hao
        out = self.relu(out)  # hao
        return out  # hao

class CBAM_ASPP_U_Net(nn.Module):  # hao
    def __init__(self, in_channels, out_channels):  # hao
        super(CBAM_ASPP_U_Net, self).__init__()  # hao

        # 编码器
        self.encoder = Encoder(in_channels, 64)  # hao
        self.aspp = ASPP(64, 64)  # hao
        self.cbam = CBAM(64)  # hao

        # Transformer编码器
        self.transformer = TransformerEncoder(64, 128)  # hao

        # 解码器
        self.decoder = nn.Conv2d(128, out_channels, 3, padding=1)  # hao

    def forward(self, x):  # hao
        # 编码器
        x = self.encoder(x)  # hao
        x = self.aspp(x)  # hao
        x = self.cbam(x)  # hao

        # Transformer编码器
        x = self.transformer(x)  # hao

        # 解码器
        x = self.decoder(x)  # hao
        return x  # hao
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # hao
model = CBAM_ASPP_U_Net(in_channels=4, out_channels=1).to(device)  # hao
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # hao
criterion = nn.BCEWithLogitsLoss()  # hao

# 训练循环
num_epochs = 10  # hao
for epoch in range(num_epochs):  # hao
    model.train()  # hao
    running_loss = 0.0  # hao
    for images, labels in train_loader:  # hao
        images, labels = images.to(device), labels.to(device)  # hao

        optimizer.zero_grad()  # hao
        outputs = model(images)  # hao
        loss = criterion(outputs, labels)  # hao
        loss.backward()  # hao
        optimizer.step()  # hao

        running_loss += loss.item()  # hao

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")  # hao
    import matplotlib.pyplot as plt  # hao

# 可视化结果
def plot_result(input_image, output_image):  # hao
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # hao
    axes[0].imshow(input_image)  # hao
    axes[0].set_title("Input Image")  # hao
    axes[1].imshow(output_image[0], cmap='gray')  # hao
    axes[1].set_title("Predicted Mask")  # hao
    plt.show()  # hao

# 示例输出
plot_result(image, output)  # hao
