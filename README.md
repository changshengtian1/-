pip install torch torchvision rasterio numpy matplotlib albumentations timm  # hao
import albumentations as A  # hao
from albumentations.pytorch import ToTensorV2  # hao

transform = A.Compose([
    A.Resize(256, 256),  # hao
    A.HorizontalFlip(),  # hao
    A.RandomBrightnessContrast(),  # hao
    A.RandomGamma(),  # hao
    A.RandomCrop(width=224, height=224),  # hao
    ToTensorV2()  # hao
])  # hao


def load_image(image_path):  # hao
    with rasterio.open(image_path) as src:  # hao
        bands = src.read([2, 3, 4, 8])  # Sentinel-2: Blue, Green, Red, NIR  # hao
        image = np.stack(bands, axis=-1)  # 堆叠成 (H, W, 4) 的形状  # hao
        image = image.astype(np.float32)  # hao
        image /= 10000.0  # 归一化  # hao
        augmented = transform(image=image)  # hao
        return augmented['image']  # hao
        
def load_label(label_path):  # hao
    with rasterio.open(label_path) as src:  # hao
        label = src.read(1)  # 假设标签为单通道  # hao
        label = label.astype(np.float32)  # hao
        label /= 255.0  # 归一化标签  # hao
        return label  # hao
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

class Encoder(nn.Module):  # hao
    def __init__(self, in_channels, out_channels):  # hao
        super(Encoder, self).__init__()  # hao
        self.res1 = ResidualBlock(in_channels, out_channels)  # hao
        self.res2 = ResidualBlock(out_channels, out_channels)  # hao

    def forward(self, x):  # hao
        x = self.res1(x)  # hao
        x = self.res2(x)  # hao
        return x  # hao
        class ImprovedASPP(nn.Module):  # hao
    def __init__(self, in_channels, out_channels):  # hao
        super(ImprovedASPP, self).__init__()  # hao
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

class AdvancedCBAM(nn.Module):  # hao
    def __init__(self, in_channels):  # hao
        super(AdvancedCBAM, self).__init__()  # hao
        self.channel_attention = ChannelAttention(in_channels)  # hao
        self.spatial_attention = SpatialAttention()  # hao

    def forward(self, x):  # hao
        x = self.channel_attention(x) * x  # hao
        x = self.spatial_attention(x) * x  # hao
        return x  # hao
        class CBAM_ASPP_U_Net_Transformer(nn.Module):  # hao
    def __init__(self, in_channels, out_channels):  # hao
        super(CBAM_ASPP_U_Net_Transformer, self).__init__()  # hao

        # 编码器（带残差块）
        self.encoder = Encoder(in_channels, 64)  # hao
        self.aspp = ImprovedASPP(64, 64)  # hao
        self.cbam = AdvancedCBAM(64)  # hao

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
model = CBAM_ASPP_U_Net_Transformer(in_channels=4, out_channels=1).to(device)  # hao
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
    import matplotlib.pyplot as plt


def plot_result(input_image, output_image):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(input_image)
    axes[0].set_title("Input Image")
    axes[1].imshow(output_image[0], cmap='gray')
    axes[1].set_title("Predicted Mask")
    plt.show()

plot_result(image, output)
