import torch
import time
import matplotlib.pyplot as plt
from torchvision import models
from torch.utils.data import TensorDataset, DataLoader

#Simulating Dataset
img_count = 10580
img_size = (3, 224, 224) #ResNet50 input: 3 channels, 224x224 resolution
img_classes = 120 #Stanford Dogs dataset has 120 breeds(classes)

#Creating synthetic image data and labels
synthetic_imgs = torch.rand((img_count, *img_size)) #Random pixel values
synthetic_labels = torch.randint(0, img_classes, (img_count, )) #Random labels 0 and 119

#Wrap the data into a tensordataset
synthetic_dataset = TensorDataset(synthetic_imgs, synthetic_labels)

#Dataloader for batch processing
batch_size = 256
sd_dataloader = DataLoader(synthetic_dataset, batch_size=batch_size, shuffle=False)

#Loading Pretrained ResNet50
sd_model = models.resnet50(pretrained=True)
sd_model.eval() #Set to Model evaluation mode

#Function for Benchmark inference
def bench_inf(model, dataloader, device):
    model = model.to(device)
    total_time = 0
    with torch.no_grad():
        for batch in dataloader:
            images, _ = batch
            images = images.to(device) #Moving image data to the same device as the model
            start = time.time()
            _ = model(images) #Perform inference
            total_time += time.time() - start
    return total_time

#Benchmark on CPU
cpu_time = bench_inf(sd_model, sd_dataloader, "cpu")
print(f"CPU Inference Time: {cpu_time: .2f} seconds")

#Benchmark on GPU
if torch.cuda.is_available():
    gpu_time = bench_inf(sd_model, sd_dataloader, "cuda")
    print(f"GPU Inference Time: {gpu_time: .2f} seconds")
    print(f"Speedup: {cpu_time/gpu_time: .2f}x")

#Visualizing Benchmark Inference
labels = ['CPU', 'GPU'] if torch.cuda.is_available() else ['CPU']
times = [cpu_time, gpu_time] if torch.cuda.is_available() else [cpu_time]

plt.bar(labels, times, color=['blue', 'green'] if torch.cuda.is_available() else ['blue'])
plt.ylabel('Time (Seconds)')
plt.title("Total Inference time: CPU vs GPU (Synthetic Stanford Dogs)")

for i, j in enumerate(times):
    plt.text(i, j + 1, f"{j: .2f}s", ha='center', fontsize=10)
plt.show()