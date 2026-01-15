import kagglehub, os

base = kagglehub.dataset_download('aryashah2k/brain-tumor-segmentation-brats-2019')
for root, _, files in os.walk(base):
    for f in files:
        print(os.path.join(root, f))