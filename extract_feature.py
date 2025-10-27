import cv2
import numpy as np
import pandas as pd
import glob
import os

# 📁 Folder containing your cropped turtle head images
input_dir = "dataset_turtle_cropped"
# 📄 Output CSV file
output_csv = "turtle_features.csv"

# 🧠 Function to extract Gabor features
def gabor_features(img):
    feats = []
    for theta in np.arange(0, np.pi, np.pi/4):  # 4 angles
        kernel = cv2.getGaborKernel((21, 21), 5, theta, 10, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        feats.append(filtered.mean())
        feats.append(filtered.std())
    return feats

# 🐢 Loop through all cropped images
data = []
for file_path in glob.glob(os.path.join(input_dir, "*.jpg")):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Can't read {file_path}")
        continue
    features = gabor_features(img)
    filename = os.path.basename(file_path)
    data.append([filename] + features)

# 📝 Create column names
columns = ["filename"]
for t in range(4):  # 4 directions
    columns += [f"mean_theta{t}", f"std_theta{t}"]

# 🧾 Save to CSV
df = pd.DataFrame(data, columns=columns)
df.to_csv(output_csv, index=False)
print(f"✅ Feature extraction completed! Saved to {output_csv}")
