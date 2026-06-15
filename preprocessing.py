import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

# ==================================
# DATASET PATH
# ==================================
DATASET_PATH = "CK+48"

EMOTIONS = [
    "anger",
    "happy",
    "sadness",
    "surprise"
]

# ==================================
# COUNT IMAGES
# ==================================
def count_images(dataset_path):

    counts = {}

    total = 0

    for emotion in EMOTIONS:

        folder = os.path.join(dataset_path, emotion)

        if os.path.exists(folder):
            count = len(os.listdir(folder))
        else:
            count = 0

        counts[emotion] = count
        total += count

    return counts, total


# ==================================
# LOAD SAMPLE IMAGES
# ==================================
def load_images(dataset_path, samples_per_class=20):

    images = []

    for emotion in EMOTIONS:

        folder = os.path.join(dataset_path, emotion)

        if not os.path.exists(folder):
            print(f"Folder missing: {folder}")
            continue

        files = os.listdir(folder)[:samples_per_class]

        for file in files:

            img_path = os.path.join(folder, file)

            img = cv2.imread(img_path)

            if img is not None:
                images.append((img, emotion))

    return images


# ==================================
# DISPLAY SAMPLE IMAGES
# ==================================
def show_samples(images):

    rows = 4
    cols = 20

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(20, 8)
    )

    count = 0

    for i in range(rows):

        for j in range(cols):

            if count >= len(images):
                axes[i, j].axis("off")
                continue

            img, label = images[count]

            img_rgb = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            axes[i, j].imshow(img_rgb)
            axes[i, j].set_title(label)
            axes[i, j].axis("off")

            count += 1

    plt.tight_layout()
    plt.show()


# ==================================
# PLOT CLASS DISTRIBUTION
# ==================================
def plot_distribution(counts):

    emotions = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(8, 5))

    sns.barplot(
        x=emotions,
        y=values,
        palette="plasma"
    )

    plt.title("CK+48 Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Number of Images")

    plt.show()


# ==================================
# MAIN
# ==================================
if __name__ == "__main__":

    if not os.path.exists(DATASET_PATH):

        print(
            f"Dataset folder '{DATASET_PATH}' not found."
        )

        print(
            "Expected structure:"
        )

        print("""
CK+48/
├── anger/
├── happy/
├── sadness/
└── surprise/
        """)

        exit()

    counts, total = count_images(DATASET_PATH)

    print(f"\nTotal Images: {total}\n")

    for emotion, count in counts.items():
        print(f"{emotion}: {count}")

    images = load_images(DATASET_PATH)

    show_samples(images)

    plot_distribution(counts)