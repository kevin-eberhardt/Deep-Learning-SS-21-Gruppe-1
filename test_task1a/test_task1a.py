import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

def load_images():
    data = []
    labels = []
    boxes = []
    paths = []
    file_object = open('test_labels.txt', 'r')
    counter = 0
    for row in file_object:
        if counter > 0:
            row = row.split(',')
            path, startX, startY, endX, endY, label = row
            image = cv2.imread(path)
            (h, w) = image.shape[:2]

            startX = float(startX) / w
            startY = float(startY) / h
            endX = float(endX) / w
            endY = float(endY) / h

            image = tf.keras.preprocessing.image.load_img(path)
            image = tf.keras.preprocessing.image.img_to_array(image)

            data.append(image)
            boxes.append((startX, startY, endX, endY))
            labels.append(label)
            paths.append(path)
        else:
            counter += 1

    return data, labels, boxes, paths
data, labels, boxes, paths = load_images()


def show_labeled_image(image):
    img = cv2.cvtColor(cv2.imread(image["path"]), cv2.COLOR_BGR2RGB)
    (h, w) = img.shape[:2]

    startX = int(image["boxes"][0] * w)
    startY = int(image["boxes"][1] * h)
    endX = int(image["boxes"][2] * w)
    endY = int(image["boxes"][3] * h)

    y = startY - 10 if startY - 10 > 10 else startY + 10
    label = image["label"].replace("\n", "")
    cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
    return img

test = {"image": data[21], "boxes": boxes[21], "label": labels[21], "path": paths[21]}
test_image = show_labeled_image(test)
plt.imshow(test_image)
plt.show()
for i in range(10):
    img = {"image": data[i], "boxes": boxes[i], "label": labels[i], "path": paths[i]}
    print(i)
    plt.subplot(1, 2, i + 1)
    plt.imshow(img)
    plt.title("Image #" + i)
    plt.xticks([])
    plt.yticks([])
    i += 1
plt.show()