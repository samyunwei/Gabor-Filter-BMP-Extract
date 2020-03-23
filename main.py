import cv2
from PalmRecognition import PalmRecognition
from Compare import similar


def main():
    pr = PalmRecognition("./data")
    sim = pr.comparePalm("data/001_1.bmp", "data/003_2.bmp", 0)
    print(sim)


def process():
    pr = PalmRecognition("./data")
    pr.process_images()


def test():
    test = PalmRecognition("./data")
    src_real, src_image = test.process(20, 0)
    all_count = 0
    for i in range(len(test.image_data)):
        dest_real, dest_image = test.process(i, 0)
        sim = similar(src_real, dest_real) * 100
        if sim > 70:
            all_count += 1
            print("percent:" + str((i + 1) / len(test.image_data) * 100) + "%")
            print(str(test.image_data[i]) + "sim:" + str(sim))

    print("all_count percent:%.2f" % (all_count % len(test.image_data)))


if __name__ == "__main__":
    main()
