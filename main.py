import cv2
from PalmRecognition import PalmRecognition
from Compare import similar
import re


def main():
    pr = PalmRecognition("./data")
    # pr.show_gabor_filters()
    # pr.show_different_gabor_filteringResult(1)
    # sim = pr.comparePalm("data/001_1.bmp", "data/003_4.bmp", 0)
    # print(sim)
    test(30)


def process():
    pr = PalmRecognition("./data")
    pr.process_images()


def test(src_index):
    test = PalmRecognition("./data")
    src_real, src_image = test.process(src_index, 0)
    all_count = 0
    true_err = 0
    false_err = 0
    reg_id = re.compile(r"(\d+)_(\d+)")
    src_id = reg_id.search(test.image_data[src_index]).group(0)
    for i in range(len(test.image_data)):
        dest_real, dest_image = test.process(i, 0)
        sim = similar(src_real, dest_real) * 100
        id = reg_id.search(test.image_data[i]).group(0)
        if sim > 70:
            if id != src_id:
                false_err += 1
            else:
                all_count += 1
            print("percent:" + str((i + 1) / len(test.image_data) * 100) + "%")
            print(str(test.image_data[i]) + "sim:" + str(sim))
        else:
            if id == src_id:
                true_err += 1

    print("right percent:%.2f %%" % ((len(test.image_data) - true_err - false_err) / len(test.image_data) * 100))


if __name__ == "__main__":
    main()
