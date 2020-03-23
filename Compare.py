def hash_img(img):
    a = []
    hash_img = ''
    width, height = 128, 128
    for y in range(width):
        avg = sum(img[y]) / len(img[y])  # 计算每一行的像素平均值
        for x in range(width):
            if img[y][x] >= avg:  # 生成特征序列,如果此点像素大于平均值则为1,反之为0
                hash_img += '1'
            else:
                hash_img += '0'

    return hash_img


def similar(img1, img2):  # 求相似度
    hash1 = hash_img(img1)  # 计算img1的特征序列
    hash2 = hash_img(img2)  # 计算img2的特征序列
    differnce = 0
    for i in range(len(hash1)):
        differnce += abs(int(hash1[i]) - int(hash2[i]))
    similar = 1 - (differnce / len(hash1))
    return similar
