import os.path
import torch
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
from scipy.cluster.vq import kmeans


def wh_iou(wh1, wh2):
    '''
    比较宽高时，默认框的左上角对齐
    Args:
        wh1: N个bboxes的宽高，dim是N * 2
        wh2: k个clusters的宽高，dim是k * 2

    Returns: wh1的每个元素和wh2的每个元素的iou，dim是N * k
    '''
    wh1 = wh1[:, None]  # N 1 2
    wh2 = wh2[None]     # 1 k 2
    # 通过广播机制变成N k 2，每一个bboxes的宽高和k个cluster的宽高对比，取小的那个，然后w*h得到inter面积
    inter = np.minimum(wh1, wh2).prod(2)  # N * k
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)

def k_means(boxes, k, update_center=np.median):
    '''
    簇中心点的选择方式为w和h的中位数，也可用平均数，距离表征使用1-IoU，不使用欧氏距离
    Args:
        boxes: 需要聚类的bboxes的宽高组成的数组
        k: 簇数（聚成几类）
        update_center: 更新簇坐标的方法

    Returns: k个簇的中心点，即k个anchor的宽高
    '''
    box_number = boxes.shape[0]
    last_nearest = np.zeros((box_number,))

    # init k clusters
    clusters = boxes[np.random.choice(box_number, k, replace=False)]
    print(clusters.shape)

    while True:
        distances = 1 - wh_iou(boxes, clusters)  # N * k
        current_nearest = np.argmin(distances, axis=1)
        if (last_nearest == current_nearest).all():
            break

        # 分别更新每个簇中心
        for cluster in range(k):
            clusters[cluster] = update_center(boxes[current_nearest==cluster], axis=0)

        last_nearest = current_nearest

    return clusters

def anchor_fitness(wh: np.ndarray, clusters: np.ndarray, thr: float):
    # 分两步：1.先找bboxes每个元素对应k个聚类中心中每个元素宽高比值中最离谱的；2.再找bboxes每个元素和哪个聚类中心匹配度最好

    # 分别计算bboxes中的每一个元素（即w和h的元组）的w和h，对k个聚类结果中w和h的比值
    r = wh[:, None] / clusters[None]  # N * k * 2
    # 将bboxes中的每个元素对k个聚类结果中的每个元素的比值统一度量，小于1的就取r里面的元素，大于1的就取1/r里面的元素。N * k * 2
    # 在维度2上找最小的，即找到宽高中最离谱的一个， N * k
    x = np.minimum(r, 1. / r).min(2)  # ratio metric
    # x = wh_iou(wh, k)  # iou metric
    # 在维度1，即k个聚类中心中，为bboxes中的每个元素找到匹配度最高的聚类中心（比值最接近1）
    best = x.max(1)
    # 求平均匹配度，筛除匹配度小于阈值
    f = (best * (best > thr).astype(np.float32)).mean()  # fitness
    # 求bboxes的召回率，是否全部bboxes都可以用这几个聚类中心找到
    bpr = (best > thr).astype(np.float32).mean()  # best possible recall
    return f, bpr

def get_info(path):
    # 读取所有数据集labels的bboxes和图片wh
    # 本数据集图片大小一致，只读取一张
    img_path = r"D:\CV\Erke\data\images\000beee2-BBC_China_8520.jpg"
    img = Image.open(img_path).convert("RGB")
    # print(img.size) # w, h
    im_wh = img.size
    # print(type(im_wh))
    all_label_files = os.listdir(path)
    label_nums = len(all_label_files)

    assert label_nums > 0, "no label.txt in path, please check it~~~"

    boxes_wh = list()
    # i = 1
    for i in range(label_nums):
        label_path = os.path.join(path, all_label_files[i])
        cls_box = np.loadtxt(label_path).reshape(-1, 5)
        # if i == 1:
        #     print(cls_box)
        #     print(cls_box.shape)
        #     print(type(cls_box))
        #     i = 0
        if not len(cls_box):
            os.remove(label_path)

        box_wh = cls_box[:, 3:]
        boxes_wh.append(box_wh)

    # print(len(boxes_wh))
    # print(boxes_wh[0])

    return im_wh, boxes_wh

def main(img_size=512, n=9, thr=0.25, gen=1000):
    # 从数据集中读取所有图片的wh以及对应bboxes的wh
    # tuple, list(ndarray)
    im_wh, boxes_wh = get_info(r"D:\CV\Erke\data\labels")

    # 最大边缩放到img_size
    im_wh = np.array(im_wh, dtype=np.float32).reshape(-1, 2).repeat(len(boxes_wh), axis=0)
    shapes = img_size * im_wh / im_wh.max(1, keepdims=True)

    # 将原图尺寸的bboxes按比例缩放到img_size大小  N * 2
    wh0 = np.concatenate([l * s for s, l in zip(shapes, boxes_wh)])

    # Filter 过滤掉小目标
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print(f"WARNING: Extremely small objects found~~~ {i} of {len(wh0)} labels are < 3 pixels in size.")

    wh = wh0[(wh0 >= 2.0).any(1)] # 只保留wh都大于等于2个像素的box

    # Kmeans calculation 欧氏距离作为距离表征
    # print(f'Running kmeans for {n} anchors on {len(wh)} points...')
    # s = wh.std(0)  # sigmas for whitening
    # k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    # assert len(k) == n, print(f'ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}')
    # k *= s

    # 1 - IoU作为距离表征
    k = k_means(wh, n)
    # 按面积排序
    k = k[np.argsort(k.prod(1))]  # sort small to large
    f, bpr = anchor_fitness(k, wh, thr)
    print("kmeans: " + " ".join([f"[{int(i[0])}, {int(i[1])}]" for i in k]))
    print(f"fitness: {f:.5f}, best possible recall: {bpr:.5f}")

    # Evolve
    # 遗传算法(在kmeans的结果基础上变异mutation)，将聚类中心，随机乘以某个范围内的值，计算匹配度，循环1000次，找fitness最高的。
    npr = np.random
    f, sh, mp, s = anchor_fitness(k, wh, thr)[0], k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc=f'Evolving anchors with Genetic Algorithm:')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg, bpr = anchor_fitness(kg, wh, thr)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'

    # 按面积排序
    k = k[np.argsort(k.prod(1))]  # sort small to large
    print("genetic: " + " ".join([f"[{int(i[0])}, {int(i[1])}]" for i in k]))
    print(f"fitness: {f:.5f}, best possible recall: {bpr:.5f}")





if __name__ == "__main__":
    main()













































