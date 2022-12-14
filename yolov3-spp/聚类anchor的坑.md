## 聚类anchor需要注意的坑

有时使用自己聚类得到的anchors的效果反而变差了，此时可以从一下几方面进行检查：

1. 注意输入网络时训练的图片尺寸。这个是很重要的点，因为一般训练/验证时输入网络的图片尺寸是固定的，比如说640 x 640，那么图片在输入网络前一般会将最大边长缩放到640，同时图片中的bboxes也会进行缩放。所以在聚类anchors时需要使用相同的方式提前去缩放bboxes，否则聚类出来的anchors不匹配。比如图片都是1280 x 1280，假设bboxes都是100 x 100大小的，如果不去缩放bboxes，那么聚类得到的anchors差不多是在100 x 100附近。而实际训练网络时bboxes都已经缩放到50 x 50大小了，此时理想的anchors应该是50 x 50左右而不是100 x 100了。
2. 如果使用预训练权重，不要冻结太多的权重。现在训练自己的数据集时一般都是使用别人在coco等大型数据集上预训练好的权重。而这些权重是基于coco等数据集上聚类得到的结果，并不是针对自己数据集聚类得到的。所以网络为了要适应新的anchors需要调整很多权重，如果你冻结了很多的层（假设只去微调最后的预测期，其他全中全部冻结），那么得到的结果很大几率还没有之前的anchors好。当可训练的权重越来越多，一般是用自己的数据集聚类得到的anchors会更好一点（前提是自己聚类的anchors是合理的）。