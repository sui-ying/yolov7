

def xyxy_four_point_convert_xywh(size, box):
    """
    :param size: (width, height)
    :param box:  [label x1 y1 x2 y2  pt1x pt1y pt2x pt2y pt3x pt3y pt4x pt4y]
    :return:
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[1] + box[3]) / 2.0
    y = (box[2] + box[4]) / 2.0
    w = box[3] - box[1]
    h = box[4] - box[2]

    # center x, y;  w, h
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    # up_left, up_right, down_right, down_left
    pt0_x = box[5] * dw
    pt0_y = box[6] * dh
    pt1_x = box[7] * dw
    pt1_y = box[8] * dh
    pt2_x = box[9] * dw
    pt2_y = box[10] * dh
    pt3_x = box[11] * dw
    pt3_y = box[12] * dh
    return x, y, w, h, pt0_x, pt0_y, pt1_x, pt1_y, pt2_x, pt2_y, pt3_x, pt3_y


def xywh_convert_xxyy_four_point(size, box):
    """
    xywh: list ->  cls, x, y, w, h, pt0_x, pt0_y, pt1_x, pt1_y, pt2_x, pt2_y, pt3_x, pt3_y
    size: tuple -> (width, height)
    xxyy: top-left, bottom-right
    """
    # where xmin, ymin=top-left, xmax,ymax=bottom-right
    xmin = size[0] * (float(box[1]) - float(box[3]) / 2)
    ymin = size[1] * (float(box[2]) - float(box[4]) / 2)
    xmax = xmin + size[0] * float(box[3])
    ymax = ymin + size[1] * float(box[4])

    pt0_x = int(size[0] * (float(box[5])))
    pt0_y = int(size[1] * (float(box[6])))
    pt1_x = int(size[0] * (float(box[7])))
    pt1_y = int(size[1] * (float(box[8])))
    pt2_x = int(size[0] * (float(box[9])))
    pt2_y = int(size[1] * (float(box[10])))
    pt3_x = int(size[0] * (float(box[11])))
    pt3_y = int(size[1] * (float(box[12])))

    return box[0], int(xmin), int(ymin), int(xmax), int(
        ymax), pt0_x, pt0_y, pt1_x, pt1_y, pt2_x, pt2_y, pt3_x, pt3_y