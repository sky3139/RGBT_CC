import cv2


def eval_game(output_src, target, L=0):
    output_img = output_src[0][0].cpu().numpy()
    target = target[0]

    H, W = target.shape
    if H==0:
        print(len(target))
    ratio = H / output_img.shape[0]
    # cv2.imshow("a",output_img)
    # cv2.waitKey(0)
    
    # print(W,H,ratio)
    output_img = cv2.resize(output_img, (W, H), interpolation=cv2.INTER_CUBIC) / (ratio*ratio)

    assert output_img.shape == target.shape

    # eg: L=3, p=8 p^2=64
    p = pow(2, L)
    abs_error = 0
    square_error = 0
    for i in range(p):
        for j in range(p):
            output_block = output_img[i*H//p:(i+1)*H//p, j*W//p:(j+1)*W//p]
            target_block = target[i*H//p:(i+1)*H//p, j*W//p:(j+1)*W//p]

            abs_error += abs(output_block.sum()-target_block.sum().float())
            square_error += (output_block.sum()-target_block.sum().float()).pow(2)

    return abs_error, square_error


def eval_relative(output, target):
    output_num = output.cpu().data.sum()
    target_num = target.sum().float()
    relative_error = abs(output_num-target_num)/target_num
    return relative_error