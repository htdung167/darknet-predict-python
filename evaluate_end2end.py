import os

def read_txt(path):
    with open(path, 'r') as f:
        txt = f.read()
    txt = [x for x in txt.split('\n') if len(x) > 2]
    return txt


def evaluate_end2end(predict_path, gt_path):
    count_pred = 0
    count_true = 0
    for fname in os.listdir(predict_path):
        pname_pred = os.path.join(predict_path, fname)
        pname_gt = os.path.join(gt_path, fname)
        pred = read_txt(pname_pred)
        gt = read_txt(pname_gt)
        for lp in pred:
            count_pred += 1
            if lp in gt:
                count_true += 1
            else:
                print(fname, gt[0], lp)
    return count_true / count_pred * 100

# txt = 'aaa\n  '
# txt = [x for x in txt.split('\n') if len(x) > 2]
# print(txt)

if __name__=='__main__':
    path_noaug_nohe = "./result_end2end_noaug_nohe"
    path_aug_nohe = "./result_end2end_aug_nohe"
    path_noaug_he = "./result_end2end_noaug_heuristic"
    path_aug_he = "./result_end2end_aug_heuristic"
    path_gt = "./gt_test_end2end"
    # print(f"Độ chính xác của noaug_nohe: {evaluate_end2end(path_noaug_nohe, path_gt)}")
    # print("####################################")
    # print(f"Độ chính xác của aug_nohe: {evaluate_end2end(path_aug_nohe, path_gt)}")
    print(f"Độ chính xác của noaug_he: {evaluate_end2end(path_noaug_he, path_gt)}")
    print("####################################")
    print(f"Độ chính xác của aug_he: {evaluate_end2end(path_aug_he, path_gt)}")
