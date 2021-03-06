import time
import cv2
import numpy as np


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes):
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    label = str(classes[class_id]) + " " + str(round(confidence, 4) * 100) + "%"
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def get_result_predict(img, cfg_path, weight_path, conf_threshold = 0.1, nms_threshold=0.9, dim=(416, 416)):
    # Load file weight và cfg
    height, width = img.shape[:2]
    scale = 1/255.0
    net = cv2.dnn.readNet(cfg_path, weight_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # Đưa ảnh vào blob object
    blob = cv2.dnn.blobFromImage(img, scale, dim, (0, 0, 0), True, crop=False)
    net.setInput(blob)
    # Get output
    outs = net.forward(get_output_layers(net))

    
    class_ids = []
    confidences = []
    boxes = []

    start = time.time()
    # print(outs)
    for out in outs:
        print(out.shape)
        for detection in out:
            scores = detection[5:] # Xác suất các lớp
            # print(scores)
            class_id = np.argmax(scores) # Lấy vị trí có xác suất lón nhất
            confidence = scores[class_id]
            # Tính tọa độ topleft và width height
            center_x = detection[0] * width
            center_y = detection[1] * height
            w = detection[2] * width
            h = detection[3] * height
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])
    
    # print(confidences)
    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    result = []
    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        result.append([class_ids[i], confidences[i], x, y, x + w, y + h])

    end = time.time()
    print("YOLO Execution time: " + str(end-start))

    # Trả về topleft và bottomright [class_ids, confidences, x, y, x + w, y + h]
    return result


if __name__=="__main__":
    # cfg_path = "./cr_net_aug/cr-net.cfg"
    # weight_path = "./cr_net_aug/cr-net_best.weights"
    # with open("./cr_net_aug/yolo.names", 'r') as f:
    #     classes = [line.strip() for line in f.readlines()]

    cfg_path = "./cr_net_focal/crnet4imblanceddata.cfg"
    weight_path = "./cr_net_focal/crnet4imblanceddata_best.weights"
    with open("./cr_net_focal/crnet4imblanceddata.names", 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    import os
    path = "./test_img_rec"
    path_save = "./result_recognize_focal"
    for f in os.listdir(path):
        fpath = os.path.join(path, f)
        image = cv2.imread(fpath)
        results = get_result_predict(image, cfg_path, weight_path, conf_threshold = 0.5, nms_threshold=0.3, dim=(352, 128))

        to_save = ""
        with open(os.path.join(path_save, f.split(".")[0] + ".txt"), 'w') as f:
            for result in results:
                class_id, confidence, tl_x, tl_y, br_x, br_y = result
                to_save = f"{classes[class_id]} {confidence} {round(tl_x)} {round(tl_y)} {round(br_x)} {round(br_y)}"
                f.write(to_save + "\n")
                print(to_save)
            print("_____")



    # image = cv2.imread("./test_img_rec/816.jpg")
    # # image = cv2.imread("./test.jpg")
    # h, w = image.shape[:2]

    

    # # cfg_path = "./cr_net/crnet.cfg"
    # # weight_path = "./cr_net/crnet_best.weights"
    # # with open("./cr_net/crnet.names", 'r') as f:
    # #     classes = [line.strip() for line in f.readlines()]
    
    # # cfg_path = "./cr_net_aug/cr-net.cfg"
    # # weight_path = "./cr_net_aug/cr-net_best.weights"
    # # with open("./cr_net_aug/yolo.names", 'r') as f:
    # #     classes = [line.strip() for line in f.readlines()]

    # cfg_path = "./cr_net_focal/crnet4imblanceddata.cfg"
    # weight_path = "./cr_net_focal/crnet4imblanceddata_best.weights"
    # with open("./cr_net_focal/crnet4imblanceddata.names", 'r') as f:
    #     classes = [line.strip() for line in f.readlines()]

    # results = get_result_predict(image, cfg_path, weight_path, conf_threshold = 0.5, nms_threshold=0.3, dim=(352, 128))
    # first_line = []
    # second_line = []
    # for result in results:
    #     class_id, confidence, tl_x, tl_y, br_x, br_y = result
    #     # print(result)
    #     if 0 < (tl_y + br_y) / 2 < h / 2:
    #         first_line.append(result)
    #     else:
    #         second_line.append(result)
        
    #     draw_prediction(image, class_id, confidence, round(tl_x), round(tl_y), round(br_x), round(br_y), classes)

    # first_line = sorted(first_line, key=lambda x:x[2])
    # second_line = sorted(second_line, key=lambda x:x[2])
    # result = "".join([*[classes[x[0]] for x in first_line], *[classes[y[0]] for y in second_line]])
    
    # print(result)

    # image = cv2.resize(image, (h*3, w*3))
    # cv2.imshow("predict", image)
    # cv2.waitKey(0)
    
