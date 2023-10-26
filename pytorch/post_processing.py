import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, roc_auc_score, f1_score


def find_contours(y_test, y_pred):
    test = y_test
    pred = y_pred.squeeze()

    ## converting to unit8
    test = cv2.convertScaleAbs(test)
    pred = cv2.convertScaleAbs(pred)

    # converting it to color image so I can show on it the contours
    pred_col = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
    test_col = cv2.cvtColor(test, cv2.COLOR_GRAY2RGB)
    pred_col[pred_col == 1] = 255
    test_col[test_col == 1] = 255

    # Find all contours in the greyscale image.
    contours_pred, hierarchy_pred = cv2.findContours(pred, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours_test, hierarchy_test = cv2.findContours(test, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    return (pred_col, contours_pred, test_col, contours_test)


def create_feature_df(img, contours_list):
    Center_x = []
    Center_y = []
    len_maj = []
    len_min = []
    Angle = []
    Area = []
    im_copy = img.copy()
    for i, cont in enumerate(contours_list):
        if len(cont) > 5:
            if cv2.contourArea(cont) > 15:
                (x, y), (MA, ma), angle = cv2.fitEllipse(cont)
                Center_x.append(x)
                Center_y.append(y)
                len_maj.append(MA)
                len_min.append(ma)
                Angle.append(angle)
                area = math.pi * MA * ma
                Area.append(area)

    list_of_columns = list(zip(Center_x, Center_y, len_maj, len_min, Angle, Area))
    contours_df = pd.DataFrame(list_of_columns, columns=['Center_x', 'Center_y', 'len_maj', 'len_min', 'Angle', 'Area'])

    return (contours_df)


def compare_test_pred_one_image(cont_df_test, cont_df_pred):
    fit_flocks = []
    cont_df_pred_index = []

    for index, row in cont_df_test.iterrows():

        low_x = row['Center_x'] - 30
        high_x = row['Center_x'] + 30
        low_y = row['Center_y'] - 30
        high_y = row['Center_y'] + 30

        if cont_df_pred['Center_x'].between(low_x, high_x).any():
            if cont_df_pred['Center_y'].between(low_y, high_y).any():

                low_size = row['Area'] * 0.3
                high_size = row['Area'] * 3
                if cont_df_pred['Area'].between(low_size, high_size).any():
                    fit_flocks.append(index)

    fit_flocks_pred = []
    for index, row in cont_df_pred.iterrows():
        low_x = row['Center_x'] - 30
        high_x = row['Center_x'] + 30
        low_y = row['Center_y'] - 30
        high_y = row['Center_y'] + 30
        if cont_df_test['Center_x'].between(low_x, high_x).any():
            if cont_df_test['Center_y'].between(low_y, high_y).any():
                low_size = row['Area'] * 0.3
                high_size = row['Area'] * 3
                if cont_df_test['Area'].between(low_size, high_size).any():
                    fit_flocks_pred.append(index)

    tp = len(fit_flocks)
    fn = cont_df_test.shape[0] - len(fit_flocks)
    fp = cont_df_pred.shape[0] - len(fit_flocks_pred)
    num_cont_test = cont_df_test.shape[0]
    num_cont_pred = cont_df_pred.shape[0]

    return tp, fn, fp, num_cont_test, num_cont_pred


def pipeline(test_np, pred_np):
    TP = []
    FN = []
    FP = []
    NUM_cont_test = []
    NUM_cont_pred = []
    for i, image_test in enumerate(test_np):
        test = test_np[i]
        pred = pred_np[i]
        pred_img, pred_conts, test_img, test_conts = find_contours(test, pred)
        test_cont_df = create_feature_df(test_img, test_conts)
        pred_cont_df = create_feature_df(pred_img, pred_conts)
        Tp, Fn, Fp, Num_cont_test, Num_cont_pred = compare_test_pred_one_image(test_cont_df, pred_cont_df)
        TP.append(Tp)
        FN.append(Fn)
        FP.append(Fp)
        NUM_cont_test.append(Num_cont_test)
        NUM_cont_pred.append(Num_cont_pred)

    list_of_columns = list(zip(TP, FN, FP, NUM_cont_test, NUM_cont_pred))
    results = pd.DataFrame(list_of_columns, columns=['TP', 'FN', 'FP', 'Num_cont_test', 'Num_cont_pred'])

    return results


y_test = torch.load("labels.pkl").numpy()
y_test = y_test.transpose(0, 2, 3, 1)
y_prob = torch.load("preds.pkl").numpy()

threshold = 0.1

y_pred = (y_prob >= threshold).astype(np.float32)

results_df = pipeline(y_test, y_pred)

tp_all = results_df["TP"].sum()
cont_test_all = results_df["Num_cont_test"].sum()

tp_success = tp_all / cont_test_all

print('TPR', tp_success)

fp_all = results_df["FP"].sum()
fp_positive = results_df.loc[results_df['FP'] > 0, 'FP'].sum()
cont_pred_all = results_df["Num_cont_pred"].sum()

fp_percent = fp_positive / cont_pred_all

print('FPDR', fp_percent)

y_test = y_test.squeeze()
predictions = y_pred.squeeze()

y_scores = predictions.reshape(predictions.shape[0] * predictions.shape[1] * predictions.shape[2], 1)
# print(y_scores.shape)

y_true = y_test.reshape(y_test.shape[0] * y_test.shape[1] * y_test.shape[2], 1)
# print(y_true.shape)


# y_scores = np.where(y_scores > threshold, 1, 0)
y_true = np.where(y_true > threshold, 1, 0)

def calc_roc_curve(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels.flatten(), probs.flatten())
    AUC_ROC = roc_auc_score(labels.flatten(), probs.flatten())
    print("\nArea under the ROC curve: " + str(AUC_ROC))
    
    plt.figure()
    plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.show()

calc_roc_curve(y_test, y_prob)

y_true = y_test.reshape(y_test.shape[0] * y_test.shape[1] * y_test.shape[2], 1)
# print(y_true.shape)


y_scores = np.where(y_scores > threshold, 1, 0)
y_true = np.where(y_true > threshold, 1, 0)

y_pred = np.empty((y_scores.shape[0]))
for i in range(y_scores.shape[0]):
    if y_scores[i] >= threshold:
        y_pred[i] = 1
    else:
        y_pred[i] = 0

F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
print("\nF1 score (F-measure): " + str(F1_score))
