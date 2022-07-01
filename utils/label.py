class Labels(object):

    def __init__(self, keys, values):
        self.label_classes = {}
        for _, (k, v) in enumerate(zip(keys, values)):
            self.label_classes[k] = v

    def label_to_name(self, label):
        res = None
        for k, v in self.label_classes.items():
            if v == label:
                res = k
                break
        if res is None:
            raise ValueError(
                "Label ({}) is not belong to label_classes {}".format(
                    label, self.label_classes))
        return res

    def name_to_label(self, name):
        res = None
        for k, v in self.label_classes.items():
            if k == name:
                res = v
                break
        if res is None:
            raise ValueError(
                "label_name ({}) is not belong to label_classes {}".format(
                    name, self.label_classes))
        return res


polyGon_classes = Labels(["left_page", "right_page"], [0, 1])

ocr_number_classes = Labels(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

det_classes = Labels([
    "correct_sign", "wrong_sign", "page_number", "Review_A", "Review_B",
    "Review_C", "Review_D", "Review_E", "work_duration"
], [0, 1, 2, 3, 4, 5, 6, 7, 8])
