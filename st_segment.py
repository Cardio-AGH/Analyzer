import numpy as np
from scipy import stats
import pandas as pd
class STSegment:
    def __init__(self, data, json_params, r_peaks, waves):
        self.data = data
        self.fs = json_params['fs']
        self.adc_gain = json_params['adc_gain'][0]
        self.r_peaks = r_peaks
        self.waves = waves
        self.data_len = len(self.data)
        self.STRAIGHT_THRESHOLD = [0.05, -0.05]
        self.ST_TYPE_THRESHOLD = 10.0
        self.OFFSET_TYPE_THRESHOLD = [0.1, -0.1]

    def st_segment(self):
        qrson_list, qrsoff_list, tpeak_list = self.waves_points()
        points_list = [qrson_list, qrsoff_list, tpeak_list]

        preprocessed_points_list = self.check_and_preprocess_points(points_list)
        new_rpeak_list, new_qrson_list, new_qrsoff_list, new_tpeak_list = preprocessed_points_list
        self.new_rpeak_list, new_qrson_list, new_qrsoff_list, new_tpeak_list = preprocessed_points_list

        t_on_list, t_on_values, j_x_point_ind, j_x_point_values = self.create_t_on_j_x_lists(new_qrsoff_list,
                                                                                             new_tpeak_list)

        j_x_dict = {"ind": j_x_point_ind,
                    "value": j_x_point_values}

        t_on_dict = {"ind": t_on_list,
                     "value": t_on_values}

        offset_list = (j_x_point_values - self.new_qrson_values) / self.adc_gain
        offset_level_type = [self.determine_offset_level_type(offset, self.OFFSET_TYPE_THRESHOLD) for offset in
                             offset_list]

        output_data = self.st_segmentation_process(j_x_dict, t_on_dict)

        degree_sign = u'\N{DEGREE SIGN}'
        slope_name = f"slope ({degree_sign})"

        columns_list = ["start", "end", "type", slope_name]

        st_segments_data_df = pd.DataFrame(output_data, columns=columns_list)
        st_segments_data_df["offset (mv)"] = np.round(offset_list, 3)
        st_segments_data_df["offset_type"] = offset_level_type

        type_counts = st_segments_data_df.type.value_counts()
        offset_type_counts = st_segments_data_df.offset_type.value_counts()

        output_dict = {"st_data": st_segments_data_df,
                       "class_data": type_counts,
                       "offset_type_data": offset_type_counts}

        return output_dict

    def determine_offset_level_type(self, offset, thresholds_list):
        th1 = thresholds_list[0]
        th2 = thresholds_list[1]
        offset_type = ''
        if offset > th1:
            offset_type = "elevated"
        elif offset < th2:
            offset_type = "depression"
        elif th1 > offset > th2:
            offset_type = "normal"

        return offset_type

    def waves_points(self):
        qrson_list = self.waves['ECG_QRS_Onsets']
        qrsoff_list = self.waves['ECG_QRS_Offsets']
        tpeak_list = self.waves['ECG_T_Offsets']

        return qrson_list, qrsoff_list, tpeak_list

    def check_and_preprocess_points(self, points_list):
        qrson_list, qrsoff_list, tpeak_list = points_list
        new_rpeak_list = []
        new_qrson_list = []
        new_qrsoff_list = []
        new_tpeak_list = []

        for rpeak in self.r_peaks:

            new_qrson = qrson_list[(qrson_list > rpeak-100) & (qrson_list < rpeak)]
            new_qrsoff = qrsoff_list[(qrsoff_list > rpeak) & (qrsoff_list < rpeak+100)]
            new_tpeak = tpeak_list[(tpeak_list > rpeak) & (tpeak_list < rpeak +200)]

            condition_1 = new_qrson.shape == new_qrsoff.shape == new_tpeak.shape
            condition_2 = new_qrson.size == 0
            if condition_1 and not condition_2:
                new_rpeak_list.append(rpeak)
                new_qrson_list.append(new_qrson[0])
                new_qrsoff_list.append(new_qrsoff[0])
                new_tpeak_list.append(new_tpeak[0])

            else:
                pass

        self.new_rpeak_values = self.data[new_rpeak_list]
        self.new_qrson_values = self.data[new_qrson_list]
        self.new_qrsoff_values = self.data[new_qrsoff_list]
        self.new_tpeak_values = self.data[new_tpeak_list]

        new_lists_return = [new_rpeak_list, new_qrson_list, new_qrsoff_list, new_tpeak_list]
        return new_lists_return

    def signal_bpm(self, rpeak_list, fs):

        diff = np.diff(rpeak_list) / fs * 1000
        bpm = 60000 / np.mean(diff)

        return bpm

    def get_X_value_for_J_X_point(self, bpm):

        if bpm < 100:
            X_value_ms = 90
        elif 100 <= bpm < 110:
            X_value_ms = 72
        elif 110 <= bpm < 120:
            X_value_ms = 64
        elif bpm >= 120:
            X_value_ms = 60

        X_value_samples = round((X_value_ms * self.fs) / 1000)

        return X_value_samples

    def create_t_on_j_x_lists(self, new_qrsoff_list, new_tpeak_list):

        t_on_list = []
        for ind1, ind2 in zip(new_qrsoff_list, new_tpeak_list):
            out = self.extract_t_on(ind1, ind2)
            if out:
                out += ind1
            else:
                out = 0
            t_on_list.append(out)

        t_on_values = self.data[t_on_list]

        x_value = 20
        bpm = self.signal_bpm(self.new_rpeak_list, self.fs)
        x_value = self.get_X_value_for_J_X_point(bpm)

        j_x_point_ind = np.array(new_qrsoff_list) + x_value
        j_x_point_values = self.data[j_x_point_ind]

        return t_on_list, t_on_values, j_x_point_ind, j_x_point_values

    def extract_t_on(self, ind1, ind2):

        data2 = self.data[ind1:ind2]

        # calculate gradient
        grad_signal = np.gradient(data2, 2)

        # grad signal max
        grad_signal_max = np.max(grad_signal)
        grad_signal_max_ind = np.argmax(grad_signal)

        try:
            # index array for gradient signal
            grad_signal_sub_ind = np.arange(grad_signal_max_ind - 30, grad_signal_max_ind)
            # values for gradient signal
            grad_signal_sub = grad_signal[grad_signal_sub_ind]

        except IndexError:
            return None

        # threshold
        grad_signal_threshold = grad_signal_max / 6

        # x closes value and index
        x_closest = min(grad_signal_sub, key=lambda x: abs(x - grad_signal_threshold))
        x_closest_ind = np.where(grad_signal == x_closest)[0][0]

        return x_closest_ind

    def calculate_max_distance(self, data, line):

        diff = np.abs(np.subtract(data, line))
        max_value = np.max(diff)
        return max_value

    def determine_st_type(self, max_, threshold):

        if max_ > threshold:
            return "curve"
        else:
            return "straight"

    def determine_straight_type(self, slope, thresholds_list):
        th1 = thresholds_list[0]
        th2 = thresholds_list[1]
        straight_type = ''
        if slope > th1:
            straight_type = "upsloping"
        elif slope < th2:
            straight_type = "downsloping"
        elif th1 > slope > th2:
            straight_type = "horizontal"

        return straight_type

    def calculate_points_above_line(self, data, line):

        diff = np.subtract(data, line)
        diff_plus = diff[diff > 0.0]
        diff_plus_len = len(diff_plus)
        return diff_plus_len

    def calculate_points_under_line(self, data, line):

        diff = np.subtract(data, line)
        diff_minus = diff[diff < 0.0]
        diff_minus_len = len(diff_minus)
        return diff_minus_len

    def determine_curve_type(self, data, line):
        data_len = len(data)

        n_above_line = self.calculate_points_above_line(data, line)
        concave_ratio = n_above_line / data_len

        n_under_line = self.calculate_points_under_line(data, line)
        convex_ratio = n_under_line / data_len

        curve_type = ""

        if convex_ratio < 0.7 and concave_ratio < 0.7:
            curve_type = None

        elif convex_ratio > 0.7 and concave_ratio > 0.7:
            curve_type = None

        elif concave_ratio > 0.7:
            curve_type = "concave"
        elif convex_ratio > 0.7:
            curve_type = "convex"

        return curve_type

    def calculate_slope_degrees(self, slope):

        output = np.rad2deg(np.arctan(slope))
        return output

    def fit_data_to_line(self, x, slope, intercept):

        y_out = x * slope + intercept
        return y_out

    def st_segmentation_process(self, j_x_dict, t_on_dict):

        j_x_point_ind = j_x_dict["ind"]
        j_x_point_values = j_x_dict["value"]

        t_on_list = t_on_dict["ind"]
        t_on_values = t_on_dict["value"]

        output_data = []

        for i, (jx_ind, jx_v, t_on_ind, t_on_v) in enumerate(
                zip(j_x_point_ind, j_x_point_values, t_on_list, t_on_values)):

            i_st_segment_data = []
            x_ = [jx_ind, t_on_ind]

            if jx_ind >= t_on_ind:
                i_st_segment_data = [None, None, None, None]
                output_data.append(i_st_segment_data)
                continue

            i_st_segment_data.append(x_[0])
            i_st_segment_data.append(x_[1])

            y_ = [jx_v, t_on_v]
            slope, intercept, _, _, _ = stats.linregress(x_, y_)

            x_to_fit = np.arange(x_[0], x_[1])
            y_fitted = self.fit_data_to_line(x_to_fit, slope, intercept)

            # Calculate slope in degrees
            slope_degrees = self.calculate_slope_degrees(slope)
            slope_degrees = np.round(slope_degrees, 2)

            # Calculate max distance between line and
            signal_sub = self.data[x_[0]:  x_[1]]
            max_distance = self.calculate_max_distance(signal_sub, y_fitted)

            st_type = self.determine_st_type(max_distance, self.ST_TYPE_THRESHOLD)

            if st_type == "curve":
                curve_type = self.determine_curve_type(signal_sub, y_fitted)
                i_st_segment_data.append(curve_type)


            elif st_type == "straight":
                straight_type = self.determine_straight_type(slope, self.STRAIGHT_THRESHOLD)
                i_st_segment_data.append(straight_type)

            i_st_segment_data.append(slope_degrees)
            output_data.append(i_st_segment_data)

        return output_data
