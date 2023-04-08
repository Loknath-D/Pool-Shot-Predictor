from Custom_Object_Detection import YOLO_Custom_Object_Detection
from Contour_Segmentation import Contour_Segmentor
import cv2
import numpy as np
import matplotlib.path as mplPath
import math
import cvzone

# Initialize all variables
cap = cv2.VideoCapture("Shot-Predictor-Video.mp4");
resize_width_ratio, resize_height_ratio = 0.85, 0.85;
h, w = None, None;
class_names_dict = {
    0 : "Pool Table",
    1 : "Table Pockets",
    2 : "Cue Stick",
    3 : "Cue Ball",
    4 : "Pool Ball"
    };
objDetect = YOLO_Custom_Object_Detection();
contours = Contour_Segmentor();
cue_stick = None;
cue_stick_hsv1 = [29, 19, 114];
cue_stick_hsv2 = [94, 132, 220];
CB_hit_zones = [i for i in range(70, 121)];
CS_impression = dict.fromkeys(CB_hit_zones);
CS = [];
CB = [];
OB = [];
GB = [];
cpos, opos = [], [];
cue_ball_hit, obj_ball_hit = False, False;
pool_table = None;
table_pockets = list();
Xpos, Ypos = [], [];
xList = [];
angle = None;
predictX, predictY = None, None;
predict = None;
path_start_point = None;

# Load the Custom Trained Object Detection Model (Trained on Pool Game essentials)
objDetect.initialize_model(trained_model_path = ".\\train\\weights\\last.pt", class_details = class_names_dict);

# Lambda function to find distance between 2 points
dist = lambda x1, y1, x2, y2: int(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2));

# Method to obtain Cue Stick orientation points
def obtain_cue_stick_points(cue_stick_info = None, hit_zones = None, cbx = None, cby = None):
    cs = [];
    csx1, csy1, csx2, csy2 = None, None, None, None;
    if (cue_stick_info is not None and hit_zones is not None):
        for i in hit_zones:
            for point in cue_stick[0]['cnt']:
                if (point[0][0] == cbx - i or point[0][0] == cbx + i):
                    if (cby - i <= point[0][1] <= cby + i):
                        if (csx1 is None):
                            csx1 = point[0][0]; csy1 = point[0][1];
                        else:
                            csx2 = point[0][0]; csy2 = point[0][1];
                elif (point[0][1] == cby - i or point[0][1] == cby + i):
                    if (cbx - i <= point[0][0] <= cbx + i):
                        if (csx1 is None):
                            csx1 = point[0][0]; csy1 = point[0][1];
                        else:
                            csx2 = point[0][0]; csy2 = point[0][1];
            if (csx1 is not None and csx2 is not None):
                cs.append([round((csx1 + csx2) / 2), round((csy1 + csy2) / 2)]);
                csx1, csy1, csx2, csy2 = None, None, None, None;
    return cs;

# Method to find slope of a line
def find_slope(pt1 = None, pt2 = None):
    m = None;
    if (pt1[1] != pt2[1] and pt1[0] != pt2[0]):
        m = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0]);
    return m;

# Method to obtain the angle formed by two points
def find_angle(pt1 = None, pt2 = None, pt3 = None):
    m1 = None; m2 = None; angle = 0;
    if (pt1 is not None and pt2 is not None and pt3 is not None):
        m1 = find_slope(pt1, pt2);
        m2 = find_slope(pt1, pt3);
        if (m1 is not None and m2 is not None):
            angle = math.atan((m2 - m1) / (1 + (m2 * m1)));
            return math.degrees(angle);
        return angle;

# Method to check if a point is enclosed within the specified shape
def point_in_polygon(polygon = None, pt = None):
    if (polygon is not None and pt is not None):
        poly_path = mplPath.Path(np.array(polygon));
        if (poly_path.contains_point(pt)):
            return True;
        else:
            return False;

# Method to generate a straight path using two points and return the final point of the path along with the vicinity details of the final point
def predict_direct_path(path = [], point1 = None, point2 = None, boundary = None, pockets = None, potting_radius = 38, X = [], Y = []):
    xList = None;
    obj = False;
    if (len(path)):
        for i in path:
            if (i[1] == 'OB'):
                obj = True;
    if (len(X)):
        m, c = np.polyfit(X, Y, 1);
        if (point1[0][0] > point2[0][0]):
            xList = [i for i in range(point1[0][0], 0, -1)];
        else:
            xList = [i for i in range(point1[0][0], fw - 1)];
        for x in xList:
            y = int(round(m * x + c));
            if (boundary[0] + 5 < x < boundary[2] - 5 and boundary[1] + 5  < y < boundary[3] - 5):
                if (not len(path) or not obj):
                    if (44 <= dist(x, y, point2[0][0], point2[0][1]) <= 49):
                        return [[x, y], 'OB'];
            else:
                if (pockets is not None):
                    for i in pockets:
                        if (dist(x, y, i[0], i[1]) < potting_radius):
                            return [[x, y], 'pot'];
                for i in path:
                    if (i[0] == [x, y] or dist(i[0][0], i[0][1], x, y) < 5):
                        break;
                else:                        
                    return [[x, y], 'border'];                

# Method to find the reflected point using the point of reflection on the unknown boundary and the incidence point
def find_reflected_point(point1 = None, point2 = None, boundary = None):
    point3 = None;
    if (boundary is not None and point1 is not None and point2 is not None):
        if (point_in_polygon(polygon = [[boundary[0] - 1, boundary[1] - 1], [boundary[0] + 20, boundary[1] - 1], [boundary[0] + 20, boundary[3] + 1], [boundary[0] - 1, boundary[3] + 1]], pt = point1[0]) or
            point_in_polygon(polygon = [[boundary[2] + 1, boundary[1] - 1], [boundary[2] - 20, boundary[1] - 1], [boundary[2] - 20, boundary[3] + 1], [boundary[2] + 1, boundary[3] + 1]], pt = point1[0])):
            if (point2[0][1] < point1[0][1]):
                point3 = [[point2[0][0], point1[0][1] + (point1[0][1] - point2[0][1])]];
            elif (point2[0][1] > point1[0][1]):
                point3 = [[point2[0][0], point1[0][1] - (point2[0][1] - point1[0][1])]];
            else:
                point3 = point2;

        elif (point_in_polygon(polygon = [[boundary[0] - 1, boundary[1] - 1], [boundary[0] - 1, boundary[1] + 20], [boundary[2] + 1, boundary[1] + 20], [boundary[2] + 1, boundary[1] - 1]], pt = point1[0]) or
              point_in_polygon(polygon = [[boundary[0] - 1, boundary[3] + 1], [boundary[0] - 1, boundary[3] - 20], [boundary[2] + 1, boundary[3] - 20], [boundary[2] + 1, boundary[3] + 1]], pt = point1[0])):
            if (point2[0][0] < point1[0][0]):
                point3 = [[point1[0][0] + (point1[0][0] - point2[0][0]), point2[0][1]]];
            elif (point2[0][0] > point1[0][0]):
                point3 = [[point1[0][0] - (point2[0][0] - point1[0][0]), point2[0][1]]];
            else:
                point3 = point2;
    return point3;

# Method to plot the path of the Ghost Ball and the Object Ball along with Prediction details
def plot_path(img = None, initial_point = None, path = None, prediction = None):
    if (img is not None and initial_point is not None and prediction is not None and path is not None):
        if (prediction == 'IN'):
            colour = (0, 205, 99);
        elif (prediction == 'OUT'):
            colour = (48, 0, 205);
        if (initial_point[0][0] > path[0][0][0]):
            textbox_start_pt = (initial_point[0][0] + 30, initial_point[0][1]);
        elif (initial_point[0][0] < path[0][0][0]):
            textbox_start_pt = (initial_point[0][0] - 200, initial_point[0][1]);
        if (prediction == 'IN'):
            textbox_end_pt = (textbox_start_pt[0] + 170, textbox_start_pt[1] + 30);
        elif (prediction == 'OUT'):
            textbox_end_pt = (textbox_start_pt[0] + 190, textbox_start_pt[1] + 30);
        text_org = (textbox_start_pt[0] + 10, textbox_start_pt[1] + 22);
        text = 'Prediction: ' + prediction;
        cv2.rectangle(img, textbox_start_pt, textbox_end_pt, colour, -1);
        cv2.putText(img, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2);

        cv2.line(img, (initial_point[0][0], initial_point[0][1]), (GB[0][0][0], GB[0][0][1]), colour, 3);
        cv2.circle(img, (path[0][0][0], path[0][0][1]), 22, colour, -1);
        if (len(path) > 1):
            for i in range(len(path) - 1):
                cv2.line(img, (path[i][0][0], path[i][0][1]), (path[i + 1][0][0], path[i + 1][0][1]), colour, 3);
                cv2.circle(img, (path[i + 1][0][0], path[i + 1][0][1]), 22, colour, -1);
            

# Process the video frame-by-frame
while (cap.isOpened()):
    ret, f = cap.read();
    if ret:
        frame = cv2.resize(f, (0, 0), None, resize_width_ratio, resize_height_ratio);
        fh, fw, _ = frame.shape;

        # Perform Custom Object Detection
        results = objDetect.detect(image = frame, draw = False);
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result;

            # Obtain Pool Table Co-ordinates
            if (class_id == 0):
                if (score > 0.5):
                    pool_table = [round(x1) + 10, round(y1) + 10, round(x2) - 10, round(y2) - 10];                

            # Obtain Cue Stick shape using Contour Segmentation
            if (class_id == 2):
                if (score > 0.5):
                    cue_stick = contours.segment(img = frame[round(y1) : round(y2), round(x1) : round(x2)], hsv_range1 = cue_stick_hsv1, hsv_range2 = cue_stick_hsv2, draw = False);
                    if (cue_stick):
                        for i in cue_stick[0]['cnt']:
                            i[0][0] = i[0][0] + round(x1);
                            i[0][1] = i[0][1] + round(y1);

            # Obtain Cue Ball Co-ordinates after removing noise and check if Cue Ball is hit or not by caching the positions for every frame
            if (class_id == 3):
                if (score > 0.6):
                    CB = [[round((x1 + x2) / 2), round((y1 + y2) / 2)]];
                    if (not cue_ball_hit):
                        if (len(cpos)):
                            if (dist(CB[0][0], CB[0][1], cpos[-1][0], cpos[-1][1]) > 3):
                                cpos.append([CB[0][0], CB[0][1]]);
                                cue_ball_hit = True;
                            else:
                                CB[0][0], CB[0][1] = cpos[-1][0], cpos[-1][1];
                        else:
                            cpos.append([CB[0][0], CB[0][1]]);
                    else:
                        if (dist(CB[0][0], CB[0][1], cpos[-1][0], cpos[-1][1]) > 3):
                            cpos.append([CB[0][0], CB[0][1]]);
                        else:
                            cpos = [[cpos[-1][0], cpos[-1][1]]];
                            cue_ball_hit = False;

            # Obtain Object Ball Co-ordinates after removing noise and check if Object Ball is hit or not by caching the positions for every frame    
            if (class_id == 4):
                if (score > 0.55):
                    OB = [[round((x1 + x2) / 2), round((y1 + y2) / 2)]];
                    if (not obj_ball_hit):
                        if (len(opos)):
                            if (dist(OB[0][0], OB[0][1], opos[-1][0], opos[-1][1]) > 3):
                                opos.append([OB[0][0], OB[0][1]]);
                                obj_ball_hit = True;
                            else:
                                OB[0][0], OB[0][1] = opos[-1][0], opos[-1][1];
                        else:
                            opos.append([OB[0][0], OB[0][1]]);
                    else:
                        if (dist(OB[0][0], OB[0][1], opos[-1][0], opos[-1][1]) > 3):
                            opos.append([OB[0][0], OB[0][1]]);
                        else:
                            opos = [[opos[-1][0], opos[-1][1]]];
                            obj_ball_hit = False;

        # Obtain Table Pockets Co-ordinates using Pool Table Co-ordionates
        if (pool_table is not None):
            table_pockets = [[pool_table[0] - 7, pool_table[1] + 5], [(pool_table[0] + pool_table[2]) // 2, pool_table[1] - 15], [pool_table[2] - 15, pool_table[1] - 15], [pool_table[2], pool_table[3]], [((pool_table[0] + pool_table[2]) // 2) - 5, pool_table[3] + 15], [pool_table[0] + 15, pool_table[3]]];

        # Process the frame for Cue Stick Aim Prediction using Cut Angle Look-up Table(Does not predict the ball potting)
        if (not cue_ball_hit and not obj_ball_hit):

            # Check for the presence of Cue Stick on the Cue Ball Hit Zones
            # Cue Ball Hit Zones are the imaginary boxes around the Cue Ball. If Cue Stick appears in those zones, its impressions are cached in a dictionary called 'CS_impression'.
            # Cue Ball Hit Zones are detected only when Cue Ball is not hit. The Cue Stick is required to appear in those zones atleast for once for the Aim Prediction.
            if (cue_stick is not None):

                # Obtain Cue Stick Orientaion points
                CS = obtain_cue_stick_points(cue_stick, CB_hit_zones, CB[0][0], CB[0][1]);

                # Cache the Cue Stick impressions on the Cue Ball Hit Zones
                if (len(CS) == len(list(CS_impression.keys()))):
                    for i in CS_impression.keys():
                        CS_impression[i] = CS[list(CS_impression.keys()).index(i)];

            # Obtain the Cut Angle using Cue Stick impressions and Object Ball location
            if (None not in CS_impression.values() and len(CB)):

                # Obtain the Cue Stick projection constants
                Xpos.clear(); Ypos.clear();
                for i in CS_impression.values():
                    Xpos.append(i[0]); Ypos.append(i[1]);
                m, c = np.polyfit(Xpos, Ypos, 1);

                # Decide the projection direction based on the Cue Stick impressions and the Cue Ball location
                GB.clear();
                if (CS_impression[CB_hit_zones[0]][0] > CB[0][0]):
                    xList = [i for i in range(CS_impression[CB_hit_zones[0]][0], 0, -1)];
                else:
                    xList = [i for i in range(CS_impression[CB_hit_zones[0]][0], fw - 1)];

                # Compute the projection and decide on the Ghost Ball Co-ordinates and the Cut Angle of the Cue Ball from the Cut Angle Look-up Table
                for x in xList:
                    y = int(round(m * x + c));
                    if (pool_table[0] < x < pool_table[2] and pool_table[1] < y < pool_table[3]):
                        if (44 <= dist(x, y, OB[0][0], OB[0][1]) <= 48 and dist(x, y, CB[0][0], CB[0][1]) <= dist(OB[0][0], OB[0][1], CB[0][0], CB[0][1])):
                            GB = [[x, y]];
                        if (43.125 <= dist(x, y, OB[0][0], OB[0][1]) < 44 and dist(x, y, CB[0][0], CB[0][1]) <= dist(OB[0][0], OB[0][1], CB[0][0], CB[0][1])):
                            angle = 69.6;
                        if (40.25 <= dist(x, y, OB[0][0], OB[0][1]) < 43.125 and dist(x, y, CB[0][0], CB[0][1]) <= dist(OB[0][0], OB[0][1], CB[0][0], CB[0][1])):
                            angle = 61;
                        if (37.375 <= dist(x, y, OB[0][0], OB[0][1]) < 40.25 and dist(x, y, CB[0][0], CB[0][1]) <= dist(OB[0][0], OB[0][1], CB[0][0], CB[0][1])):
                            angle = 54.3;
                        if (34.5 <= dist(x, y, OB[0][0], OB[0][1]) < 37.375 and dist(x, y, CB[0][0], CB[0][1]) <= dist(OB[0][0], OB[0][1], CB[0][0], CB[0][1])):
                            angle = 48.6;
                        if (31.625 <= dist(x, y, OB[0][0], OB[0][1]) < 34.5 and dist(x, y, CB[0][0], CB[0][1]) <= dist(OB[0][0], OB[0][1], CB[0][0], CB[0][1])):
                            angle = 43.4;
                        if (28.75 <= dist(x, y, OB[0][0], OB[0][1]) < 31.625 and dist(x, y, CB[0][0], CB[0][1]) <= dist(OB[0][0], OB[0][1], CB[0][0], CB[0][1])):
                            angle = 38.7;
                        if (25.875 <= dist(x, y, OB[0][0], OB[0][1]) < 28.75 and dist(x, y, CB[0][0], CB[0][1]) <= dist(OB[0][0], OB[0][1], CB[0][0], CB[0][1])):
                            angle = 34.2;
                        if (23 <= dist(x, y, OB[0][0], OB[0][1]) < 25.875 and dist(x, y, CB[0][0], CB[0][1]) <= dist(OB[0][0], OB[0][1], CB[0][0], CB[0][1])):
                            angle = 30;
                        if (20.125 <= dist(x, y, OB[0][0], OB[0][1]) < 23 and dist(x, y, CB[0][0], CB[0][1]) <= dist(OB[0][0], OB[0][1], CB[0][0], CB[0][1])):
                            angle = 26;
                        if (17.25 <= dist(x, y, OB[0][0], OB[0][1]) < 20.125 and dist(x, y, CB[0][0], CB[0][1]) <= dist(OB[0][0], OB[0][1], CB[0][0], CB[0][1])):
                            angle = 22;
                        if (14.375 <= dist(x, y, OB[0][0], OB[0][1]) < 17.25 and dist(x, y, CB[0][0], CB[0][1]) <= dist(OB[0][0], OB[0][1], CB[0][0], CB[0][1])):
                            angle = 18.2;
                        if (11.5 <= dist(x, y, OB[0][0], OB[0][1]) < 14.375 and dist(x, y, CB[0][0], CB[0][1]) <= dist(OB[0][0], OB[0][1], CB[0][0], CB[0][1])):
                            angle = 14.5;
                        if (8.625 <= dist(x, y, OB[0][0], OB[0][1]) < 11.5 and dist(x, y, CB[0][0], CB[0][1]) <= dist(OB[0][0], OB[0][1], CB[0][0], CB[0][1])):
                            angle = 10.8;
                        if (5.75 <= dist(x, y, OB[0][0], OB[0][1]) < 8.625 and dist(x, y, CB[0][0], CB[0][1]) <= dist(OB[0][0], OB[0][1], CB[0][0], CB[0][1])):
                            angle = 7.2;
                        if (2.875 <= dist(x, y, OB[0][0], OB[0][1]) < 5.75 and dist(x, y, CB[0][0], CB[0][1]) <= dist(OB[0][0], OB[0][1], CB[0][0], CB[0][1])):
                            angle = 3.6;
                        if (0 <= dist(x, y, OB[0][0], OB[0][1]) < 2.875 and dist(x, y, CB[0][0], CB[0][1]) <= dist(OB[0][0], OB[0][1], CB[0][0], CB[0][1])):
                            angle = 0;
                    elif (x == pool_table[0] or x == pool_table[2] or y == pool_table[1] or y == pool_table[3]):
                        break;
                    else:
                        break;

                # Compute the final point of the projection at a specified Cut Angle from the Ghost Ball
                # For this, a polygon is formed using Cue Ball Co-ordinates, one corner of Pool Table and sweeping projection point.
                # If the Object Ball is inside the polygon, the projection point is swept in a particular direction until the specified Cut Angle is obtained.
                # If not, the Object Ball is checked if it is inside another polygon. The Pool Table is divided into 4 polygons based on the location of the Cue Ball.
                # The projection point is swept across every polygon for the Object Ball and the desired Cut Angle.
                # If the Cut Angle is 0, it is a head-on shot. If not, it is a cut shot.
                if (len(GB)):
                    if (angle != 0):
                        if (pool_table[0] - 13 <= x <= pool_table[0] + 13):
                            if (point_in_polygon(polygon = [CB[0], [x, y], [pool_table[0], pool_table[1]], [CB[0][0], pool_table[1]]], pt = OB[0])):
                                for i in range(y, pool_table[1], -1):
                                    if ((angle - 1) <= abs(find_angle(GB[0], [x, y], [x, i])) <= (angle + 1)):
                                        predictX, predictY = x, i;
                                        break;
                                else:
                                    for i in range(x, GB[0][0]):
                                        if ((angle - 1) <= abs(find_angle(GB[0], [x, y], [i, pool_table[1]])) <= (angle + 1)):
                                            predictX, predictY = i, pool_table[1];
                                            break;
                            elif (point_in_polygon(polygon = [CB[0], [x, y], [pool_table[0], pool_table[3]], [CB[0][0], pool_table[3]]], pt = OB[0])):
                                for i in range(y, pool_table[3]):
                                    if ((angle - 1) <= abs(find_angle(GB[0], [x, y], [x, i])) <= (angle + 1)):
                                        predictX, predictY = x, i;
                                        break;
                                else:
                                    for i in range(x, GB[0][0]):
                                        if ((angle - 1) <= abs(find_angle(GB[0], [x, y], [i, pool_table[3]])) <= (angle + 1)):
                                            predictX, predictY = i, pool_table[3];
                                            break;

                        elif (pool_table[2] - 13 <= x <= pool_table[2] + 13):
                            if (point_in_polygon(polygon = [CB[0], [x, y], [pool_table[2], pool_table[1]], [CB[0][0], pool_table[1]]], pt = OB[0])):
                                for i in range(y, pool_table[1], -1):
                                    if ((angle - 1) <= abs(find_angle(GB[0], [x, y], [x, i])) <= (angle + 1)):
                                        predictX, predictY = x, i;
                                        break;
                                else:
                                    for i in range(x, GB[0][0], -1):
                                        if ((angle - 1) <= abs(find_angle(GB[0], [x, y], [i, pool_table[1]])) <= (angle + 1)):
                                            predictX, predictY = i, pool_table[1];
                                            break;
                            elif (point_in_polygon(polygon = [CB[0], [x, y], [pool_table[2], pool_table[3]], [CB[0][0], pool_table[3]]], pt = OB[0])):
                                for i in range(y, pool_table[3]):
                                    if ((angle - 1) <= abs(find_angle(GB[0], [x, y], [x, i])) <= (angle + 1)):
                                        predictX, predictY = x, i;
                                        break;
                                else:
                                    for i in range(x, GB[0][0], -1):
                                        if ((angle - 1) <= abs(find_angle(GB[0], [x, y], [i, pool_table[3]])) <= (angle + 1)):
                                            predictX, predictY = i, pool_table[3]
                                            break;

                        elif (pool_table[1] - 13 <= y <= pool_table[1] + 13):
                            if (point_in_polygon(polygon = [CB[0], [x, y], [pool_table[0], pool_table[1]], [pool_table[0], CB[0][1]]], pt = OB[0])):
                                for i in range(x, pool_table[0], -1):
                                    if ((angle - 1) <= abs(find_angle(GB[0], [x, y], [i, y])) <= (angle + 1)):
                                        predictX, predictY = i, y;
                                        break;
                                else:
                                    for i in range(y, GB[0][1]):
                                        if ((angle - 1) <= abs(find_angle(GB[0], [x, y], [pool_table[0], i])) <= (angle + 1)):
                                            predictX, predictY = pool_table[0], i;
                                            break;
                            elif (point_in_polygon(polygon = [CB[0], [x, y], [pool_table[2], pool_table[1]], [pool_table[2], CB[0][1]]], pt = OB[0])):
                                for i in range(x, pool_table[2]):
                                    if ((angle - 1) <= abs(find_angle(GB[0], [x, y], [i, y])) <= (angle + 1)):
                                        predictX, predictY = i, y;
                                        break;
                                else:
                                    for i in range(y, GB[0][1]):
                                        if ((angle - 1) <= abs(find_angle(GB[0], [x, y], [pool_table[2], i])) <= (angle + 1)):
                                            predictX, predictY = pool_table[2], i;
                                            break;

                        elif (pool_table[3] - 13 <= y <= pool_table[3] + 13):
                            if (point_in_polygon(polygon = [CB[0], [x, y], [pool_table[0], pool_table[3]], [pool_table[0], CB[0][1]]], pt = OB[0])):
                                for i in range(x, pool_table[0], -1):
                                    if ((angle - 1) <= abs(find_angle(GB[0], [x, y], [i, y])) <= (angle + 1)):
                                        predictX, predictY = i, y;
                                        break;
                                else:
                                    for i in range(y, GB[0][1], -1):
                                        if ((angle - 1) <= abs(find_angle(GB[0], [x, y], [pool_table[0], i])) <= (angle + 1)):
                                            predictX, predictY = pool_table[0], i;
                                            break;
                            elif (point_in_polygon(polygon = [CB[0], [x, y], [pool_table[2], pool_table[3]], [pool_table[2], CB[0][1]]], pt = OB[0])):
                                for i in range(x, pool_table[2]):
                                    if ((angle - 1) <= abs(find_angle(GB[0], [x, y], [i, y])) <= (angle + 1)):
                                        predictX, predictY = i, y;
                                        break;
                                else:
                                    for i in range(y, GB[0][1], -1):
                                        if ((angle - 1) <= abs(find_angle(GB[0], [x, y], [pool_table[2], i])) <= (angle + 1)):
                                            predictX, predictY = pool_table[2], i;
                                            break;
                    else:
                        predictX = x; predictY = y;

                # Plot the Cue Stick Aim Prediction on the current frame
                if (len(GB)):
                    cv2.line(frame, (CS_impression[CB_hit_zones[0]][0], CS_impression[CB_hit_zones[0]][1]), (GB[0][0], GB[0][1]), (138, 180, 0), 3);
                    if (predictX is not None):
                        cv2.line(frame, (GB[0][0], GB[0][1]), (predictX, predictY), (138, 180, 0), 3);
                    cv2.circle(frame, (GB[0][0], GB[0][1]), 22, (138, 180, 0), -1);

        # Process the frame for the Path Prediction of Cue Ball and Object Ball along with Potting Prediction
        # The Prediction process starts only after the Cue Ball is hit and ends after the Object Ball is hit
        if (cue_ball_hit and not obj_ball_hit and cue_stick is not None):

            # Predict for max of 3 paths
            # The overall path is cached in the list called GB along with the vicinity details
            GB.clear();
            while (len(GB) != 3):
                
                # Check if GB is empty and if empty cache x values and y values of Cue Ball points
                if (len(GB) == 0):
                    Xpos.clear();
                    Ypos.clear();
                    for i in cpos:
                        Xpos.append(i[0]); Ypos.append(i[1]);

                    # Obtain a single path and cache it in GB
                    if (len(OB)):
                        GB.append(predict_direct_path(path = GB, point1 = [cpos[0]], point2 = OB, boundary = pool_table, pockets = table_pockets, X = Xpos, Y = Ypos));
                    else:
                        GB.append(predict_direct_path(path = GB, point1 = [cpos[0]], point2 = [cpos[-1]], boundary = pool_table, pockets = table_pockets, X = Xpos, Y = Ypos));

                # If GB is not empty
                else:
                    Xpos.clear();
                    Ypos.clear();

                    # Check if the current path ends around Object Ball
                    if (GB[-1][1] == 'OB'):
                        Xpos.append(GB[-1][0][0]); Ypos.append(GB[-1][0][1]);
                        Xpos.append(OB[0][0]); Ypos.append(OB[0][1]);

                        # Obtain a single path from the end point of the current path and cache it in GB
                        GB.append(predict_direct_path(path = GB, point1 = [GB[-1][0]], point2 = OB, boundary = pool_table, pockets = table_pockets, X = Xpos, Y = Ypos));

                    # Check if the current path ends around the border
                    elif (GB[-1][1] == 'border'):

                        # Check if there is no path cached previously other than the current path
                        if (len(GB) == 1):

                            # Find the reflected point using Cue Ball as incidence point and the end point of the current path as reflection point
                            reflected_point = find_reflected_point(point1 = [GB[-1][0]], point2 = [cpos[0]], boundary = pool_table);

                        # If there is a path cached previously
                        else:
                            
                            # Check if previously cached path is around Object Ball
                            if (GB[-2][1] == 'OB'):

                                # Check if Object Ball is touching the boundary
                                if (abs(OB[0][0] - pool_table[0]) < 25 or abs(OB[0][0] - pool_table[2]) < 25 or abs(OB[0][1] - pool_table[1]) < 25 or abs(OB[0][1] - pool_table[3]) < 25):
                                    # Find the reflected point using Object Ball as incidence point and the end point of the current path as reflection point
                                    reflected_point = find_reflected_point(point1 = [GB[-1][0]], point2 = OB, boundary = pool_table);

                                # If Object Ball is not touching the boundary
                                else:
                                    # Find the reflected point using Cue Ball as incidence point and the end point of the current path as reflection point
                                    reflected_point = find_reflected_point(point1 = [GB[-1][0]], point2 = [cpos[0]], boundary = pool_table);

                            # Check if previously cached path is around the border
                            elif (GB[-2][1] == 'border'):
                                # Find the reflected point using the previously cached path end point as incidence point and current path end point as reflection point
                                reflected_point = find_reflected_point(point1 = [GB[-1][0]], point2 = [GB[-2][0]], boundary = pool_table);

                        # Obtain a single path from the current path end point and reflected point
                        Xpos.append(GB[-1][0][0]); Ypos.append(GB[-1][0][1]);
                        Xpos.append(reflected_point[0][0]); Ypos.append(reflected_point[0][1]);
                        GB.append(predict_direct_path(path = GB, point1 = [GB[-1][0]], point2 = reflected_point, boundary = pool_table, pockets = table_pockets, X = Xpos, Y = Ypos));

                        # Check if current path ends around the table pocket
                        if (GB[-1][1] == 'pot'):

                            # Assign predict to be 'IN' and exit the path prediction loop
                            predict = 'IN';
                            break;

                    # Check if current path ends around the table pocket
                    elif (GB[-1][1] == 'pot'):

                        # Assign predict to be 'IN' and exit the path prediction loop
                        predict = 'IN';
                        break;

            # If the loop is completed without exiting mid-way
            else:

                # Assign predict to be 'OUT'
                predict = 'OUT';

            # Check if GB is not empty
            if (len(GB)):

                # Assign path_start_point with Cue Ball Co-ordinate for plotting
                path_start_point = [cpos[0]];

                # Plot the overall predicted path
                plot_path(img = frame, initial_point = [cpos[0]], path = GB, prediction = predict);                    


        # Check if Object Ball is hit
        if (obj_ball_hit):

            # Check if GB is not empty
            if (len(GB)):

                # Plot the overall predicted path
                plot_path(img = frame, initial_point = path_start_point, path = GB, prediction = predict);                    

        # Clear Cue Stick contour segmentation points
        cue_stick = None;

        # Clear Pocket Co-ordinates
        table_pockets.clear();

        # Clear Pool Table Co-ordinates
        pool_table = None;
        cv2.imshow('v', frame);
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break;
        else:
            continue;
    else:
        break;

cap.release();
cv2.destroyAllWindows();
