import pyrealsense2 as rs
import json
import os

now_path = os.path.dirname(os.path.realpath(__file__))
#config camera
jsonObj = json.load(open(os.path.join(now_path,'dep.json')))
json_string = str(jsonObj).replace("'",'\"')

pipeline = rs.pipeline()
config = rs.config()

W,H,freq = 1280,720,30
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, freq)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, freq)


#filter

decimation = rs.decimation_filter()
decimation.set_option(rs.option.filter_magnitude,4)
# filtered_frame = decimation.process(frame)
# colorizer = rs.colorizer()
# colorized_depth = np.asanyarray(colorizer.colorize(filtered_depth).get_data())

spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude,5) # 1-5
spatial.set_option(rs.option.filter_smooth_alpha,1) # 0.25-1
spatial.set_option(rs.option.filter_smooth_delta,50) # 1-50
spatial.set_option(rs.option.holes_fill,5) # 0-5

temporal = rs.temporal_filter()
temporal.set_option(rs.option.filter_smooth_alpha,1) # 0-1
temporal.set_option(rs.option.filter_smooth_delta,50) # 1-100

hole = rs.hole_filling_filter()

print('Done config Realsense filter, let get started')

# def read_last_line(file_path):
#     if not os.path.exists(file_path):
#         return ''
    
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#         if lines:
#             return lines[-1].strip()
#         else:
#             return ''

# def write_to_file(file_path, value):
#     with open(file_path, 'a') as file:
#         file.write(value + '\n')

# # Example usage:
# file_path = 'call_time.txt'  # Replace with the desired file path

# last_line = read_last_line(file_path)
# if last_line:
#     extracted_value = int(last_line)  # Assuming the value is an integer
#     new_value = extracted_value + 1
# else:
#     new_value = 1

# write_to_file(file_path, str(new_value))