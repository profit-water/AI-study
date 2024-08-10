import argparse
import csv
import re

def parse_log_file(input, output):
    _start_parsing = False
    # 정규 표현식을 사용하여 로그 메시지에서 필요한 정보 추출
    log_pattern = re.compile(r"CUDA : (\d+) --- Token ID : (\S+) --- Component : \s+(\d+\/\/)?(\S+) --- Start_time : (\d{4}-\d{2}-\d{2}\s+\d{2}):(\d{2}):(\d{2}\.\d{6}) --- End_time : (\d{4}-\d{2}-\d{2}\s+\d{2}):(\d{2}):(\d{2}\.\d{6}) --- Latency\(ms\) : ([\d\.\deE+-]+)")

    # CSV 파일의 헤더
    csv_header = ['CUDA', 'Token ID', 'layer_idx', 'Component', 'Start_time', 'End_time', 'Latency(ms)']

    # CSV 파일 작성
    with open(output, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header)
        
        # 로그 파일 읽기
        with open(input, mode='r') as log_file:
            for line in log_file:
                if "ignore the logs before this line - Warmup state" in line:
                    _start_parsing = True
                    continue
                if _start_parsing:
                    match = log_pattern.search(line)
                    if match:
                        cuda = match.group(1)
                        token_id = match.group(2)
                        can_layer_idx = match.group(3)
                        if can_layer_idx:
                            layer_idx = can_layer_idx.split('//')[0]
                        else:
                            layer_idx = "-"
                        component = match.group(4)
                        start_time = int(match.group(6))*100 + float(match.group(7))
                        end_time = int(match.group(9))*100 + float(match.group(10))
                        latency = float(match.group(11)) * 1000
                        csv_writer.writerow([cuda, token_id, layer_idx, component, start_time, end_time, latency])

    print(f'Parsed log has been saved to {output}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse log file and save to CSV')
    parser.add_argument('--input', type=str, help='Path to the input log file')
    parser.add_argument('--output', type=str, help='Path to the output CSV file')

    args = parser.parse_args()
    
    parse_log_file(args.input, args.output)
