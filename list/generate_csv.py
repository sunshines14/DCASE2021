
def generate_csv(list_path, result_path):
    fw = open(result_path, 'w')
    fw.write('filename')
    fw.write('\t' + 'scene_label')
    fw.write('\n')
    lines = open(list_path, 'r').readlines()
    for line in lines:
        line_split = line.split('/')
        label = line_split[-1].split('-')[0]
        fw.write(line.replace('\n',''))
        fw.write('\t' + label)
        fw.write('\n')
    fw.close()    
 
if __name__ == "__main__":
    list_path = #listfile
    result_path = #csvfile
    generate_csv(list_path, result_path)