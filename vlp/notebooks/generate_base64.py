import os, json
import base64
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
output_img_tsv_dir = '/data/yingshac/MMMHQA/base64'
output_hw_tsv_dir = '/data/yingshac/MMMHQA/base64'

img_tsv_file = os.path.join(output_img_tsv_dir, 'neg_img.tsv')
hw_tsv_file = os.path.join(output_hw_tsv_dir, 'neg_hw.tsv')
if os.path.exists(img_tsv_file): 
    #raise ValueError("path exist, are you sure to overwrite?")
    os.remove(os.path.join(output_img_tsv_dir, 'neg_img.tsv'))
if os.path.exists(hw_tsv_file): 
    #raise ValueError("path exist, are you sure to overwrite?")
    os.remove(os.path.join(output_hw_tsv_dir, 'neg_hw.tsv'))

folder = '/data/yingshac/MMMHQA/distractors'
print("folder = ", folder)
i = 0
with open(img_tsv_file, 'w') as fp, open(hw_tsv_file, 'w') as fphw:
    for im in sorted(os.listdir(folder)):
        if i%1000 == 999: print(i)
        jpg_file = open(os.path.join(folder, im), 'rb')
        jpg_img = jpg_file.read()
        image64 = base64.b64encode(jpg_img).decode('ascii')

        #row = row.append({0: i, 1:json.dumps(str({"image_id": i})), 2: image64}, ignore_index=True)
        row = [i, json.dumps({"image_id": im.split('.')[0]}), image64]
        v = '{0}\n'.format('\t'.join(map(str, row)))
        fp.write(v)
            
        PIL_img = Image.open(jpg_file).convert('RGB')
        #hw_row = hw_row.append({0: i, 1:json.dumps(str([{'height': PIL_img.height, 'width': PIL_img.width}]))}, ignore_index=True)
        hw_row = [i, json.dumps([{'height': PIL_img.height, 'width': PIL_img.width}])]
        v = '{0}\n'.format('\t'.join(map(str, hw_row)))
        fphw.write(v)
        i += 1