#load modules
from PIL import Image
from PIL.ExifTags import TAGS

import os
import pandas as pd # CSV handling with operations on tabular data.

column_names = [
        'filename','ExifVersion', 'ExifImageWidth', 'ExifImageHeight', 'ExifInteroperabilityOffset', 
        'SceneCaptureType', 'MeteringMode', 'LightSource', 'Flash', 'ColorSpace', 'ImageWidth', 
        'ImageLength', 'SensingMethod', 'Make', 'ExposureProgram', 'ISOSpeedRatings', 'ResolutionUnit', 'Exception_num', 
        'Exception_string', 'label'
        ]
my_df = pd.DataFrame(columns = column_names)

# Read Data
directorpaths =[f'./Mal/', f'./Ben_1/', f'./Ben_2/']

# Read Metadata
for directorypath in directorpaths:
    label_col = 0
    dir_list = os.listdir(directorypath)
    if directorypath.find('Mal') != -1:
        label_col = 1
    for imagename in dir_list:
        if imagename.endswith(".jpg") or imagename.endswith(".jpeg") or True:
            ExifVersion_col = -1
            ExifImageWidth_col = -1 
            ExifImageHeight_col = -1
            ExifInteroperabilityOffset_col = -1
            SceneCaptureType_col = -1
            MeteringMode_col = -1
            LightSource_col = -1
            Flash_col = -1
            ColorSpace_col = -1
            ImageWidth_col = -1
            ImageLength_col = -1
            SensingMethod_col = -1
            Make_col = 'N\A'
            ExposureProgram_col = -1
            ISOSpeedRatings_col = -1
            ResolutionUnit_col = -1
            Exception_num_col = 0
            Exception_string_col = []
            # read the image data using PIL
            try:
                image = Image.open(f'{directorypath}{imagename}')

                # extract EXIF data
                exifdata = image.getexif()

                # iterating over all EXIF data fields
                for tag_id in exifdata:
                    # get the tag name, instead of human unreadable tag id
                    tag = TAGS.get(tag_id, tag_id)
                    data = exifdata.get(tag_id)
                    # decode bytes 
                    if isinstance(data, bytes):
                        try:
                            data = data.decode()
                        except Exception as e:
                            Exception_num_col += 1 
                            Exception_string_col.append(str(e))
                            continue 
            
                    if 'ExifVersion' == tag:
                        ExifVersion_col = data
                    elif 'ExifImageWidth'  == tag:
                        ExifImageWidth_col = data
                    elif 'ExifImageHeight'  == tag:
                        ExifImageHeight_col = data
                    elif 'ExifInteroperabilityOffset'  == tag:
                        ExifInteroperabilityOffset_col = data
                    elif 'SceneCaptureType'  == tag:
                        SceneCaptureType_col = data
                    elif 'MeteringMode'  == tag:
                        MeteringMode_col = data
                    elif 'LightSource'  == tag:
                        LightSource_col = data
                    elif 'Flash'  == tag:
                        Flash_col = data
                    elif 'ColorSpace'  == tag:
                        ColorSpace_col = data
                    elif 'ImageWidth'  == tag:
                        ImageWidth_col = data
                    elif 'ImageLength'  == tag:
                        ImageLength_col = data
                    elif 'SensingMethod'  == tag:
                        SensingMethod_col = data
                    elif 'Make'  == tag:
                        Make_col = data
                    elif 'ExposureProgram'  == tag:
                        ExposureProgram_col = data
                    elif 'ISOSpeedRatings'  == tag:
                        ISOSpeedRatings_col = data
                    elif 'ResolutionUnit'  == tag:
                        ResolutionUnit_col = data
            except Exception as e:
                Exception_num_col += 1 
                Exception_string_col.append(str(e))

            new_row = {'filename': imagename, 'ExifVersion': ExifVersion_col, 'ExifImageWidth': ExifImageWidth_col, 'ExifImageHeight': ExifImageHeight_col, 
            'ExifInteroperabilityOffset': ExifInteroperabilityOffset_col, 'SceneCaptureType': SceneCaptureType_col, 'MeteringMode': MeteringMode_col, 
            'LightSource': LightSource_col, 'Flash': Flash_col, 'ColorSpace': ColorSpace_col, 'ImageWidth': ImageWidth_col, 'ImageLength': ImageLength_col, 
            'SensingMethod': SensingMethod_col, 'Make': Make_col, 'ExposureProgram': ExposureProgram_col, 'ISOSpeedRatings': ISOSpeedRatings_col, 
            'ResolutionUnit': ResolutionUnit_col, 'Exception_num': Exception_num_col, 'Exception_string': Exception_string_col, 'label': label_col
            }
            my_df = my_df.append(new_row, ignore_index=True)

my_df.to_csv('metadata.csv', index=False)