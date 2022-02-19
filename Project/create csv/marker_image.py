import os
import pandas as pd # CSV handling with operations on tabular data.
from struct import unpack
from tkinter import Tk, Canvas

marker_mapping = {
    0xFFD8: "Start of Image",
    0xFFE0: "Application Default Header",
    0xFFDB: "Quantization Table",
    0xFFC4: "Huffman Table",
    0xFFDA: "Start of Scan",
    0xFFD9: "End of Image",
    0xFFCC:	"Define Arithmetic Coding",	
    0xFFDC:	"Define Number of Lines",
    0xFFDD:	"Define Restart Interval",
    0xFFDE:	"Define Hierarchical Progression",
    0xFFDF:	"Expand Reference Component",
    0xFFFE: "Comment",
    0xFF01: "For temporary use",

    0xFFC0: "Start of Frame 0",
    0xFFC1: "Start of Frame 1",
    0xFFC2:	"Start of Frame 2",
    0xFFC3: "Start of Frame 3",	
    0xFFC5:	"Start of Frame 5",	
    0xFFC6:	"Start of Frame 6",	
    0xFFC7:	"Start of Frame 7",
    0xFFC9:	"Start of Frame 9",
    0xFFCA:	"Start of Frame 10",
    0xFFCB:	"Start of Frame 11",
    0xFFCD:	"Start of Frame 13",
    0xFFCE:	"Start of Frame 14",
    0xFFCF:	"Start of Frame 15",

    0xFFC8: "JPEG Extensions",
    0xFFF0: "JPEG Extension 0",
    0xFFF1: "JPEG Extension 1",
    0xFFF2: "JPEG Extension 2",
    0xFFF3: "JPEG Extension 3",
    0xFFF4: "JPEG Extension 4",
    0xFFF5: "JPEG Extension 5",
    0xFFF6: "JPEG Extension 6",
    0xFFF7: "JPEG Extension 7",
    0xFFF8: "JPEG Extension 8",	
    0xFFF9: "JPEG Extension 9",	
    0xFFFA: "JPEG Extension 10",	
    0xFFFB: "JPEG Extension 11",	
    0xFFFC: "JPEG Extension 12",	
    0xFFFD: "JPEG Extension 13",

    0xFFD0:	"Restart Marker 0",	
    0xFFD1:	"Restart Marker 1",	
    0xFFD2:	"Restart Marker 2",	
    0xFFD3:	"Restart Marker 3",	
    0xFFD4:	"Restart Marker 4",	
    0xFFD5:	"Restart Marker 5",	
    0xFFD6:	"Restart Marker 6",	
    0xFFD7:	"Restart Marker 7",

    # 0xFFE0:	"Application Segment 0",  #0xFFE0: "Application Default Header",
    0xFFE1:	"Application Segment 1",
    0xFFE2:	"Application Segment 2",
    0xFFE3:	"Application Segment 3",
    0xFFE4:	"Application Segment 4",	
    0xFFE5:	"Application Segment 5",
    0xFFE6:	"Application Segment 6",	
    0xFFE7:	"Application Segment 7",	
    0xFFE8:	"Application Segment 8",	
    0xFFE9:	"Application Segment 9",	
    0xFFEA:	"Application Segment 10",	
    0xFFEB:	"Application Segment 11",	
    0xFFEC:	"Application Segment 12",	
    0xFFED:	"Application Segment 13",	
    0xFFEE:	"Application Segment 14",	
    0xFFEF:	"Application Segment 15",	
}

def RemoveFF00(data):
    """
    Removes 0x00 after 0xff in the image scan section of JPEG
    """
    i = 0
    while True:
        b, bnext = unpack("BB", data[i : i + 2])
        if b == 0xFF:
            if bnext != 0:
                break
            i += 2
        else:
            i += 1
        if i == len(data)-2 or i == len(data)-1:
            break
    return i

class JPEG:
    def __init__(self, image_file):
        self.height = 0
        self.width = 0

        # MalJPEG
        self.Marker_EOI_content_after_num = 0   # Numbers of bytes after the EOI (end of file) marker.
        self.File_markers_num = 0			    # Total number of markers found in the file.
        self.File_size	= 0					    # Image file size in bytes.
        self.Marker_APP1_size_max = 0		    # Maximal APP1 marker size found in the file.
        self.Marker_APP12_size_max = 0			# Maximal APP12 marker size found in the file.
        self.Marker_COM_size_max = 0 			# Maximal COM marker size found in the file.
        self.Marker_DHT_num = 0					# Number of DHT markers found in the file.
        self.Marker_DHT_size_max = 0			# Maximal DHT marker size found in the file.
        self.Marker_DQT_num = 0					# Number of DQT markers found in the file.
        self.Marker_DQT_size_max = 0			# Maximal DQT marker size found in the file.

        # my
        self.Marker_APP_other_size_max = 0		# Maximal APP0-12 markers size found in the file.
        self.Marker_APP_other_num = 0		    # Number of APP0-12 markers found in the file.
        self.Marker_SOF_num = 0					# Number of SOF markers found in the file.
        self.Marker_SOF_size_max = 0			# Maximal SOF marker size found in the file.
        self.Marker_DNL_num = 0                 # Number of number of lines markers found in the file.
        self.Marker_DRI_num = 0                 # Number of restart Interval markers found in the file.
        self.Marker_EXP_num = 0                 # Number of expand reference image(s) markers found in the file.
        self.Marker_JPG_num = 0                 # Number of JPEG extensions markers found in the file.
        self.Marker_RST_num = 0                 # Number of Restart Marker markers found in the file.
        self.Marker_TEM_num = 0                 # Number of For temporary use markers found in the file.
        self.Marker_DAC_num = 0                 # Number of Arithmetic conditioning table(s) markers found in the file.
        self.Marker_DHP_num = 0                 # Number of Define hierarchical progression markers found in the file.
        self.Marker_SOS_len = 0                 # Length of Start of Scan segment marker found in the file.
        self.Marker_EOI = 0                     # Flag for end of file marker found in the file.
        self.Exception_flag = 0
        with open(image_file, 'rb') as f:
            self.img_data = f.read()
        
        self.File_size = os.path.getsize(image_file)
    
    def decodeHuffman(self, len_chunk):
        self.Marker_DHT_num += 1
        if self.Marker_DHT_size_max < len_chunk - 4:
            self.Marker_DHT_size_max = len_chunk - 4
    
    def DefineQuantizationTables(self, data, len_chunk):
        self.Marker_DQT_num += 1
        if self.Marker_DQT_size_max < len_chunk - 2:
            self.Marker_DQT_size_max = len_chunk - 2
        _, = unpack("B",data[0:1])
        data = data[65:]

    def DefineApplicationSegments(self, marker, len_chunk):
        if marker  == 0xFFE1:
            if (len_chunk - 2) > self.Marker_APP1_size_max:
                self.Marker_APP1_size_max = len_chunk - 2
        elif marker == 0xFFEC:
            if (len_chunk - 2) > self.Marker_APP12_size_max:
                self.Marker_APP12_size_max = len_chunk - 2
        else:
            self.Marker_APP_other_num += 1
            if (len_chunk - 2) > self.Marker_APP_other_size_max:
                self.Marker_APP_other_size_max = len_chunk - 2
    
    def DefineSOFSegment(self, len_chunk):
        self.Marker_SOF_num += 1
        if self.Marker_SOF_size_max < len_chunk - 2:
            self.Marker_SOF_size_max = len_chunk - 2

    def DefineCommentSegment(self, len_chunk):
        if (len_chunk - 2) > self.Marker_COM_size_max:
            self.Marker_COM_size_max = len_chunk - 2
    
    def DefineDNL(self):
        self.Marker_DNL_num += 1

    def DefineDRI(self):
        self.Marker_DRI_num += 1

    def DefineEXP(self):
        self.Marker_EXP_num += 1

    def BaselineDCT(self, data):
        _, self.height, self.width, _ = unpack(">BHHB", data[0:6])

    def StartOfScan(self, data, hdrlen):
        lenchunk = RemoveFF00(data[hdrlen:])
        return lenchunk + hdrlen
    
    def CountBytesEOI(self, data):
        self.Marker_EOI = 1
        self.Marker_EOI_content_after_num = len(data) - 2
    
    def DefineRST(self, data):
        i = 2
        self.Marker_RST_num += 1
        while True:
            b, bnext = unpack("BB", data[i : i + 2])
            if b == 0xFF:
                if bnext >= 0xD0 and bnext <= 0xD7:
                    self.File_markers_num += 1
                    self.Marker_RST_num += 1
                elif bnext == 0xD9:
                    break
                i += 2
            else:
                i += 1
            if i == len(data)-2 or i == len(data)-1:
                break
        return i
    
    def decode(self):
        data = self.img_data
        marker, = unpack(">H", data[0:2])
        if (marker != 0xFFFE and marker != 0xFFD8):
            marker, = unpack(">H", data[-2:])
            if (marker != 0xFFD9):
                self.Marker_EOI_content_after_num = len(data)
            return -1
        last_marker, = unpack(">H", data[-2:])
        while(True):
            marker, = unpack(">H", data[0:2])
            marker_str = marker_mapping.get(marker)
            if marker_str:
                self.File_markers_num += 1
            if marker == 0xFFD8:                                                # Start of Image
                data = data[2:]
            elif marker == 0xFFD9:                                              # End of Image
                self.CountBytesEOI(data)
                break
            elif marker == 0xFFDC:                                              # Number of lines
                self.DefineDNL()
                data = data[6:]
            elif marker == 0xFFDD:                                              # Restart interval
                self.DefineDRI()
                data = data[6:]   
            elif marker == 0xFFDF:                                              # Expand reference image(s)
                self.DefineEXP()
                data = data[5:]
            elif marker == 0xFFC8 or (marker >= 0xFFF0 and marker <= 0xFFED):   # JPEG extensions
                self.Marker_JPG_num += 1
                data = data[2:]
            elif marker >= 0xFFD0 and marker <= 0xFFD7:                         # Restart Marker
                len_chunk = self.DefineRST(data)
                data = data[len_chunk:]
            elif marker == 0xFF01:                                              # For temporary use
                self.Marker_TEM_num += 1
                data = data[2:]
            else:
                try:
                    len_chunk, = unpack(">H", data[2:4])
                    len_chunk += 2
                    chunk = data[4:len_chunk]
                except:
                    if (last_marker == 0xFFD9):
                        self.Exception_flag += 1
                        return -2
                    break

                if marker == 0xFFC4:                                    # Huffman Table
                    self.decodeHuffman(len_chunk)
                elif marker == 0xFFDB:                                  # Quantization Table
                    self.DefineQuantizationTables(chunk, len_chunk)
                elif marker >= 0xFFE0 and marker <= 0xFFEF:             # Application Segment
                    self.DefineApplicationSegments(marker, len_chunk)
                elif marker == 0xFFFE:                                  # Comment
                    self.DefineCommentSegment(len_chunk)
                elif (marker >= 0xFFC0 and marker <= 0xFFC7) or (marker >= 0xFFC9 and marker <= 0xFFCF):             # Start of Frame
                    self.BaselineDCT(chunk)
                    self.DefineSOFSegment(len_chunk)
                elif marker == 0xFFDA:                                  # Start of Scan
                    len_chunk = self.StartOfScan(data, len_chunk)
                    self.Marker_SOS_len = len_chunk - 2
                elif marker == 0xFFCC:                                  # Arithmetic conditioning table(s)
                    self.Marker_DAC_num += 1
                elif marker == 0xFFDE:                                  # Define hierarchical progression
                    self.Marker_DHP_num += 1

                data = data[len_chunk:]            
            if len(data)==0:
                break 
        return 1     

if __name__ == "__main__":

    column_names = [
        'filename', 'Marker_EOI_content_after_num', 'File_markers_num', 'File_size', 'Marker_APP1_size_max', 'Marker_APP12_size_max', 
        'Marker_COM_size_max', 'Marker_DHT_num', 'Marker_DHT_size_max', 'Marker_DQT_num', 'Marker_DQT_size_max', 
        'Marker_SOF_num', 'Marker_SOF_size_max', 'Marker_DNL_num', 'Marker_DRI_num',  'Marker_EXP_num',  
        'Marker_JPG_num', 'Marker_RST_num', 'Marker_TEM_num', 'Marker_DAC_num',
        'Marker_DHP_num', 'Marker_SOS_len', 'height', 'width', 'Marker_APP_other_size_max', 'Marker_APP_other_num',
        'Marker_EOI', 'Exception_flag', 'label'
        ]
    my_df = pd.DataFrame(columns = column_names)

    # Read Data
    directorpaths =[f'./Mal/', f'./Ben_1/', f'./Ben_2/']
    #directorpaths =[f'./Mal/']

    # Read Markers
    for directorypath in directorpaths:
        label_col = 0
        dir_list = os.listdir(directorypath)
        if directorypath.find('Mal') != -1:
            label_col = 1
        for imagename in dir_list:
            img = JPEG(f'{directorypath}{imagename}')
            problem_image = img.decode()
            if (problem_image == -1 and label_col!=1):
                print(imagename)
            if (problem_image == -2):
                print(imagename)
            new_row = {'filename': imagename, 'Marker_EOI_content_after_num': img.Marker_EOI_content_after_num, 'File_markers_num': img.File_markers_num, 
            'File_size': img.File_size, 'Marker_APP1_size_max': img.Marker_APP1_size_max, 'Marker_APP12_size_max': img.Marker_APP12_size_max, 
            'Marker_COM_size_max': img.Marker_COM_size_max, 'Marker_DHT_num': img.Marker_DHT_num, 'Marker_DHT_size_max': img.Marker_DHT_size_max, 
            'Marker_DQT_num': img.Marker_DQT_num, 'Marker_DQT_size_max': img.Marker_DQT_size_max, 'Marker_SOF_num': img.Marker_SOF_num, 
            'Marker_SOF_size_max': img.Marker_SOF_size_max, 'Marker_DNL_num': img.Marker_DNL_num, 'Marker_DRI_num': img.Marker_DRI_num,
            'Marker_EXP_num': img.Marker_EXP_num, 'Marker_JPG_num': img.Marker_JPG_num, 'Marker_RST_num': img.Marker_RST_num, 
            'Marker_TEM_num': img.Marker_TEM_num, 'Marker_DAC_num': img.Marker_DAC_num, 'Marker_DHP_num': img.Marker_DHP_num, 'Marker_SOS_len': img.Marker_SOS_len, 
            'height': img.height, 'width': img.width, 'Marker_APP_other_size_max': img.Marker_APP_other_size_max, 'Marker_APP_other_num': img.Marker_APP_other_num,
            'Marker_EOI': img.Marker_EOI, 'Exception_flag': img.Exception_flag, 'label': label_col
            }
            my_df = my_df.append(new_row, ignore_index=True)

    my_df.to_csv('markers_image.csv', index=False)